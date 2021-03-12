from model import Generator_3 as Generator, Discriminator
from model import InterpLnr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pickle
import shutil

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy

# use demo data for simplicity
# make your own validation set as needed
# validation_pt = pickle.load(open('assets/demo.pkl', "rb"))

class Solver(object):
    """Solver for training"""

    def __init__(self, data_loader_list, config):
        """Initialize configurations."""

        # Configuration
        self.config = config

        # Data loader.
        self.train_loader = data_loader_list[0]
        self.val_loader = data_loader_list[1]
        self.test_loader = data_loader_list[2]
        self.train_plot_loader = data_loader_list[3]
        self.val_plot_loader = data_loader_list[4]
        self.test_plot_loader = data_loader_list[5]

        # Training configurations.
        self.num_iters = self.config.num_iters
        self.g_lr = self.config.g_lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.resume_iters = self.config.resume_iters
        
        # Miscellaneous.
        self.name = self.config.name
        self.use_tensorboard = self.config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(self.config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(self.config.root_dir, self.config.log_dir, self.config.experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.sample_dir = os.path.join(self.config.root_dir, self.config.sample_dir, self.config.experiment)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        self.model_save_dir = os.path.join(self.config.root_dir, self.config.model_save_dir, self.config.experiment)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.best_model_dir = os.path.join(self.config.root_dir, self.config.best_model_dir, self.config.experiment)
        if not os.path.exists(self.best_model_dir):
            os.makedirs(self.best_model_dir)

        # Step size.
        self.log_step = self.config.log_step
        self.sample_step = self.config.sample_step
        self.model_save_step = self.config.model_save_step
        
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # logging
        self.min_val_loss = (None, float('inf'))

            
    def build_model(self):        
        self.G = Generator(self.config)
        self.Interp = InterpLnr(self.config)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
    
        self.G.to(self.device)
        self.Interp.to(self.device)

        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        
        
    def print_optimizer(self, opt, name):
        print(opt)
        print(name)
        
        
    def restore_model(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        if resume_iters == -1:
            G_path = os.path.join(self.best_model_dir, '{}-G-best.ckpt'.format(self.name))
            g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(g_checkpoint['model'])
        else:
            G_path = os.path.join(self.model_save_dir, '{}-G-{}.ckpt'.format(self.name, resume_iters))
            g_checkpoint = torch.load(G_path, map_location=lambda storage, loc: storage)
            self.G.load_state_dict(g_checkpoint['model'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer'])
            self.g_lr = self.g_optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)
        
                
    def train(self):
        data_iter = iter(self.train_loader)

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')
                        
        # Learning rate cache for decaying.
        g_lr = self.g_lr
        print ('Current learning rates, g_lr: {}.'.format(g_lr))
            
        # Start training.
        print('Start training...')
        print(len(self.train_loader))
        start_time = time.time()
        adv_loss = nn.CrossEntropyLoss()
        self.G = self.G.train()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(data_iter)
            
            x_real_org = x_real_org.to(self.device)
            x_real_org_filt = x_real_org_filt.to(self.device)
            x_real_org_warp = x_real_org_warp.to(self.device)
            emb_org = emb_org.to(self.device)
            len_org = len_org.to(self.device)
            f0_org = f0_org.to(self.device)
                                        
                
            # =================================================================================== #
            #                              2. Train the generator                                 #
            # =================================================================================== #

            # Identity mapping loss
            x_f0_intrp = torch.cat((x_real_org_warp, f0_org), dim=-1) # [B, T, F+1]
            x_f0_intrp = self.Interp(x_f0_intrp, len_org) # [B, T, F+1]
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0] # [B, T, 257]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1) # [B, T, F+257]
            
            x_identic, _ = self.G(x_f0_intrp_org, x_real_org_filt, emb_org)
            g_loss_id = F.mse_loss(x_real_org, x_identic) 
           
            # Backward and optimize.
            g_loss = g_loss_id
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # Logging.
            train_g_loss_id = g_loss_id.item()

            # =================================================================================== #
            #                           3. Logging and saveing checkpoints                        #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                log += ", G/train_loss_id: {:.8f}".format(train_g_loss_id)
                print(log)
                if self.use_tensorboard:
                    self.writer.add_scalar('G/train_loss_id', train_g_loss_id, i+1)

            # Plot spectrograms for training and validation data
            if (i+1) % self.sample_step == 0:
                self.plot('train', i)
                self.plot('val', i)
                        
            # Save model checkpoints and the best one if possible
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G-{}.ckpt'.format(self.name, i+1))
                torch.save({'model': self.G.state_dict(),
                            'optimizer': self.g_optimizer.state_dict()}, G_path)

                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                # save the best checkpoint so far if validation loss decreases
                val_loss = self.validate()
                print("Iteration {}, G/val_g_loss_id: {}".format(i+1, val_loss))

                if self.use_tensorboard:
                    self.writer.add_scalar('G/val_g_loss_id', val_loss, i+1)
                
                if val_loss < self.min_val_loss[1]:
                    self.min_val_loss = (i+1, val_loss)
                    G_path = os.path.join(self.model_save_dir, '{}-G-best.ckpt'.format(self.name))
                    torch.save({'model': self.G.state_dict()}, G_path)
                    print('Best checkpoint so far: Iteration {}, Validation loss: {}'.format(self.min_val_loss[0], 
                                                                                             self.min_val_loss[1]))
                # else:
                #     break

        G_path = os.path.join(self.model_save_dir, '{}-G-best.ckpt'.format(self.name))
        shutil.copy2(G_path, self.best_model_dir)
        print('Best checkpoint for model {}: Iteration {}, Validation loss: {}'.format(self.name,
                                                                                       self.min_val_loss[0], 
                                                                                       self.min_val_loss[1]))



    def test(self):
        data_iter = iter(self.test_loader)

        # Start testing from scratch or resume a checkpoint.
        if self.resume_iters:
            print('Resuming ...')
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.g_optimizer, 'G_optimizer')

        # Test the model
        self.G = self.G.eval()
        print(len(self.test_loader))
        with torch.no_grad():
            test_g_loss_id = 0
            sample_cnt = 0
            while True:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    _, x_real_org, x_real_org_filt, _, emb_org, f0_org, len_org = next(data_iter)
                except:
                    break
                
                batch_size = len(x_real_org)
                x_real_org = x_real_org.to(self.device)
                x_real_org_filt = x_real_org_filt.to(self.device)
                emb_org = emb_org.to(self.device)
                len_org = len_org.to(self.device)
                f0_org = f0_org.to(self.device)
                
                # =================================================================================== #
                #                             2. Test the generator                                   #
                # =================================================================================== #
                            
                # Identity mapping loss
                f0_org_quantized = quantize_f0_torch(f0_org)[0] # [B, T, 256]
                x_f0 = torch.cat((x_real_org, f0_org_quantized), dim=-1) # [B, T, F+256]
            
                x_identic, code_c = self.G(x_f0, x_real_org_filt, emb_org, rr=False)
                g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='sum')
            
                # log testing loss.
                test_g_loss_id += g_loss_id.item()
                sample_cnt += batch_size
            
            test_g_loss_id /= sample_cnt
            print("Iteration {}, G/test_g_loss_id: {}".format(self.resume_iters, test_g_loss_id))
            if self.use_tensorboard:
                self.writer.add_scalar('G/test_g_loss_id', test_g_loss_id, self.resume_iters)

        self.plot('test', self.resume_iters-1)

    
    def validate(self):
        data_iter = iter(self.val_loader)

        # Evaluate the model
        self.G = self.G.eval()
        print(len(self.val_loader))
        with torch.no_grad():
            val_g_loss_id = 0
            sample_cnt = 0
            while True:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    _, x_real_org, x_real_org_filt, _, emb_org, f0_org, len_org = next(data_iter)
                except:
                    break
                
                batch_size = len(x_real_org)
                x_real_org = x_real_org.to(self.device)
                x_real_org_filt = x_real_org_filt.to(self.device)
                emb_org = emb_org.to(self.device)
                len_org = len_org.to(self.device)
                f0_org = f0_org.to(self.device)
                
                # =================================================================================== #
                #                             2. Evaluate the generator                               #
                # =================================================================================== #
                            
                # Identity mapping loss
                f0_org_quantized = quantize_f0_torch(f0_org)[0] # [B, T, 256]
                x_f0 = torch.cat((x_real_org, f0_org_quantized), dim=-1) # [B, T, F+256]
            
                x_identic, code_c = self.G(x_f0, x_real_org_filt, emb_org, rr=False)
                g_loss_id = F.mse_loss(x_real_org, x_identic, reduction='sum')
            
                # log validation loss.
                val_g_loss_id += g_loss_id.item()
                sample_cnt += batch_size
            
            val_g_loss_id /= sample_cnt

        # return to training mode
        self.G = self.G.train()

        return val_g_loss_id

    
    def plot(self, mode, i):
        # Fetch fixed inputs for debugging depending on the mode.
        if mode == 'train':
            data_iter = iter(self.train_plot_loader)
        elif mode == 'val':
            data_iter = iter(self.val_plot_loader)
        elif mode == 'test':
            data_iter = iter(self.test_plot_loader)
        else:
            raise ValueError

        # plot samples

        self.G = self.G.eval()
        
        with torch.no_grad():
            while True:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    spk_id_org, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(data_iter)
                except:
                    break

                x_real_org = x_real_org.to(self.device)
                x_real_org_filt = x_real_org_filt.to(self.device)
                x_real_org_warp = x_real_org_warp.to(self.device)
                emb_org = emb_org.to(self.device)
                len_org = len_org.to(self.device)
                f0_org = f0_org.to(self.device)
                
                        
                # =================================================================================== #
                #                             2. Evaluate the generator                               #
                # =================================================================================== #
                            
                # Identity mapping loss
                f0_org_quantized = quantize_f0_torch(f0_org)[0] # [B, T, 256]
                x_f0 = torch.cat((x_real_org, f0_org_quantized), dim=-1) # [B, T, F+256]
                x_f0_woF = torch.cat((x_real_org, torch.zeros_like(f0_org_quantized)), dim=-1) # [B, T, F+256]
                x_f0_woC = torch.cat((torch.zeros_like(x_real_org), f0_org_quantized), dim=-1) # [B, T, F+256]

                x_identic, _ = self.G(x_f0, x_real_org_filt, emb_org, rr=False)
                x_identic_woF, _ = self.G(x_f0_woF, x_real_org_filt, emb_org, rr=False)
                x_identic_woR, _ = self.G(x_f0, torch.zeros_like(x_real_org_filt), emb_org, rr=False)
                x_identic_woC, _ = self.G(x_f0_woC, x_real_org_filt, emb_org, rr=False)
                x_identic_woT, _ = self.G(x_f0, x_real_org_filt, torch.zeros_like(emb_org), rr=False)

                # plot output
                melsp_gd_pad = x_real_org[0].cpu().numpy().T
                melsp_out = x_identic[0].cpu().numpy().T
                melsp_woF = x_identic_woF[0].cpu().numpy().T
                melsp_woR = x_identic_woR[0].cpu().numpy().T
                melsp_woC = x_identic_woC[0].cpu().numpy().T
                melsp_woT = x_identic_woT[0].cpu().numpy().T

                min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woT]))
                max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woT]))
                
                fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6, 1, sharex=True, figsize=(12, 10))
                ax1.set_title('Original Mel-Spectrogram', fontsize=10)
                ax2.set_title('Output Mel-Spectrogram', fontsize=10)
                ax3.set_title('Output Mel-Spectrogram Without Content', fontsize=10)
                ax4.set_title('Output Mel-Spectrogram Without Rhythm', fontsize=10)
                ax5.set_title('Output Mel-Spectrogram Without Pitch', fontsize=10)
                ax6.set_title('Output Mel-Spectrogram Without Timbre', fontsize=10)
                im1 = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
                im2 = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
                im3 = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
                im4 = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
                im5 = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
                im6 = ax6.imshow(melsp_woT, aspect='auto', vmin=min_value, vmax=max_value)
                plt.savefig(f'{self.sample_dir}/{self.name}_{mode}_output_{spk_id_org[0]}_{i+1}.png', dpi=150)
                plt.close(fig)

        self.G = self.G.train()