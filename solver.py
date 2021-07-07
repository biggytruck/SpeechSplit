from model import Generator_3 as Generator
from model import Generator_6 as F_Converter
from model import InterpLnr, STLR
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from collections import OrderedDict
from utils import quantize_f0_torch, tensor2onehot


class Solver(object):
    """Solver for training"""


    def __init__(self, data_loader, config):
        """Initialize configurations."""

        # Configuration
        self.config = config

        # Data loader.
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

        # Training configurations.
        self.num_iters = self.config.num_iters
        self.lr = self.config.lr
        self.beta1 = self.config.beta1
        self.beta2 = self.config.beta2
        self.resume_iters = self.config.resume_iters
        self.mode = self.config.mode
        self.cutoff = self.config.cutoff
        
        # Miscellaneous.
        self.experiment = self.config.experiment
        self.model_name = self.config.model_name
        self.model_type = self.config.model_type
        self.use_tensorboard = self.config.use_tensorboard
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:{}'.format(self.config.device_id) if self.use_cuda else 'cpu')

        # Directories.
        self.log_dir = os.path.join(self.config.root_dir, self.config.log_dir, self.experiment)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.model_save_dir = os.path.join(self.config.root_dir, self.config.model_save_dir, self.experiment)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.sample_dir = os.path.join(self.config.root_dir, self.config.sample_dir, self.experiment)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

        # Step size.
        self.log_step = self.config.log_step
        self.sample_step = self.config.sample_step
        self.model_save_step = self.config.model_save_step
        
        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Logging
        self.min_loss_step = 0
        self.min_loss = float('inf')

            
    def build_model(self):        
        self.model = Generator(self.config) if self.model_type == 'G' else F_Converter(self.config)
        self.print_network(self.model, self.model_type)
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        self.Interp = InterpLnr(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=1e-6)
        self.Interp.to(self.device)

        self.scheduler = STLR(self.optimizer, num_iters=self.num_iters, cut_frac=0.1, ratio=32)

        
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
            ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}-best.ckpt'.format(self.model_name, self.model_type, self.cutoff))
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            try:
                self.model.load_state_dict(ckpt['model'])
            except:
                new_state_dict = OrderedDict()
                for k, v in ckpt['model'].items():
                    new_state_dict[k[7:]] = v
                self.model.load_state_dict(new_state_dict)
        else:
            ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}-{}.ckpt'.format(self.model_name, self.model_type, self.cutoff, resume_iters))
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            try:
                self.model.load_state_dict(ckpt['model'])
            except:
                new_state_dict = OrderedDict()
                for k, v in ckpt['model'].items():
                    new_state_dict[k[7:]] = v
                self.model.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.lr = self.optimizer.param_groups[0]['lr']
        
        
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.log_dir)


    def reload_data_loader(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        
                
    def train(self):
        if self.num_iters == 1:
            ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}-best.ckpt'.format(self.model_name, self.model_type, self.cutoff))
            torch.save({'model': self.model.state_dict()}, ckpt_path)
            return

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            print('Resuming ...')
            start_iters = self.resume_iters
            self.num_iters += self.resume_iters
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.optimizer, 'optimizer')
                        
        # Learning rate cache for decaying.
        lr = self.lr
        print ('Current learning rates, lr: {}.'.format(lr))
            
        # Start training.
        print('Start training...')
        start_time = time.time()
        self.model = self.model.train()
        for i in range(start_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Load data
            try:
                _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
            
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
            x_f0 = torch.cat((x_real_org_warp, f0_org), dim=-1) # [B, T, F+1]
            x_filt_org = torch.cat((x_real_org_filt[:,:,:1], x_real_org_warp), dim=-1) # [B, T, F+1]
            x_f0_intrp = self.Interp(x_f0, len_org) # [B, T, F+1]
            f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0] # [B, T, 257]
            x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1) # [B, T, F+257]
            
            if self.experiment == 'spsp1':
                if self.model_type == 'G':
                    x_identic = self.model(x_f0_intrp_org, x_real_org, emb_org)
                    loss_id = F.mse_loss(x_identic, x_real_org) 
                else:
                    x_identic = self.model(x_real_org, f0_org_intrp).view(-1, self.config.dim_f0)
                    f0_org_quantized = quantize_f0_torch(f0_org)[1].view(-1)
                    loss_id = F.cross_entropy(x_identic, f0_org_quantized)
            elif self.experiment == 'spsp2':
                if self.model_type == 'G':
                    x_identic = self.model(x_f0_intrp_org, x_real_org_filt, emb_org)
                    loss_id = F.mse_loss(x_identic, x_real_org) 
                else:
                    x_identic = self.model(x_filt_org, f0_org_intrp).view(-1, self.config.dim_f0)
                    f0_org_quantized = quantize_f0_torch(f0_org)[1].view(-1)
                    loss_id = F.cross_entropy(x_identic, f0_org_quantized)
            else:
                raise ValueError
           
            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Logging.
            train_loss_id = loss_id.item()

            # =================================================================================== #
            #                           3. Logging and saveing checkpoints                        #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                log += ", {}/train_loss_id: {:.8f}".format(self.model_type, train_loss_id)
                print(log)
                if self.use_tensorboard:
                    self.writer.add_scalar(f'{self.model_type}/train_loss_id', train_loss_id, i+1)

            # Plot spectrograms for training and validation data
            if (i+1) % self.sample_step == 0:
                if self.model_type == 'G':
                    self.plot_G(i+1)
                else:
                    self.plot_F(i+1)
    
            # Save model checkpoints and the best one if possible
            if (i+1) % self.model_save_step == 0:
                ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}-{}.ckpt'.format(self.model_name, self.model_type, self.cutoff, i+1))
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'scheduler': self.scheduler.state_dict()}, ckpt_path)

                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                if self.use_tensorboard:
                    self.writer.add_scalar(f'{self.model_type}/train_loss_id', train_loss_id, i+1)
                
                # current use training loss to find best model
                # may change to validation loss for better generalization but need to do train-val split
                if train_loss_id < self.min_loss:
                    self.min_loss = train_loss_id
                    self.min_loss_step = i+1
                    ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}-best.ckpt'.format(self.model_name, self.model_type, self.cutoff))
                    torch.save({'model': self.model.state_dict()}, ckpt_path)
                    print('Best checkpoint so far: Iteration {}, Training loss: {}'.format(self.min_loss_step, 
                                                                                           self.min_loss))

                # val_loss_id = self.validate()
                # if val_loss_id < self.min_loss:
                #     self.min_loss = val_loss_id
                #     self.min_loss_step = i+1
                #     ckpt_path = os.path.join(self.model_save_dir, '{}-{}-best.ckpt'.format(self.model_name, self.model_type))
                #     torch.save({'model': self.model.state_dict()}, ckpt_path)
                #     print('Best checkpoint so far: Iteration {}, Training loss: {}'.format(self.min_loss, 
                #                                                                            self.min_loss_step))

        print('Best checkpoint for model {}: Iteration {}, Validation loss: {}'.format(self.model_name,
                                                                                       self.min_loss_step, 
                                                                                       self.min_loss))


    def test(self):
        # Start testing from scratch or resume a checkpoint.
        if self.resume_iters:
            print('Resuming ...')
            self.restore_model(self.resume_iters)
            self.print_optimizer(self.optimizer, 'optimizer')

        # Test the model
        self.model = self.model.eval()
        with torch.no_grad():
            test_loss_id = 0
            sample_cnt = 0
            while True:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
                except:
                    break
                
                batch_size = len(x_real_org)
                x_real_org = x_real_org.to(self.device)
                x_real_org_filt = x_real_org_filt.to(self.device)
                x_real_org_warp = x_real_org_warp.to(self.device)
                emb_org = emb_org.to(self.device)
                len_org = len_org.to(self.device)
                f0_org = f0_org.to(self.device)
                
                # =================================================================================== #
                #                             2. Test the generator                                   #
                # =================================================================================== #
                            
                # Identity mapping loss
                x_filt_org = torch.cat((x_real_org_filt[:,:,:1], x_real_org_warp), dim=-1) # [B, T, F+1]
                f0_org_one_hot, f0_org_quantized = quantize_f0_torch(f0_org) # [B, T, 257], [B, T, 1]
                x_f0 = torch.cat((x_real_org_warp, f0_org_one_hot), dim=-1) # [B, T, F+257]
            
                if self.experiment == 'spsp1':
                    if self.model_type == 'G':
                        x_identic = self.model(x_f0, x_real_org, emb_org, rr=False)
                        loss_id = F.mse_loss(x_identic, x_real_org, reduction='sum')
                    else:
                        x_identic = self.model(x_real_org, f0_org_one_hot, rr=False).view(-1, self.config.dim_f0)
                        f0_org_quantized = f0_org_quantized.view(-1)
                        loss_id = F.cross_entropy(x_identic, f0_org_quantized, reduction='sum')
                elif self.experiment == 'spsp2':
                    if self.model_type == 'G':
                        x_identic = self.model(x_f0, x_real_org_filt, emb_org, rr=False)
                        loss_id = F.mse_loss(x_identic, x_real_org, reduction='sum')
                    else:
                        x_identic = self.model(x_filt_org, f0_org_one_hot, rr=False).view(-1, self.config.dim_f0)
                        f0_org_quantized = f0_org_quantized.view(-1)
                        loss_id = F.cross_entropy(x_identic, f0_org_quantized, reduction='sum')
                else:
                    raise ValueError
            
                # log testing loss.
                test_loss_id += loss_id.item()
                sample_cnt += batch_size
            
            test_loss_id /= sample_cnt
            print("Iteration {}, {}/test_loss_id: {}".format(self.resume_iters, self.model_type, test_loss_id))
            if self.use_tensorboard:
                self.writer.add_scalar(f'{self.model_type}/test_loss_id', test_loss_id, self.resume_iters)

        self.plot_G(self.resume_iters)


    def validate(self):
        # Evaluate the model
        self.model = self.model.eval()
        with torch.no_grad():
            val_loss_id = 0
            sample_cnt = 0
            while True:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    _, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
                except:
                    break
                
                batch_size = len(x_real_org)
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
                x_filt_org = torch.cat((x_real_org_filt[:,:,:1], x_real_org_warp), dim=-1) # [B, T, F+1]
                f0_org_one_hot, f0_org_quantized = quantize_f0_torch(f0_org) # [B, T, 257], [B, T, 1]
                x_f0 = torch.cat((x_real_org_warp, f0_org_one_hot), dim=-1) # [B, T, F+257]
            
                if self.experiment == 'spsp1':
                    if self.model_type == 'G':
                        x_identic = self.model(x_f0, x_real_org, emb_org, rr=False)
                        loss_id = F.mse_loss(x_identic, x_real_org, reduction='sum')
                    else:
                        x_identic = self.model(x_real_org, f0_org_one_hot, rr=False).view(-1, self.config.dim_f0)
                        f0_org_quantized = f0_org_quantized.view(-1)
                        loss_id = F.cross_entropy(x_identic, f0_org_quantized, reduction='sum')
                elif self.experiment == 'spsp2':
                    if self.model_type == 'G':
                        x_identic = self.model(x_f0, x_real_org_filt, emb_org, rr=False)
                        loss_id = F.mse_loss(x_identic, x_real_org, reduction='sum')
                    else:
                        x_identic = self.model(x_filt_org, f0_org_one_hot, rr=False).view(-1, self.config.dim_f0)
                        f0_org_quantized = f0_org_quantized.view(-1)
                        loss_id = F.cross_entropy(x_identic, f0_org_quantized, reduction='sum')
                else:
                    raise ValueError
            
                # log validation loss.
                val_loss_id += loss_id.item()
                sample_cnt += batch_size
            
            val_loss_id /= sample_cnt
            print("Iteration {}, {}/val_loss_id: {}".format(self.resume_iters, self.model_type, val_loss_id))
            if self.use_tensorboard:
                self.writer.add_scalar(f'{self.model_type}/val_loss_id', val_loss_id, self.resume_iters)

        # return to training mode
        self.model = self.model.train()

        return val_loss_id

    
    def plot_G(self, step):
        # plot samples
        self.model = self.model.eval()
        
        with torch.no_grad():

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Load data
            try:
                spk_id_org, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                spk_id_org, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)

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
            if self.experiment == 'spsp1':

                f0_org_one_hot = quantize_f0_torch(f0_org)[0] # [B, T, 257]
                x_f0 = torch.cat((x_real_org, f0_org_one_hot), dim=-1) # [B, T, F+257]
                x_f0_woF = torch.cat((x_real_org, torch.zeros_like(f0_org_one_hot)), dim=-1) # [B, T, F+257]
                x_f0_woC = torch.cat((torch.zeros_like(x_real_org), f0_org_one_hot), dim=-1) # [B, T, F+257]
                x_f0_woCF = torch.zeros_like(x_f0) # [B, T, F+257]
            
                x_identic = self.model(x_f0, x_real_org, emb_org, rr=False)
                x_identic_woF = self.model(x_f0_woF, x_real_org, emb_org, rr=False)
                x_identic_woR = self.model(x_f0, torch.zeros_like(x_real_org), emb_org, rr=False)
                x_identic_woC = self.model(x_f0_woC, x_real_org, emb_org, rr=False)
                x_identic_woT = self.model(x_f0, x_real_org, torch.zeros_like(emb_org), rr=False)
                x_identic_woCF = self.model(x_f0_woCF, x_real_org, emb_org, rr=False)
            elif self.experiment == 'spsp2':
                f0_org_one_hot = quantize_f0_torch(f0_org)[0] # [B, T, 257]
                x_f0 = torch.cat((x_real_org_warp, f0_org_one_hot), dim=-1) # [B, T, F+1+257]
                x_f0_woF = torch.cat((x_real_org_warp, torch.zeros_like(f0_org_one_hot)), dim=-1) # [B, T, F+1+257]
                x_f0_woC = torch.cat((torch.zeros_like(x_real_org_warp), f0_org_one_hot), dim=-1) # [B, T, F+1+257]
                x_f0_woCF = torch.zeros_like(x_f0) # [B, T, F+1+257]

                x_identic = self.model(x_f0, x_real_org_filt, emb_org, rr=False)
                x_identic_woF = self.model(x_f0_woF, x_real_org_filt, emb_org, rr=False)
                x_identic_woR = self.model(x_f0, torch.zeros_like(x_real_org_filt), emb_org, rr=False)
                x_identic_woC = self.model(x_f0_woC, x_real_org_filt, emb_org, rr=False)
                x_identic_woT = self.model(x_f0, x_real_org_filt, torch.zeros_like(emb_org), rr=False)
                x_identic_woCF = self.model(x_f0_woCF, x_real_org_filt, emb_org, rr=False)
            else:
                raise ValueError

            # plot input
            x_real_org_filt = x_real_org_filt[0].cpu().numpy()
            x_real_org_warp = x_real_org_warp[0].cpu().numpy()
            f0_org_one_hot = f0_org_one_hot[0].cpu().numpy()

            spenv = x_real_org_filt[:, 1:]
            dog_output = x_real_org_filt[:, :1]
            dog_output = np.repeat(dog_output, spenv.shape[1], axis=1)
            spenv = spenv.T
            dog_output = dog_output.T
            mono_spmel = x_real_org_warp.T
            pitch_contour = f0_org_one_hot.T

            min_value = np.min(np.vstack([spenv, dog_output, mono_spmel, pitch_contour]))
            max_value = np.max(np.vstack([spenv, dog_output, mono_spmel, pitch_contour]))
            
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Spectral Envelope', fontsize=10)
            ax2.set_title('DoG output', fontsize=10)
            ax3.set_title('Monotonic Mel-Spectrogram', fontsize=10)
            ax4.set_title('Pitch Contour', fontsize=10)
            _ = ax1.imshow(spenv, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(dog_output, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(mono_spmel, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(pitch_contour, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}_{self.mode}_{self.model_type}_input_{spk_id_org[0]}_{step}.png', dpi=150)
            plt.close(fig)

            # plot output
            melsp_gd_pad = x_real_org[0].cpu().numpy().T
            melsp_out = x_identic[0].cpu().numpy().T
            melsp_woF = x_identic_woF[0].cpu().numpy().T
            melsp_woR = x_identic_woR[0].cpu().numpy().T
            melsp_woC = x_identic_woC[0].cpu().numpy().T
            melsp_woT = x_identic_woT[0].cpu().numpy().T
            melsp_woCF = x_identic_woCF[0].cpu().numpy().T

            min_value = np.min(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woT, melsp_woCF]))
            max_value = np.max(np.hstack([melsp_gd_pad, melsp_out, melsp_woF, melsp_woR, melsp_woC, melsp_woT, melsp_woCF]))
            
            fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Original Mel-Spectrogram', fontsize=10)
            ax2.set_title('Output Mel-Spectrogram', fontsize=10)
            ax3.set_title('Output Mel-Spectrogram Without Content', fontsize=10)
            ax4.set_title('Output Mel-Spectrogram Without Rhythm', fontsize=10)
            ax5.set_title('Output Mel-Spectrogram Without Pitch', fontsize=10)
            ax6.set_title('Output Mel-Spectrogram Without Timbre', fontsize=10)
            ax7.set_title('Output Mel-Spectrogram Without Content And Pitch', fontsize=10)
            _ = ax1.imshow(melsp_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(melsp_out, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(melsp_woC, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(melsp_woR, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax5.imshow(melsp_woF, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax6.imshow(melsp_woT, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax7.imshow(melsp_woCF, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}_{self.mode}_{self.model_type}_output_{spk_id_org[0]}_{step}.png', dpi=150)
            plt.close(fig)

        self.model = self.model.train()


    def plot_F(self, step):
        # plot samples
        self.model = self.model.eval()
        
        with torch.no_grad():

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Load data
            try:
                spk_id_org, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                spk_id_org, x_real_org, x_real_org_filt, x_real_org_warp, emb_org, f0_org, len_org = next(self.data_iter)

            x_real_org = x_real_org.to(self.device)
            x_real_org_filt = x_real_org_filt.to(self.device)
            x_real_org_warp = x_real_org_warp.to(self.device)
            f0_org = f0_org.to(self.device)

            f0_org_one_hot = quantize_f0_torch(f0_org)[0] # [B, T, 257]

            # =================================================================================== #
            #                             2. Evaluate the generator                               #
            # =================================================================================== #
                        
            # Identity mapping loss
            if self.experiment == 'spsp1':
                f0_identic = self.model(x_real_org, f0_org_one_hot, rr=False)
                f0_identic_woR = self.model(torch.zeros_like(x_real_org), f0_org_one_hot, rr=False)
                f0_identic_woF = self.model(x_real_org, torch.zeros_like(f0_org_one_hot), rr=False)
            elif self.experiment == 'spsp2':
                x_filt_org = torch.cat((x_real_org_filt[:,:,:1], x_real_org_warp), dim=-1) # [B, T, F+1]

                f0_identic = self.model(x_filt_org, f0_org_one_hot, rr=False)
                f0_identic_woR = self.model(torch.zeros_like(x_filt_org), f0_org_one_hot, rr=False)
                f0_identic_woF = self.model(x_filt_org, torch.zeros_like(f0_org_one_hot), rr=False)
            else:
                raise ValueError

            # plot input
            x_real_org_filt = x_real_org_filt[0].cpu().numpy()
            x_real_org_warp = x_real_org_warp[0].cpu().numpy()
            f0_org_one_hot = f0_org_one_hot[0].cpu().numpy()

            spenv = x_real_org_filt[:, 1:]
            dog_output = x_real_org_filt[:, :1]
            dog_output = np.repeat(dog_output, spenv.shape[1], axis=1)
            dog_output = dog_output.T
            mono_spmel = x_real_org_warp.T
            pitch_contour = f0_org_one_hot.T

            min_value = np.min(np.vstack([dog_output, mono_spmel, pitch_contour]))
            max_value = np.max(np.vstack([dog_output, mono_spmel, pitch_contour]))
            
            fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('DoG output', fontsize=10)
            ax2.set_title('Monotonic Mel-Spectrogram', fontsize=10)
            ax3.set_title('Pitch Contour', fontsize=10)
            _ = ax1.imshow(dog_output, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(mono_spmel, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(pitch_contour, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}_{self.mode}_{self.model_type}_input_{spk_id_org[0]}_{step}.png', dpi=150)
            plt.close(fig)

            # plot output
            f0_gd_pad = f0_org_one_hot.T
            f0_out = tensor2onehot(f0_identic)[0].cpu().numpy().T
            f0_woR = tensor2onehot(f0_identic_woR)[0].cpu().numpy().T
            f0_woF = tensor2onehot(f0_identic_woF)[0].cpu().numpy().T

            min_value = np.min(np.hstack([f0_gd_pad, f0_out, f0_woR, f0_woF]))
            max_value = np.max(np.hstack([f0_gd_pad, f0_out, f0_woR, f0_woF]))
            
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Original Pitch Contour', fontsize=10)
            ax2.set_title('Output Pitch Contour', fontsize=10)
            ax3.set_title('Output Pitch Contour Without Rhythm', fontsize=10)
            ax4.set_title('Output Pitch Contour Without Pitch', fontsize=10)
            _ = ax1.imshow(f0_gd_pad, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(f0_out, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(f0_woR, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(f0_woF, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}_{self.mode}_{self.model_type}_output_{spk_id_org[0]}_{step}.png', dpi=150)
            plt.close(fig)

        self.model = self.model.train()
