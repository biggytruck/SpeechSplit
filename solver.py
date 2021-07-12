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
        self.ckpt_save_step = self.config.ckpt_save_step
        
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
            ckpt_path = os.path.join(self.model_save_dir, '{}-{}-best.ckpt'.format(self.model_name, self.model_type))
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
                _, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                _, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
            
            spmel_gt = spmel_gt.to(self.device)
            rhythm_input = rhythm_input.to(self.device)
            content_input = content_input.to(self.device)
            pitch_input = pitch_input.to(self.device)
            timbre_input = timbre_input.to(self.device)
            len_crop = len_crop.to(self.device)

            # =================================================================================== #
            #                              2. Train the model                                     #
            # =================================================================================== #

            if self.model_type == 'G':
                # Prepare input data and apply random resampling
                content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+1]
                content_pitch_input_intrp = self.Interp(content_pitch_input, len_crop) # [B, T, F+1]
                pitch_input_intrp = quantize_f0_torch(content_pitch_input_intrp[:, :, -1])[0] # [B, T, 257]
                content_pitch_input_intrp = torch.cat((content_pitch_input_intrp[:,:,:-1], pitch_input_intrp), dim=-1) # [B, T, F+257]

                # Identity mapping loss
                spmel_output = self.model(content_pitch_input_intrp, rhythm_input, timbre_input)
                loss_id = F.mse_loss(spmel_output, spmel_gt)
            elif self.model_type == 'F':
                # Prepare input data and apply random resampling
                pitch_gt = quantize_f0_torch(pitch_input)[1].view(-1)
                pitch_input_intrp = self.Interp(pitch_input, len_crop) # [B, T, 1]
                pitch_input_intrp = quantize_f0_torch(pitch_input_intrp)[0] # [B, T, 257]

                # Cross entropy loss
                pitch_output = self.model(rhythm_input, pitch_input_intrp).view(-1, self.config.dim_f0)
                loss_id = F.cross_entropy(pitch_output, pitch_gt)
            else:
                raise ValueError

            # Backward and optimize.
            loss = loss_id
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging.
            train_loss_id = loss_id.item()

            # =================================================================================== #
            #                           3. Logging and saving checkpoints                         #
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
    
            # Save model checkpoints
            if (i+1) % self.ckpt_save_step == 0:
                ckpt_path = os.path.join(self.model_save_dir, '{}-{}-{}.ckpt'.format(self.model_name, self.model_type, i+1))
                torch.save({'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}, ckpt_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))
                if self.use_tensorboard:
                    self.writer.add_scalar(f'{self.model_type}/train_loss_id', train_loss_id, i+1)

            # Save the best model if possible
            if (i+1) % self.model_save_step == 0:
                # currently use training loss to find best model
                # may change to validation loss for better generalization but need to do train-val split
                if train_loss_id < self.min_loss:
                    self.min_loss = train_loss_id
                    self.min_loss_step = i+1
                    ckpt_path = os.path.join(self.model_save_dir, '{}-{}-best.ckpt'.format(self.model_name, self.model_type))
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

                # Load data
                try:
                    _, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
                except:
                    break
                
                batch_size = len(spmel_gt)
                spmel_gt = spmel_gt.to(self.device)
                rhythm_input = rhythm_input.to(self.device)
                content_input = content_input.to(self.device)
                pitch_input = pitch_input.to(self.device)
                timbre_input = timbre_input.to(self.device)
                len_crop = len_crop.to(self.device)
                
                # =================================================================================== #
                #                              2. Test the model                                      #
                # =================================================================================== #

                if self.model_type == 'G':
                    # Prepare input data and apply random resampling
                    content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+1]
                    pitch_input = quantize_f0_torch(content_pitch_input[:, :, -1])[0] # [B, T, 257]
                    content_pitch_input = torch.cat((content_pitch_input[:,:,:-1], pitch_input), dim=-1) # [B, T, F+257]

                    # Identity mapping loss
                    spmel_output = self.model(content_pitch_input, rhythm_input, timbre_input, rr=False)
                    loss_id = F.mse_loss(spmel_output, spmel_gt)
                elif self.model_type == 'F':
                    # Prepare input data and apply random resampling
                    pitch_gt = quantize_f0_torch(pitch_input)[1].view(-1)
                    pitch_input = quantize_f0_torch(pitch_input)[0] # [B, T, 257]

                    # Cross entropy loss
                    pitch_output = self.model(rhythm_input, pitch_input, rr=False).view(-1, self.config.dim_f0)
                    loss_id = F.cross_entropy(pitch_output, pitch_gt)
                else:
                    raise ValueError
            
                # log testing loss.
                test_loss_id += loss_id.item()
                sample_cnt += batch_size
            
            test_loss_id /= sample_cnt
            print("Iteration {}, {}/test_loss_id: {}".format(self.resume_iters, self.model_type, test_loss_id))
            if self.use_tensorboard:
                self.writer.add_scalar(f'{self.model_type}/test_loss_id', test_loss_id, self.resume_iters)


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

                # Load data
                try:
                    _, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
                except:
                    break
                
                batch_size = len(spmel_gt)
                spmel_gt = spmel_gt.to(self.device)
                rhythm_input = rhythm_input.to(self.device)
                content_input = content_input.to(self.device)
                pitch_input = pitch_input.to(self.device)
                timbre_input = timbre_input.to(self.device)
                len_crop = len_crop.to(self.device)
                
                # =================================================================================== #
                #                              2. Evaluate the model                                  #
                # =================================================================================== #

                if self.model_type == 'G':
                    # Prepare input data and apply random resampling
                    content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+1]
                    pitch_input = quantize_f0_torch(content_pitch_input[:, :, -1])[0] # [B, T, 257]
                    content_pitch_input = torch.cat((content_pitch_input[:,:,:-1], pitch_input), dim=-1) # [B, T, F+257]

                    # Identity mapping loss
                    spmel_output = self.model(content_pitch_input, rhythm_input, timbre_input, rr=False)
                    loss_id = F.mse_loss(spmel_output, spmel_gt)
                elif self.model_type == 'F':
                    # Prepare input data and apply random resampling
                    pitch_gt = quantize_f0_torch(pitch_input)[1].view(-1)
                    pitch_input = quantize_f0_torch(pitch_input)[0] # [B, T, 257]

                    # Cross entropy loss
                    pitch_output = self.model(rhythm_input, pitch_input, rr=False).view(-1, self.config.dim_f0)
                    loss_id = F.cross_entropy(pitch_output, pitch_gt)
                else:
                    raise ValueError
            
                # log testing loss.
                val_loss_id += loss_id.item()
                sample_cnt += batch_size
            
            val_loss_id /= sample_cnt
            print("Iteration {}, {}/test_loss_id: {}".format(self.resume_iters, self.model_type, val_loss_id))
            if self.use_tensorboard:
                self.writer.add_scalar(f'{self.model_type}/test_loss_id', val_loss_id, self.resume_iters)

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
                spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)

            spmel_gt = spmel_gt.to(self.device)
            rhythm_input = rhythm_input.to(self.device)
            content_input = content_input.to(self.device)
            pitch_input = pitch_input.to(self.device)
            timbre_input = timbre_input.to(self.device)
            len_crop = len_crop.to(self.device)
            
            # =================================================================================== #
            #                             2. Generate different outputs                           #
            # =================================================================================== #
                        
            pitch_input = quantize_f0_torch(pitch_input)[0] # [B, T, 257]
            content_pitch_input = torch.cat((content_input, pitch_input), dim=-1) # [B, T, F+257]
            content_pitch_input_woF = torch.cat((content_input, torch.zeros_like(pitch_input)), dim=-1) # [B, T, F+257]
            content_pitch_input_woC = torch.cat((torch.zeros_like(content_input), pitch_input), dim=-1) # [B, T, F+257]
            content_pitch_input_woCF = torch.zeros_like(content_pitch_input) # [B, T, F+257]
        
            spmel_output = self.model(content_pitch_input, rhythm_input, timbre_input, rr=False)
            spmel_output_woF = self.model(content_pitch_input_woF, rhythm_input, timbre_input, rr=False)
            spmel_output_woR = self.model(content_pitch_input, torch.zeros_like(rhythm_input), timbre_input, rr=False)
            spmel_output_woC = self.model(content_pitch_input_woC, rhythm_input, timbre_input, rr=False)
            spmel_output_woT = self.model(content_pitch_input, rhythm_input, torch.zeros_like(timbre_input), rr=False)
            spmel_output_woCF = self.model(content_pitch_input_woCF, rhythm_input, timbre_input, rr=False)

            # plot input
            rhythm_input = rhythm_input[0].cpu().numpy()
            content_input = content_input[0].cpu().numpy()
            pitch_input = pitch_input[0].cpu().numpy()

            spenv = rhythm_input[:, 1:]
            spmel_filt = rhythm_input[:, :1]
            spmel_filt = np.repeat(spmel_filt, spenv.shape[1], axis=1)
            spenv = spenv.T
            spmel_filt = spmel_filt.T
            spmel_mono = content_input.T
            f0 = pitch_input.T

            min_value = np.min(np.vstack([spenv, spmel_filt, spmel_mono, f0]))
            max_value = np.max(np.vstack([spenv, spmel_filt, spmel_mono, f0]))
            
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Spectral Envelope', fontsize=10)
            ax2.set_title('DoG output', fontsize=10)
            ax3.set_title('Monotonic Mel-Spectrogram', fontsize=10)
            ax4.set_title('Pitch Contour', fontsize=10)
            _ = ax1.imshow(spenv, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(spmel_filt, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(spmel_mono, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(f0, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}-{self.mode}-{self.model_type}-input-{spk_id_org[0]}-{step}.png', dpi=150)
            plt.close(fig)

            # plot output
            spmel_gt = spmel_gt[0].cpu().numpy().T
            spmel_output = spmel_output[0].cpu().numpy().T
            spmel_output_woF = spmel_output_woF[0].cpu().numpy().T
            spmel_output_woR = spmel_output_woR[0].cpu().numpy().T
            spmel_output_woC = spmel_output_woC[0].cpu().numpy().T
            spmel_output_woT = spmel_output_woT[0].cpu().numpy().T
            spmel_output_woCF = spmel_output_woCF[0].cpu().numpy().T

            min_value = np.min(np.hstack([spmel_gt, spmel_output, spmel_output_woF, spmel_output_woR, spmel_output_woC, spmel_output_woT, spmel_output_woCF]))
            max_value = np.max(np.hstack([spmel_gt, spmel_output, spmel_output_woF, spmel_output_woR, spmel_output_woC, spmel_output_woT, spmel_output_woCF]))
            
            fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(7, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Original Mel-Spectrogram', fontsize=10)
            ax2.set_title('Output Mel-Spectrogram', fontsize=10)
            ax3.set_title('Output Mel-Spectrogram Without Content', fontsize=10)
            ax4.set_title('Output Mel-Spectrogram Without Rhythm', fontsize=10)
            ax5.set_title('Output Mel-Spectrogram Without Pitch', fontsize=10)
            ax6.set_title('Output Mel-Spectrogram Without Timbre', fontsize=10)
            ax7.set_title('Output Mel-Spectrogram Without Content And Pitch', fontsize=10)
            _ = ax1.imshow(spmel_gt, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(spmel_output, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(spmel_output_woC, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(spmel_output_woR, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax5.imshow(spmel_output_woF, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax6.imshow(spmel_output_woT, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax7.imshow(spmel_output_woCF, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}-{self.mode}-{self.model_type}-output-{spk_id_org[0]}-{step}.png', dpi=150)
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
                spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)
            except:
                self.data_iter = iter(self.data_loader)
                spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = next(self.data_iter)

            rhythm_input = rhythm_input.to(self.device)
            pitch_input = pitch_input.to(self.device)
            
            # =================================================================================== #
            #                             2. Generate different outputs                           #
            # =================================================================================== #
                        
            pitch_gt = quantize_f0_torch(pitch_input)[0] # [B, T, 257]
            pitch_input = quantize_f0_torch(pitch_input)[0] # [B, T, 257]
            
            pitch_output = self.model(rhythm_input, pitch_input, rr=False)
            pitch_output_woR = self.model(torch.zeros_like(rhythm_input), pitch_input, rr=False)
            pitch_output_woF = self.model(rhythm_input, torch.zeros_like(pitch_input), rr=False)

            # plot input
            rhythm_input = rhythm_input[0].cpu().numpy()
            pitch_gt = pitch_gt[0].cpu().numpy()

            spmel_mono = rhythm_input[:, 1:]
            spmel_filt = rhythm_input[:, :1]
            spmel_filt = np.repeat(spmel_filt, spmel_mono.shape[1], axis=1)
            spmel_mono = spmel_mono.T
            spmel_filt = spmel_filt.T
            f0 = pitch_gt.T

            min_value = np.min(np.vstack([spmel_mono, spmel_filt, f0]))
            max_value = np.max(np.vstack([spmel_mono, spmel_filt, f0]))
            
            fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Monotonic Mel-Spectrogram', fontsize=10)
            ax2.set_title('DoG output', fontsize=10)
            ax3.set_title('Pitch Contour', fontsize=10)
            _ = ax1.imshow(spmel_mono, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(spmel_filt, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(f0, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}-{self.mode}-{self.model_type}-input-{spk_id_org[0]}-{step}.png', dpi=150)
            plt.close(fig)

            # plot output
            pitch_gt = pitch_gt.T
            pitch_output = tensor2onehot(pitch_output)[0].cpu().numpy().T
            pitch_output_woR = tensor2onehot(pitch_output_woR)[0].cpu().numpy().T
            pitch_output_woF = tensor2onehot(pitch_output_woF)[0].cpu().numpy().T

            min_value = np.min(np.hstack([pitch_gt, pitch_output, pitch_output_woR, pitch_output_woF]))
            max_value = np.max(np.hstack([pitch_gt, pitch_output, pitch_output_woR, pitch_output_woF]))
            
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
            ax1.set_title('Original Pitch Contour', fontsize=10)
            ax2.set_title('Output Pitch Contour', fontsize=10)
            ax3.set_title('Output Pitch Contour Without Rhythm', fontsize=10)
            ax4.set_title('Output Pitch Contour Without Pitch', fontsize=10)
            _ = ax1.imshow(pitch_gt, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax2.imshow(pitch_output, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax3.imshow(pitch_output_woR, aspect='auto', vmin=min_value, vmax=max_value)
            _ = ax4.imshow(pitch_output_woF, aspect='auto', vmin=min_value, vmax=max_value)
            plt.savefig(f'{self.sample_dir}/{self.model_name}-{self.mode}-{self.model_type}-output-{spk_id_org[0]}-{step}.png', dpi=150)
            plt.close(fig)

        self.model = self.model.train()
