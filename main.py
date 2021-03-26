import os
import torch
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from data_preprocessing import make_metadata, make_spect_f0
from config import get_config


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    log_dir = os.path.join(config.root_dir, config.log_dir)
    model_save_dir = os.path.join(config.root_dir, config.model_save_dir)
    sample_dir = os.path.join(config.root_dir, config.sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Data preprocessing(optional).
    if config.make_spect_f0:
        make_spect_f0(config)
    if config.make_metadata:
        make_metadata(config)
    if config.run_model:

        # Data loader.
        data_loader_list = get_loader(config)

        # Experiments
        experiments = [
            'spsp1',
            'spsp2'
        ]
        
        # Bottleneck size settings
        settings = {
                    'R_8_1': [8,8,8,8,1,32],
                    'R_1_1': [8,1,8,8,1,32],
                    'R_8_32': [8,8,8,8,32,32],
                    'R_1_32': [8,1,8,8,32,32],
                    }


        for experiment in experiments:
            for model_name, hparams in settings.items():
                
                config.experiment = experiment
                config.model_name = model_name
                config.freq = hparams[0]
                config.freq_2 = hparams[1]
                config.freq_3 = hparams[2]
                config.dim_neck = hparams[3]
                config.dim_neck_2 = hparams[4]
                config.dim_neck_3 = hparams[5] 

                # Solver for training
                solver = Solver(data_loader_list, config)

                if config.mode == 'train':
                    solver.train()
                elif config.mode == 'test':
                    solver.test()
                else:
                    raise ValueError
            
if __name__ == '__main__':
    
    config = get_config()
    main(config)