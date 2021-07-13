import os
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from data_preprocessing import make_metadata, make_spect_f0
from config import get_config


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Data preprocessing(optional).
    if config.make_spect_f0:
        make_spect_f0(config)
    if config.make_metadata:
        make_metadata(config)
    if config.run_model:

        # Experiments
        experiments = [
            # 'spsp1',
            'spsp2'
        ]
        
        # Bottleneck size settings
        settings = {
                    # 'R_8_1': [8,8,8,8,1,32],
                    'R_1_32': [1,1,1,32,32,32],
        }

        # G or F
        model_types = [
            'G',
            # 'F'
        ]


        for experiment in experiments:
            for model_name, hparams in settings.items():
                for model_type in model_types:
                
                    config.experiment = experiment
                    config.model_name = model_name
                    config.freq = hparams[0]
                    config.freq_2 = hparams[1]
                    config.freq_3 = hparams[2]
                    config.dim_neck = hparams[3]
                    config.dim_neck_2 = hparams[4]
                    config.dim_neck_3 = hparams[5] 
                    config.model_type = model_type

                    # Data loader.
                    data_loader = get_loader(config)

                    # Solver for training
                    solver = Solver(data_loader, config)

                    if config.mode == 'train':
                        solver.train()
                    elif config.mode == 'test':
                        solver.test()
                    else:
                        raise ValueError
            
if __name__ == '__main__':
    
    config = get_config()
    main(config)
