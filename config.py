import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def get_config():

    # Directories
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--src_dir', type=str, default='assets/wavs')
    parser.add_argument('--wav_dir', type=str, default='assets/filt_wav')
    parser.add_argument('--spmel_dir', type=str, default='assets/spmel')
    parser.add_argument('--spmel_filt_dir', type=str, default='assets/spmel_filt')
    parser.add_argument('--f0_dir', type=str, default='assets/raptf0')
    parser.add_argument('--txt_dir', type=str, default='assets/txt')
    parser.add_argument('--meta_dir', type=str, default='assets/meta')

    # Data Preprocessing
    parser.add_argument('--make_metadata', type=str2bool, default=False)
    parser.add_argument('--make_spect_f0', type=str2bool, default=False)

    # Dataloader
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--samplier', type=int, default=8)

    # Model hyperparameters
    parser.add_argument('--freq', type=int, default=8)
    parser.add_argument('--freq_2', type=int, default=8)
    parser.add_argument('--freq_3', type=int, default=8)
    parser.add_argument('--dim_neck', type=int, default=8)
    parser.add_argument('--dim_neck_2', type=int, default=1)
    parser.add_argument('--dim_neck_3', type=int, default=32)
    parser.add_argument('--dim_enc', type=int, default=512)
    parser.add_argument('--dim_enc_2', type=int, default=128)
    parser.add_argument('--dim_enc_3', type=int, default=256)
    parser.add_argument('--dim_freq', type=int, default=80)
    parser.add_argument('--dim_spk_emb', type=int, default=82)
    parser.add_argument('--dim_f0', type=int, default=257)
    parser.add_argument('--dim_dec', type=int, default=512)
    parser.add_argument('--len_raw', type=int, default=128)
    parser.add_argument('--chs_grp', type=int, default=16)
    parser.add_argument('--min_len_seg', type=int, default=19)
    parser.add_argument('--max_len_seg', type=int, default=32)
    parser.add_argument('--min_len_seq', type=int, default=64)
    parser.add_argument('--max_len_seq', type=int, default=128)
    parser.add_argument('--max_len_pad', type=int, default=192)

    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=500000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Miscellaneous.
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='run/logs/')
    parser.add_argument('--model_save_dir', type=str, default='run/models/')
    parser.add_argument('--best_model_dir', type=str, default='eval/models/')
    parser.add_argument('--sample_dir', type=str, default='run/samples/')
    parser.add_argument('--experiment', type=str, default='speech-split2')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=50000)
    parser.add_argument('--model_save_step', type=int, default=50000)

    config = parser.parse_args()

    return config