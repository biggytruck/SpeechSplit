# coding: utf-8
"""
Synthesis waveform from trained WaveNet.
Modified from https://github.com/r9y9/wavenet_vocoder
"""

import torch
from tqdm import tqdm
import librosa
from hparams import wavenet_hparams
from wavenet_vocoder import builder

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def build_model():
    
    model = getattr(builder, wavenet_hparams.builder)(
        out_channels=wavenet_hparams.out_channels,
        layers=wavenet_hparams.layers,
        stacks=wavenet_hparams.stacks,
        residual_channels=wavenet_hparams.residual_channels,
        gate_channels=wavenet_hparams.gate_channels,
        skip_out_channels=wavenet_hparams.skip_out_channels,
        cin_channels=wavenet_hparams.cin_channels,
        gin_channels=wavenet_hparams.gin_channels,
        weight_normalization=wavenet_hparams.weight_normalization,
        n_speakers=wavenet_hparams.n_speakers,
        dropout=wavenet_hparams.dropout,
        kernel_size=wavenet_hparams.kernel_size,
        upsample_conditional_features=wavenet_hparams.upsample_conditional_features,
        upsample_scales=wavenet_hparams.upsample_scales,
        freq_axis_kernel_size=wavenet_hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=wavenet_hparams.legacy,
    )
    return model



def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet.
    
    """

    model.eval()
    model.make_generation_fast_()

    Tc = c.shape[0]
    upsample_factor = wavenet_hparams.hop_sizekwargs
    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=wavenet_hparams.log_scale_min)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat