import torch
from tqdm import tqdm
import librosa
from hparams import wavenet_hparams
from wavenet_vocoder import builder
import numpy as np
import os
from soundfile import read, write
from utils import get_spmel, quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from config import get_config

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
config = get_config()

class Synthesizer(object):

    def __init__(self):
        
        self.model = getattr(builder, wavenet_hparams.builder)(
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

    def load_ckpt(self, ckpt):
        self.model = self.model.to(device)
        self.model.load_state_dict(ckpt['state_dict'])

    def spect2wav(self, c=None, tqdm=tqdm):
        self.model.eval()
        self.model.make_generation_fast_()

        Tc = c.shape[0]
        upsample_factor = wavenet_hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

        initial_input = torch.zeros(1, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(device)
        c = None if c is None else c.to(device)

        with torch.no_grad():
            y_hat = self.model.incremental_forward(
                initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=wavenet_hparams.log_scale_min)

        y_hat = y_hat.view(-1).cpu().data.numpy()

        return y_hat

    def file2wav(self, fname):
        spect = np.load(fname)
        return self.spect2wav(c=spect)



if __name__ == '__main__':
    spmel_dir = './assets/spmel/p225'
    spmel_filt_dir = './assets/spmel_filt/p225'
    raptf0_dir = './assets/raptf0/p225'
    model_dir = './run/models/speech-split2'
    dst_dir = 'eval/wavs'
    fs = 16000
    s = Synthesizer()
    ckpt = torch.load('./run/models/wavenet_vocoder.pth')
    s.load_ckpt(ckpt)
    settings = {
                # 'default': [8,8,8,8,1,32],
                # 'wide_C': [1,8,8,32,1,32],
                'wide_R': [8,1,8,8,32,32],
                # 'wide_CR': [1,1,8,32,32,32]
                }

    with torch.no_grad():
        for name, params in settings.items():

            config.name = name
            config.freq = params[0]
            config.freq_2 = params[1]
            config.freq_3 = params[2]
            config.dim_neck = params[3]
            config.dim_neck_2 = params[4]
            config.dim_neck_3 = params[5] 
        
            G = Generator(config).eval().to(device)
            ckpt = torch.load(os.path.join(model_dir, name+'-G-400000.ckpt'))
            G.load_state_dict(ckpt['model'])

            for fname in sorted(os.listdir(spmel_dir))[:5]:
                spmel = np.load(os.path.join(spmel_dir, fname))
                if len(spmel)%8 != 0:
                    len_pad = 8 - (len(spmel)%8)
                    spmel = np.pad(spmel, ((0,len_pad), (0,0)), 'constant')
                spmel = spmel[np.newaxis, :, :]
                spmel = torch.from_numpy(spmel).to(device)

                spmel_filt = np.load(os.path.join(spmel_filt_dir, fname))
                if len(spmel_filt)%8 != 0:
                    len_pad = 8 - (len(spmel_filt)%8)
                    spmel_filt = np.pad(spmel_filt, ((0,len_pad), (0,0)), 'constant')
                spmel_filt = spmel_filt[np.newaxis, :, :]
                spmel_filt = torch.from_numpy(spmel_filt).to(device)

                raptf0 = np.load(os.path.join(raptf0_dir, fname))
                raptf0 = quantize_f0_numpy(raptf0)[0]
                if len(raptf0)%8 != 0:
                    len_pad = 8 - (len(raptf0)%8)
                    raptf0 = np.pad(raptf0, ((0,len_pad), (0,0)), 'constant')
                raptf0 = raptf0[np.newaxis, :, :]
                raptf0 = torch.from_numpy(raptf0).to(device)

                spmel_f0 = torch.cat((spmel, raptf0), dim=-1)
                spk_emb = torch.zeros((1, 82)).to(device)
                spk_emb[0,0] = 1

                spmel_output = G(spmel_f0, spmel_filt, spk_emb, rr=False)[0].cpu().numpy()
                wav = s.spect2wav(c=spmel_output)
                write(os.path.join(dst_dir, name+'_'+os.path.splitext(fname)[0]+'.wav'), wav, fs)

                del spmel, spmel_filt, raptf0

            del G



            