import torch
from tqdm import tqdm
import librosa
from hparams import wavenet_hparams
from wavenet_vocoder import builder
import numpy as np
import os
import random
from soundfile import read, write
from utils import get_spmel, quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from config import get_config
from collections import OrderedDict
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
config = get_config()
spmel_dir = './assets/spmel/'
spmel_filt_dir = './assets/spmel_filt/'
raptf0_dir = './assets/raptf0/'
model_dir = './run/models/speech-split2-find-content-optimal/'
dst_dir = 'eval/wavs'

def load_spmel(fname):
    spmel = np.load(os.path.join(spmel_dir, fname))
    if len(spmel)%8 != 0:
        len_pad = 8 - (len(spmel)%8)
        spmel = np.pad(spmel, ((0,len_pad), (0,0)), 'constant')
    spmel = spmel[np.newaxis, :, :]
    
    return spmel

def load_spmel_filt(fname):
    spmel_filt = np.load(os.path.join(spmel_filt_dir, fname))
    if len(spmel_filt)%8 != 0:
        len_pad = 8 - (len(spmel_filt)%8)
        spmel_filt = np.pad(spmel_filt, ((0,len_pad), (0,0)), 'constant')
    spmel_filt = spmel_filt[np.newaxis, :, :]
    
    return spmel_filt

def load_raptf0(fname):
    raptf0 = np.load(os.path.join(raptf0_dir, fname))
    raptf0 = quantize_f0_numpy(raptf0)[0]
    if len(raptf0)%8 != 0:
        len_pad = 8 - (len(raptf0)%8)
        raptf0 = np.pad(raptf0, ((0,len_pad), (0,0)), 'constant')
    raptf0 = raptf0[np.newaxis, :, :]

    return raptf0

def get_spk_emb(spk_id):
    spk_emb = torch.zeros((1, 82)).to(device)
    spk_emb[0,src_id] = 1

    return spk_emb

def convert(model, ctype, src_path, tgt_path, src_id, tgt_id):
    print(ctype, src_path, tgt_path, src_id, tgt_id)
    if ctype == 'R':
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(tgt_path)
        raptf0 = load_raptf0(src_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'C':
        spmel = load_spmel(tgt_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(src_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'F':
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(tgt_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'U':
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(src_path)
        spk_emb = get_spk_emb(tgt_id)
    else:
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(src_path)
        spk_emb = get_spk_emb(src_id)

    T = max(spmel.shape[1], spmel_filt.shape[1], raptf0.shape[1])
    spmel = np.pad(spmel, ((0,0), (0,T-spmel.shape[1]), (0,0)), 'constant')
    spmel_filt = np.pad(spmel_filt, ((0,0), (0,T-spmel_filt.shape[1]), (0,0)), 'constant')
    raptf0 = np.pad(raptf0, ((0,0), (0,T-raptf0.shape[1]), (0,0)), 'constant')

    spmel = torch.from_numpy(spmel).to(device)
    spmel_filt = torch.from_numpy(spmel_filt).to(device)
    raptf0 = torch.from_numpy(raptf0).to(device)
    spmel_f0 = torch.cat((spmel, raptf0), dim=-1)
    
    rhythm = model.rhythm(spmel_filt)
    content, pitch = model.content_pitch(spmel_f0, rr=False)
    spmel_output = model.decode(content, rhythm, pitch, spk_emb, T).cpu().numpy()[0]

    return spmel_output

def draw_plot(spmel, title):
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(title, fontsize=10)
    im1 = ax1.imshow(spmel.T, aspect='auto')
    plt.savefig(f'{title}.png', dpi=150)
    plt.close(fig)
        

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
    
    fs = 16000
    s = Synthesizer()
    ckpt = torch.load('./run/models/wavenet_vocoder.pth')
    s.load_ckpt(ckpt)
    settings = {
                'wide_CR_4_8': [4,1,8,8,32,32,500000],
                'wide_CR_4_16': [4,1,8,16,32,32,500000],
                'wide_CR_2_8': [2,1,8,8,32,32,500000],
                'wide_CR_2_16': [2,1,8,16,32,32,500000],
                # 'wide_CR_1_8': [1,1,8,8,32,32,500000],
                # 'wide_CR_1_16': [1,1,8,16,32,32,500000],
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
            config.resume_iters = params[6]
        
            G = Generator(config).eval().to(device)
            ckpt = torch.load(os.path.join(model_dir, name+'-G-'+str(config.resume_iters)+'.ckpt'))
            
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                new_state_dict[k[7:]] = v
            G.load_state_dict(new_state_dict)

            # wav_pairs = process_conversion_list()
            # random.shuffle(wav_pairs)
            # convert_cnt = 0
            # for ctype, src_path, tgt_path, src_id, tgt_id, output_name in wav_pairs:
            #     if ctype != 'U':
            #         continue
            #     if convert_cnt >= 5:
            #         break
            #     print(ctype, src_path, tgt_path, src_id, tgt_id, output_name)
            #     spmel_output = convert(G, 'C', tgt_path, src_path, tgt_id, src_id)
            #     wav = s.spect2wav(c=spmel_output)
            #     write(os.path.join(dst_dir, name+'_'+output_name+'.wav'), wav, fs)
            #     convert_cnt += 1

            src_path = 'p225/p225_001_mic1.npy'
            tgt_path = 'p258/p258_001_mic1.npy'
            src_id = 0
            tgt_id = 1

            for ctype in ['R', 'C', 'F', 'U', 'None']:
                spmel_output = convert(G, ctype, src_path, tgt_path, src_id, tgt_id)
                draw_plot(spmel_output, ctype)
                wav = s.spect2wav(c=spmel_output)
                write(os.path.join(dst_dir, name+'_'+'p225_p258_001001_'+ctype+'.wav'), wav, fs)

