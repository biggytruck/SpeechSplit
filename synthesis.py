import torch
from tqdm import tqdm
import librosa
from hparams import wavenet_hparams
from wavenet_vocoder import builder
import numpy as np
import os
import random
import pickle
from soundfile import read, write
from utils import get_spmel, quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from config import get_config
from collections import OrderedDict
import matplotlib.pyplot as plt


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
    spk_emb[0,spk_id] = 1

    return spk_emb

def convert_pitch(F, rhythm_input, pitch_input):
    max_len = max(rhythm_input.shape[1], pitch_input.shape[1])
    rhythm_input = np.pad(rhythm_input, ((0,0), (0,max_len-rhythm_input.shape[1]), (0,0)), 'constant')
    pitch_input = np.pad(pitch_input, ((0,0), (0,max_len-pitch_input.shape[1]), (0,0)), 'constant')
    rhythm_input = torch.from_numpy(rhythm_input).float().to(device)
    pitch_input = torch.from_numpy(pitch_input).float().to(device)
    pitch_output = F(rhythm_input, pitch_input, rr=False)

    return pitch_output.cpu().numpy()

def convert(G, F, model_type, ctype, src_path, tgt_path, src_id, tgt_id):
    if ctype == 'R':
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(tgt_path)
        raptf0 = load_raptf0(src_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'C':
        spmel = load_spmel(tgt_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(src_path)
        if model_type == 'spsp1':
            raptf0 = convert_pitch(F, spmel, raptf0)
        else:
            raptf0 = convert_pitch(F, spmel_filt, raptf0)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'F':
        spmel = load_spmel(src_path)
        spmel_filt = load_spmel_filt(src_path)
        raptf0 = load_raptf0(tgt_path)
        if model_type == 'spsp1':
            raptf0 = convert_pitch(F, spmel, raptf0)
        else:
            raptf0 = convert_pitch(F, spmel_filt, raptf0)
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

    T = 192
    spmel = np.pad(spmel, ((0,0), (0,T-spmel.shape[1]), (0,0)), 'constant')
    spmel_filt = np.pad(spmel_filt, ((0,0), (0,T-spmel_filt.shape[1]), (0,0)), 'constant')
    raptf0 = np.pad(raptf0, ((0,0), (0,T-raptf0.shape[1]), (0,0)), 'constant')

    spmel = torch.from_numpy(spmel).to(device)
    spmel_filt = torch.from_numpy(spmel_filt).to(device)
    raptf0 = torch.from_numpy(raptf0).to(device)
    spmel_f0 = torch.cat((spmel, raptf0), dim=-1)
    
    if model_type == 'spsp1':
        rhythm = G.rhythm(spmel)
    else:
        rhythm = G.rhythm(spmel_filt)
    content, pitch = G.content_pitch(spmel_f0, rr=False)
    spmel_output = G.decode(content, rhythm, pitch, spk_emb, T).cpu().numpy()[0]

    return spmel_output

def draw_plot(src_spmel, tgt_spmel, cvt_spmel, save_path):
    max_len = max(len(src_spmel), len(tgt_spmel), len(cvt_spmel))
    src_spmel = np.pad(src_spmel, ((0, max_len-len(src_spmel)), (0, 0)), 'constant')
    tgt_spmel = np.pad(tgt_spmel, ((0, max_len-len(tgt_spmel)), (0, 0)), 'constant')
    cvt_spmel = np.pad(cvt_spmel, ((0, max_len-len(cvt_spmel)), (0, 0)), 'constant')
    min_value = np.min(np.vstack([src_spmel, tgt_spmel, cvt_spmel]))
    max_value = np.max(np.vstack([src_spmel, tgt_spmel, cvt_spmel]))
    
    fig, (ax1,ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 5))
    ax1.set_title('Source Mel-Spectrogram', fontsize=10)
    ax2.set_title('Target Mel-Spectrogram', fontsize=10)
    ax3.set_title('Convertedd Mel-Spectrogram', fontsize=10)
    im1 = ax1.imshow(src_spmel.T, aspect='auto', vmin=min_value, vmax=max_value)
    im2 = ax2.imshow(tgt_spmel.T, aspect='auto', vmin=min_value, vmax=max_value)
    im3 = ax3.imshow(cvt_spmel.T, aspect='auto', vmin=min_value, vmax=max_value)
    plt.savefig(f'{save_path}', dpi=150)
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

    def batchspect2wav(self, c=None, tqdm=tqdm):
        self.model.eval()
        self.model.make_generation_fast_()

        B = c.size(0)
        Tc = c.size(1)
        upsample_factor = wavenet_hparams.hop_size
        # Overwrite length according to feature size
        length = Tc * upsample_factor

        # B x C x T
        c = torch.FloatTensor(c.permute(0, 2, 1))

        initial_input = torch.zeros(B, 1, 1).fill_(0.0)

        # Transform data to GPU
        initial_input = initial_input.to(device)
        c = None if c is None else c.to(device)

        with torch.no_grad():
            y_hat = self.model.incremental_forward(
                initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=wavenet_hparams.log_scale_min)

        y_hat = y_hat.view(B, -1).cpu().data.numpy()

        return y_hat

    def file2wav(self, fname):
        spect = np.load(fname)
        return self.spect2wav(c=spect)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config = get_config()
    wav_dir = 'eval/assets/wavs/'
    spmel_dir = 'eval/assets/spmel/'
    spmel_filt_dir = 'eval/assets/spmel_filt/'
    raptf0_dir = 'eval/assets/raptf0/'
    model_dir = 'eval/models/'
    result_dir = 'eval/results'
    plot_dir = 'eval/plots'
    test_data_by_ctype = pickle.load(open('eval/assets/test_data_by_ctype.pkl', 'rb'))
    fs = 16000
    s = Synthesizer()
    ckpt = torch.load('./run/models/wavenet_vocoder.pth')
    s.load_ckpt(ckpt)
    model_type_list = [
        'spsp1',
        # 'spsp2',
    ]
    settings = {
                'R_8_1': [8,8,8,8,1,32],
                'R_1_1': [8,1,8,8,1,32],
                'R_8_32': [8,8,8,8,32,32],
                'R_1_32': [8,1,8,8,32,32],
                # 'wide_CR_8_8': [8,1,8,8,32,32],
                }

    ctype_list = [
        'F',
        # 'C',
        'R',
        'U',
    ]

    cvt_spmel_list = []
    cvt_save_path_list = []

    with torch.no_grad():
        for model_type in model_type_list:
            for model_name, params in settings.items():

                config.model_name = model_name
                config.freq = params[0]
                config.freq_2 = params[1]
                config.freq_3 = params[2]
                config.dim_neck = params[3]
                config.dim_neck_2 = params[4]
                config.dim_neck_3 = params[5]
            
                G = Generator(config).eval().to(device)
                ckpt = torch.load(os.path.join(model_dir, model_type, model_name+'-G-'+'best.ckpt'))
                try:
                    G.load_state_dict(ckpt['model'])
                except:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt['model'].items():
                        new_state_dict[k[7:]] = v
                    G.load_state_dict(new_state_dict)

                F = F0_Converter(config).eval().to(device)
                ckpt = torch.load(os.path.join(model_dir, model_type, model_name+'-F-'+'best.ckpt'))
                try:
                    F.load_state_dict(ckpt['model'])
                except:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt['model'].items():
                        new_state_dict[k[7:]] = v
                    F.load_state_dict(new_state_dict)

                for ctype in ctype_list:
                    pairs = test_data_by_ctype[ctype]
                    for (src_name, src_id), (tgt_name, tgt_id) in pairs[:40]:
                        fname = src_name.split('/')[-1]+'_'+tgt_name.split('/')[-1]

                        src_spmel = np.load(os.path.join(spmel_dir, src_name+'.npy'))
                        src_wav, _ = read(os.path.join(wav_dir, src_name+'.wav'))
                        src_save_path = os.path.join(result_dir, model_type, model_name, ctype, fname+'_s.wav')
                        write(src_save_path, src_wav, fs)

                        tgt_spmel = np.load(os.path.join(spmel_dir, tgt_name+'.npy'))
                        tgt_wav, _ = read(os.path.join(wav_dir, tgt_name+'.wav'))
                        tgt_save_path = os.path.join(result_dir, model_type, model_name, ctype, fname+'_t.wav')
                        write(tgt_save_path, tgt_wav, fs)

                        cvt_spmel = convert(G, F, model_type, ctype, src_name+'.npy', tgt_name+'.npy', src_id, tgt_id)
                        cvt_save_path = os.path.join(result_dir, model_type, model_name, ctype, fname+'_c.wav')
                        cvt_spmel_list.append(cvt_spmel)
                        cvt_save_path_list.append(cvt_save_path)
                        if len(cvt_save_path_list)==16:
                            cvt_spmel_batch = torch.nn.utils.rnn.pad_sequence(cvt_spmel_list, batch_first=False)
                            cvt_wav_batch = s.batchspect2wav(c=cvt_spmel_batch)
                            for path, wav in zip(cvt_save_path_list, cvt_wav_batch):
                                write(path, wav, fs)
                            cvt_spmel_list = []
                            cvt_save_path_list = []
                        
                        plot_path = os.path.join(plot_dir, model_type, model_name, ctype, fname+'.png')
                        draw_plot(src_spmel, tgt_spmel, cvt_spmel, plot_path)

        cvt_spmel_batch = torch.nn.utils.rnn.pad_sequence(cvt_spmel_list, batch_first=False)
        cvt_wav_batch = s.batchspect2wav(c=cvt_spmel_batch)
        for path, wav in zip(cvt_save_path_list, cvt_wav_batch):
            write(path, wav, fs)