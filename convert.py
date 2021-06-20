import os
import pickle
from collections import OrderedDict

import torch
import numpy as np
import matplotlib.pyplot as plt
from soundfile import read, write
from tqdm import tqdm

from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
from config import get_config
from wavenet import Synthesizer
from utils import quantize_f0_numpy, inverse_quantize_f0_numpy


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

def load_spenv(fname):
    spenv = np.load(os.path.join(spenv_dir, fname))
    if len(spenv)%8 != 0:
        len_pad = 8 - (len(spenv)%8)
        spenv = np.pad(spenv, ((0,len_pad), (0,0)), 'constant')
    spenv = spenv[np.newaxis, :, :]
    
    return spenv

def load_f0(fname):
    f0 = np.load(os.path.join(f0_dir, fname))
    f0 = quantize_f0_numpy(f0)[0]
    if len(f0)%8 != 0:
        len_pad = 8 - (len(f0)%8)
        f0 = np.pad(f0, ((0,len_pad), (0,0)), 'constant')
    f0 = f0[np.newaxis, :, :]

    return f0

def get_spk_emb(spk_id):
    spk_emb = torch.zeros((1, 82)).to(device)
    spk_emb[0,spk_id] = 1

    return spk_emb

def convert_pitch(rhythm_input, pitch_input):
    max_len = max(rhythm_input.shape[1], pitch_input.shape[1])
    rhythm_input = np.pad(rhythm_input, ((0,0), (0,max_len-rhythm_input.shape[1]), (0,0)), 'constant')
    pitch_input = np.pad(pitch_input, ((0,0), (0,max_len-pitch_input.shape[1]), (0,0)), 'constant')
    rhythm_input = torch.from_numpy(rhythm_input).float().to(device)
    pitch_input = torch.from_numpy(pitch_input).float().to(device)
    pitch_output = F(rhythm_input, pitch_input, rr=False)

    return pitch_output.cpu().numpy()

def convert(model_type, ctype, src_path, tgt_path, src_id, tgt_id):
    if ctype == 'R':
        spmel = load_spmel(src_path)
        if model_type == 'spsp1':
            spmel_filt = load_spmel(tgt_path)
        else:
            spmel_filt = np.concatenate((load_spmel_filt(tgt_path), load_spenv(tgt_path)), axis=-1)
        f0 = load_f0(src_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'C':
        spmel = load_spmel(src_path)
        if model_type == 'spsp1':
            spmel_filt = load_spmel(tgt_path)
        else:
            spmel_filt = np.concatenate((load_spmel_filt(tgt_path), load_spenv(tgt_path)), axis=-1)
        f0 = load_f0(src_path)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'F':
        spmel = load_spmel(src_path)
        if model_type == 'spsp1':
            spmel_filt = load_spmel(src_path)
        else:
            spmel_filt = np.concatenate((load_spmel_filt(src_path), load_spenv(src_path)), axis=-1)
        f0 = load_f0(tgt_path)
        if model_type == 'spsp1':
            f0 = convert_pitch(spmel, f0)
        else:
            f0 = convert_pitch(spmel_filt, f0)
        spk_emb = get_spk_emb(src_id)
    elif ctype == 'U':
        spmel = load_spmel(src_path)
        if model_type == 'spsp1':
            spmel_filt = load_spmel(src_path)
        else:
            spmel_filt = np.concatenate((load_spmel_filt(src_path), load_spenv(src_path)), axis=-1)
        f0 = load_f0(src_path)
        spk_emb = get_spk_emb(tgt_id)
    else:
        spmel = load_spmel(src_path)
        if model_type == 'spsp1':
            spmel_filt = load_spmel(src_path)
        else:
            spmel_filt = np.concatenate((load_spmel_filt(src_path), load_spenv(src_path)), axis=-1)
        f0 = load_f0(src_path)
        spk_emb = get_spk_emb(src_id)

    T = 192
    spmel = np.pad(spmel, ((0,0), (0,T-spmel.shape[1]), (0,0)), 'constant')
    spmel_filt = np.pad(spmel_filt, ((0,0), (0,T-spmel_filt.shape[1]), (0,0)), 'constant')
    f0 = np.pad(f0, ((0,0), (0,T-f0.shape[1]), (0,0)), 'constant')
    f0_1d = inverse_quantize_f0_numpy(f0[0])

    spmel = torch.from_numpy(spmel).to(device)
    spmel_filt = torch.from_numpy(spmel_filt).to(device)
    f0 = torch.from_numpy(f0).to(device)
    spmel_f0 = torch.cat((spmel, f0), dim=-1)
    
    rhythm_input = spmel_filt[0].cpu().numpy()
    content_input = spmel_f0[0][:, :-257].cpu().numpy()
    pitch_input = spmel_f0[0][:, -257:].cpu().numpy()

    rhythm = G.rhythm(spmel_filt)
    content, pitch = G.content_pitch(spmel_f0, rr=False)
    spmel_output = G.decode(content, rhythm, pitch, spk_emb, T).cpu().numpy()[0]

    return spmel_output, f0_1d, rhythm_input, content_input, pitch_input

def draw_plot(images, names, plot_path):
    assert len(images) == len(names), 'Length of images must be equal to length of names'
    N = len(images)
    max_len = max([len(image) for image in images])
    for i in range(N):
        images[i] = np.pad(images[i], ((0, max_len-len(images[i])), (0, 0)), 'constant')
    min_value = np.min(np.concatenate([image for image in images], axis=-1))
    max_value = np.max(np.concatenate([image for image in images], axis=-1))

    fig, axes = plt.subplots(N, 1, sharex=True, figsize=(N*2, N*2))
    for ax, image, name in zip(axes, images, names):
        ax.set_title(name, fontsize=10)
        ax.imshow(image.T, aspect='auto', vmin=min_value, vmax=max_value)
    plt.savefig(f'{plot_path}', dpi=150)
    plt.close(fig)


if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config = get_config()
    config.mode = 'test'

    root_dir = config.root_dir
    src_dir = os.path.join(root_dir, config.mode, config.src_dir)
    spmel_dir = os.path.join(root_dir, config.mode, config.spmel_dir)
    spmel_filt_dir = os.path.join(root_dir, config.mode, config.spmel_filt_dir)
    spenv_dir = os.path.join(root_dir, config.mode, config.spenv_dir+str(config.cutoff))
    f0_dir = os.path.join(root_dir, config.mode, config.f0_dir)

    model_save_dir = os.path.join(root_dir, config.model_save_dir)
    result_dir = os.path.join(root_dir, config.result_dir)
    plot_dir = os.path.join(root_dir, config.plot_dir)

    test_data_by_ctype = pickle.load(open(os.path.join(root_dir, config.mode, 'test_data_by_ctype.pkl'), 'rb'))
    fs = 16000
    S = Synthesizer(device)
    S.load_ckpt(os.path.join(result_dir, 'wavenet_vocoder.pth'))

    model_type_list = [
        # 'spsp1',
        'spsp2',
    ]
    settings = {
                # 'R_8_1': [8,8,8,8,1,32],
                # 'R_1_1': [8,1,8,8,1,32],
                # 'R_8_32': [8,8,8,8,32,32],
                'R_1_32': [8,1,8,8,32,32],
                }

    ctype_list = [
        'F',
        # 'C',
        'R',
        'U',
    ]

    spmel_list = []
    spmel_path_list = []

    with torch.no_grad():
        for model_type in model_type_list:
            for model_name, params in settings.items():

                config.model_name = model_name
                config.experiment = model_type
                config.freq = params[0]
                config.freq_2 = params[1]
                config.freq_3 = params[2]
                config.dim_neck = params[3]
                config.dim_neck_2 = params[4]
                config.dim_neck_3 = params[5]
                cutoff = config.cutoff
            
                G = Generator(config).eval().to(device)
                ckpt = torch.load(os.path.join(model_save_dir, model_type, model_name+'-G-'+str(cutoff)+'-best.ckpt'))
                try:
                    G.load_state_dict(ckpt['model'])
                except:
                    new_state_dict = OrderedDict()
                    for k, v in ckpt['model'].items():
                        new_state_dict[k[7:]] = v
                    G.load_state_dict(new_state_dict)

                F = F0_Converter(config).eval().to(device)
                ckpt = torch.load(os.path.join(model_save_dir, model_type, model_name+'-F-'+str(cutoff)+'-best.ckpt'))
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
                        result_path = os.path.join(result_dir, model_type, model_name, ctype)
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)

                        src_spmel_path = os.path.join(result_path, fname+'_s.wav')
                        src_spmel = np.load(os.path.join(spmel_dir, src_name+'.npy'))
                        src_wav, _ = read(os.path.join(src_dir, src_name+'.wav'))
                        src_save_path = os.path.join(result_path, fname+'_s.wav')
                        write(src_save_path, src_wav, fs)

                        tgt_spmel_path = os.path.join(result_path, fname+'_t.wav')
                        tgt_spmel = np.load(os.path.join(spmel_dir, tgt_name+'.npy'))
                        tgt_wav, _ = read(os.path.join(src_dir, tgt_name+'.wav'))
                        tgt_save_path = os.path.join(result_path, fname+'_t.wav')
                        write(tgt_save_path, tgt_wav, fs)

                        cvt_spmel_path = os.path.join(result_path, fname+'_c.wav')
                        cvt_spmel, tgt_f0, rhythm_input, content_input, pitch_input = convert(model_type, ctype, src_name+'.npy', tgt_name+'.npy', src_id, tgt_id)
                        spmel_list.append(torch.from_numpy(cvt_spmel))
                        spmel_path_list.append(cvt_spmel_path)

                        tgt_f0_path = os.path.join(result_path, fname+'_t.npy')
                        np.save(tgt_f0_path, tgt_f0)
                        
                        if not os.path.exists(os.path.join(plot_dir, model_type, model_name, ctype)):
                            os.makedirs(os.path.join(plot_dir, model_type, model_name, ctype))

                        # plot input
                        images = [rhythm_input, content_input, pitch_input]
                        names = ['Rhythm Input', 'Content Input', 'Pitch Input']
                        plot_path = os.path.join(plot_dir, model_type, model_name, ctype, fname+'_input.png')
                        draw_plot(images, names, plot_path)

                        # plot output
                        images = [src_spmel, tgt_spmel, cvt_spmel]
                        names = ['Source Mel-Spectrogram', 'Target Mel-Spectrogram', 'Converted Mel-Spectrogram']
                        plot_path = os.path.join(plot_dir, model_type, model_name, ctype, fname+'_output.png')
                        draw_plot(images, names, plot_path)

            i = 0
            batch_size = 120
            print(len(spmel_path_list), len(spmel_list))
            while i<len(spmel_list):
                cvt_spmel_batch = torch.nn.utils.rnn.pad_sequence(spmel_list[i:i+batch_size], batch_first=True)
                cvt_wav_batch = S.batch_spect2wav(c=cvt_spmel_batch)
                for path, wav in zip(spmel_path_list[i:i+batch_size], cvt_wav_batch):
                    print(path)
                    write(path, wav, fs)
                i += batch_size
            spmel_list = []
            spmel_path_list = []
