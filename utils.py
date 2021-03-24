import copy
import torch
import numpy as np
import os

from collections import OrderedDict
from random import choice
from pysptk import sptk
from scipy import signal
from librosa.filters import mel
from librosa.core import resample
from librosa.util import fix_length
from scipy.signal import get_window, filtfilt, medfilt2d
from math import pi, sqrt, exp


mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    



def speaker_normalization(f0, index_nonzero, mean_f0, std_f0):
    # f0 is logf0
    f0 = f0.astype(float).copy()
    #index_nonzero = f0 != 0
    std_f0 += 1e-6
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0



def quantize_f0_numpy(x, num_bins=256):
    # x is logf0
    assert x.ndim==1
    x = x.astype(float).copy()
    uv = (x<=0)
    x[uv] = 0.0
    assert (x >= 0).all() and (x <= 1).all()
    x = np.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0.0
    enc = np.zeros((len(x), num_bins+1), dtype=np.float32)
    enc[np.arange(len(x)), x.astype(np.int32)] = 1.0
    return enc, x.astype(np.int64)



def quantize_f0_torch(x, num_bins=256):
    # x is logf0
    B = x.size(0)
    x = x.view(-1).clone()
    uv = (x<=0)
    x[uv] = 0
    assert (x >= 0).all() and (x <= 1).all()
    x = torch.round(x * (num_bins-1))
    x = x + 1
    x[uv] = 0
    enc = torch.zeros((x.size(0), num_bins+1), device=x.device)
    enc[torch.arange(x.size(0)), x.long()] = 1
    return enc.view(B, -1, num_bins+1), x.view(B, -1).long()



def get_mask_from_lengths(lengths, max_len):
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids >= lengths.unsqueeze(1)).bool()
    return mask
    
    

def pad_seq_to_2(x, len_out=128):
    len_pad = (len_out - x.shape[1])
    assert len_pad >= 0
    return np.pad(x, ((0,0),(0,len_pad),(0,0)), 'constant'), len_pad    



b, a = butter_highpass(30, 16000, order=5)
def filter_wav(x, prng):
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
    return wav



def get_spmel(wav):
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100       
    return S



def extract_f0(wav, fs, lo, hi):
    f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
    return f0_rapt, f0_norm



def windowing(data, rate, frame=0.03, stride=0.01):
    frame_i = 0
    data_length = len(data)
    frame_length = int(frame * rate)
    stride_length = int(2 ** np.floor(np.log2(stride * rate)))
    frame_number = int((data_length - frame_length) / stride_length + 1)
    windowed_frames = np.zeros((frame_number, frame_length))
    for i in range(frame_number):
        curr_frame = data[frame_i:frame_i+np.minimum(frame_length, data_length-(frame_i+frame_length))]
        if len(curr_frame) < frame_length:
            pad_length = frame_length - len(curr_frame)
            curr_frame = np.pad(curr_frame, (int(pad_length/2), pad_length - int(pad_length/2)), 'constant')
        windowed_frames[i] = curr_frame * np.hanning(frame_length)
        frame_i += stride_length
        
    return windowed_frames, frame_length, stride_length



def phase_norm(phase):
    phase += 2 * np.pi
    return phase % (2 * np.pi)



def time_stretch(data, rate=16000, alpha=1.0, robotic=False, frame=0.03, stride=0.01, pre_pha=None):
    frames, N, Ra = windowing(data, rate, frame, stride)
    Rs = int(Ra * alpha)
    k = np.linspace(0, int(N/2), int(N/2+1))
    fft_frames = np.fft.rfft(frames, axis=1)

    ana_delta_pha = phase_norm(k*2*np.pi*Ra/N)
    syn_delta_pha = phase_norm(k*2*np.pi*Rs/N)
    ana_prev_pha = np.angle(fft_frames[0]) if pre_pha is None else pre_pha
    syn_prev_pha = np.angle(fft_frames[0]) if pre_pha is None else pre_pha

    syn_frames = []
    first_mag = np.abs(fft_frames[0])
    first_pha = np.angle(fft_frames[0])
    syn_frames.append((first_mag * np.exp(1j * first_pha)).real)
    for f in fft_frames[1:]:
        ana_curr_pha = phase_norm(np.angle(f).real)
        delta_pha = ana_curr_pha - ana_prev_pha
        delta_pha = phase_norm(delta_pha)
        delta_pha -= ana_delta_pha
        ana_prev_pha = ana_curr_pha
        syn_mag = np.abs(f)
        syn_curr_pha = phase_norm(syn_prev_pha + syn_delta_pha + delta_pha)
        syn_fft = syn_mag if robotic else syn_mag * np.exp(1j * syn_curr_pha)
        syn_prev_pha = syn_curr_pha
        syn_f = np.fft.irfft(syn_fft).real
        syn_frames.append(syn_f)
    F, W = frames.shape
    output_length = (F-1) * Rs + W
    output = np.zeros(output_length)
    concat_i = 0
    for syn_f in syn_frames:
        output[concat_i:concat_i+len(syn_f)] += syn_f
        concat_i += Rs
    # output /= np.amax(output)
    return output, syn_curr_pha



def pitch_shift(x, fs, n_steps, bins_per_octave=12, pre_pha=None):

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)
    rate = 1/rate
    x_stretch, new_pre_pha = time_stretch(x, alpha=rate, pre_pha=pre_pha)
    x_shift = resample(x_stretch, float(fs) * rate, fs)
    return fix_length(x_shift, len(x)), new_pre_pha


def random_warping(wav, fs=16000, trunc=[0.3, 0.6]):
    # rate = choice([np.random.uniform(0.6, 0.8), np.random.uniform(1.2, 1.4)])
    # wav = time_stretch(wav, 16000, rate)
    i = 0
    pre_pha = None
    new_wav = np.array(wav)
    while i < (len(wav)-int(fs*trunc[0])):
        curr_len = int(fs * np.random.uniform(trunc[0], trunc[1]))
        n_steps = choice([np.random.uniform(5, 12), np.random.uniform(-5, -12)])
        new_wav[i:i+curr_len], pre_pha = pitch_shift(wav[i:i+curr_len], 16000, n_steps, pre_pha=pre_pha)
        i += curr_len
    return new_wav/np.max(new_wav)


def get_dog_filter(n):
    r1 = range(-int(n/2),int(n/2)+1)
    r2 = range(-int(n/2)-1,int(n/2))
    f1 = [1 / sqrt(2*pi) * exp(-float(x1)**2/2) for x1 in r1]
    f2 = [1 / sqrt(2*pi) * exp(-float(x2)**2/2) for x2 in r2]
    return [(x1-x2) for x1, x2 in zip(f1, f2)]


def make_filt_spect(S, n=31):
    S_T = S.T
    S_filt = np.zeros_like(S_T)
    f = get_dog_filter(n)
    for i in range(len(S_T)):
        S_filt[i] = np.convolve(S_T[i], f, mode='same')

    return S_filt.T


def spectral_subtraction(S, Tn=0.25, alpha=0.35, med=3, fs=16000):
    S_noise = S[:int(Tn*fs)]
    rep = len(S) // len(S_noise)
    res = len(S) % len(S_noise)
    S_noise_rp = np.tile(S_noise, (rep, 1))
    S_noise_rp = np.vstack((S_noise, S_noise[:res]))

    pha = np.angle(S)
    diff = np.abs(S) - alpha * np.abs(S_noise_rp)
    diff[diff<0] = 0
    S_subt = diff * np.exp(1j*pha)

    if med:
        pha = np.angle(S_subt)
        mag = np.abs(S_subt)
        mag = medfilt2d(mag, med)
        S_subt = mag * np.exp(1j*pha)

    return np.abs(S_subt)


def zero_one_norm(S):
    S_norm = S - np.min(S)
    S_norm /= np.max(S_norm)

    return S_norm


def smooth_spmel(S, n_mels=80):
    S_smooth = np.mean(S, axis=1)
    S_smooth = np.repeat(S_smooth[:, None], repeats=n_mels, axis=1)
    
    return S_smooth


def get_common_wav_ids(txt_dir):
    pairs = OrderedDict()
    dirName, subdirList, _ = next(os.walk(txt_dir))
    for i, speaker in enumerate(sorted(subdirList)[:20]):
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        fileList = sorted(fileList)
        for fname in fileList:
            path = os.path.join(dirName,speaker,fname)
            f = open(path, 'r')
            key = ''.join(c for c in f.read() if c.isalpha())
            value = os.path.splitext(fname)[0]
            if key not in pairs:
                pairs[key] = [value]
            else:
                pairs[key].append(value)

    wav_ids = []
    for v in sorted(pairs.values(), key=lambda x: len(x), reverse=False)[-20:]:
        wav_ids += v

    return wav_ids

def get_test_data_set(turk_list_fname = 'turk_list.txt', spk_list_fname = 'spk_list.txt'):
    spk2id = dict()
    with open(spk_list_fname, 'r') as f:
        for line in f:
            speaker, i = line.strip().split(' ')
            spk2id[speaker] = int(i)

    test_data = set()
    test_data_by_ctype = dict()
    curr_key = ''
    with open(turk_list_fname, 'r') as f:
        for line in f:
            line_list = line.strip().split('_')
            if line_list[0] in ['F', 'R', 'U']:
                test_data_by_ctype[line_list[0]] = []
                curr_key = line_list[0]
            elif len(line_list) == 4:
                src_dir = line_list[0]
                tgt_dir = line_list[1]
                wav_id = line_list[2]
                src_name = '_'.join([src_dir, wav_id])
                tgt_name = '_'.join([tgt_dir, wav_id])
                item = ((src_dir+'/'+src_name+'.npy', spk2id[src_dir], src_name), \
                        (tgt_dir+'/'+tgt_name+'.npy', spk2id[tgt_dir], tgt_name))
                if not test_data_by_ctype[curr_key] or item != test_data_by_ctype[curr_key][-1]:
                    test_data_by_ctype[curr_key].append(item)
                src_path1 = src_dir+'/'+'_'.join([src_dir, wav_id[:3], 'mic1.npy'])
                src_path2 = src_dir+'/'+'_'.join([src_dir, wav_id[:3], 'mic2.npy'])
                tgt_path1 = tgt_dir+'/'+'_'.join([tgt_dir, wav_id[:3], 'mic1.npy'])
                tgt_path2 = tgt_dir+'/'+'_'.join([tgt_dir, wav_id[:3], 'mic2.npy'])
                for path in [src_path1, tgt_path1, src_path2, tgt_path2]:
                    test_data.add(path)

    return test_data, test_data_by_ctype