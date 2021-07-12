import os
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from utils import *

def make_spect_f0(config):
    fs = 16000
    mode = config.mode
    root_dir = os.path.join(config.root_dir, mode)
    src_dir = os.path.join(root_dir, config.src_dir)
    wav_dir = os.path.join(root_dir, config.wav_dir)
    spmel_dir = os.path.join(root_dir, config.spmel_dir)
    spmel_filt_dir = os.path.join(root_dir, config.spmel_filt_dir)
    spmel_mono_dir = os.path.join(root_dir, config.spmel_mono_dir)
    spenv_dir = os.path.join(root_dir, config.spenv_dir)
    f0_dir = os.path.join(root_dir, config.f0_dir)
    spk2gen = pickle.load(open('spk2gen.pkl', "rb"))

    dir_name, sub_dir_list, _ = next(os.walk(src_dir))
    state_count = 1
    print('Found directory: %s' % dir_name)

    for sub_dir in sorted(sub_dir_list):
        print(sub_dir)
        
        # create directories if not exist
        for fea_dir in [wav_dir, spmel_dir, spmel_filt_dir, spmel_mono_dir, spenv_dir, f0_dir]:
            if not os.path.exists(os.path.join(fea_dir, sub_dir)):
                os.makedirs(os.path.join(fea_dir, sub_dir))

        _,_, file_list = next(os.walk(os.path.join(dir_name,sub_dir)))
        
        if spk2gen[sub_dir] == 'M':
            lo, hi = 50, 250
        elif spk2gen[sub_dir] == 'F':
            lo, hi = 100, 600
        else:
            continue

        prng = RandomState(state_count) 
        wavs, f0s, sps, aps = [], [], [], []
        for filename in sorted(file_list):
            # read audio file
            x, _ = sf.read(os.path.join(dir_name,sub_dir,filename))
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)

            if mode == 'train':
                # apply high-pass filter
                wav = filter_wav(x, prng)
            else:
                wav = x

            # get WORLD analyzer parameters
            f0, sp, ap = get_world_params(wav, fs)

            wavs.append(wav)
            f0s.append(f0)
            sps.append(sp)
            aps.append(ap)

        if mode == 'train':
            # normalize all f0s to be the global mean
            f0s = average_f0s(f0s, mode='global')
        else:
            # normalize all f0s to be the local mean
            f0s = average_f0s(f0s, mode='local')

        for wav, f0, sp, ap in zip(wavs, f0s, sps, aps):

            # compute spectrogram
            spmel = get_spmel(wav)

            # extract f0
            f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)

            # synthesize monotonic waveforms using WORLD
            wav_mono = get_monotonic_wav(wav, f0, sp, ap, fs)
            
            # compute the monotonic spectrogram
            spmel_mono = get_spmel(wav_mono)

            # compute filtered spectrogram
            spmel_filt = get_spmel_filt(spmel_mono)

            # get spectral envelope
            spenv = get_spenv(wav_mono)
            
            assert len(spmel) == len(spmel_filt) == len(spenv) == len(spmel_mono) == len(f0_rapt)

            if mode == 'train':
                # pad filtered waveform
                start_idx = 0
                trunk_len = 49152
                while start_idx*trunk_len < len(wav):
                    wav_trunk = wav[start_idx*trunk_len:(start_idx+1)*trunk_len]
                    if len(wav_trunk) < trunk_len:
                        wav_trunk = np.pad(wav_trunk, (0, trunk_len-len(wav_trunk)))
                    np.save(os.path.join(wav_dir, sub_dir, os.path.splitext(filename)[0]+'_'+str(start_idx)),
                            wav_trunk.astype(np.float32), allow_pickle=False)
                    start_idx += 1

                # pad other features
                feas = [spmel, spmel_filt, spenv, spmel_mono, f0_norm]
                fea_dirs = [spmel_dir, spmel_filt_dir, spenv_dir, spmel_mono_dir, f0_dir]
                for fea, fea_dir in zip(feas, fea_dirs):
                    start_idx = 0
                    trunk_len = 192
                    while start_idx*trunk_len < len(fea):
                        fea_trunk = fea[start_idx*trunk_len:(start_idx+1)*trunk_len]
                        if len(fea_trunk) < trunk_len:
                            if fea_trunk.ndim==2:
                                fea_trunk = np.pad(fea_trunk, ((0, trunk_len-len(fea_trunk)), (0, 0)))
                            else:
                                fea_trunk = np.pad(fea_trunk, ((0, trunk_len-len(fea_trunk)), ))
                        np.save(os.path.join(fea_dir, sub_dir, os.path.splitext(filename)[0]+'_'+str(start_idx)),
                                fea_trunk.astype(np.float32), allow_pickle=False)
                        start_idx += 1
            else:
                feas = [wav, spmel, spmel_filt, spenv, spmel_mono, f0_norm]
                fea_dirs = [wav_dir, spmel_dir, spmel_filt_dir, spenv_dir, spmel_mono_dir, f0_dir]
                for fea, fea_dir in zip(feas, fea_dirs):
                    np.save(os.path.join(fea_dir, sub_dir, os.path.splitext(filename)[0]),
                            fea.astype(np.float32), allow_pickle=False)


def make_metadata(config):
    root_dir = os.path.join(config.root_dir, config.mode)
    wav_dir = os.path.join(root_dir, config.wav_dir) # use wav directory simply because all inputs have the same filename

    dir_name, _, _ = next(os.walk(wav_dir))
    
    if config.mode == 'test':
        test_data_by_ctype = get_test_data_set(turk_list_fname = './test/turk_list.txt', 
                                               spk_list_fname = './test/spk_list.txt')
        with open(os.path.join(root_dir, 'test_data_by_ctype.pkl'), 'wb') as handle:
            pickle.dump(test_data_by_ctype, handle)

    dataset = []

    with open(os.path.join(root_dir, 'spk_list.txt'), 'r') as f:
        for line in f:
            spk, spk_id = line.strip().split(' ')
            print('Processing speaker: %s; Speaker ID: %s' %(spk, spk_id))
            
            # may use generalized speaker embedding for zero-shot conversion
            spk_emb = np.zeros((82,), dtype=np.float32)
            spk_emb[int(spk_id)] = 1.0

            # create file list
            utterances = []
            _, _, file_list = next(os.walk(os.path.join(dir_name,spk)))
            file_list = sorted(file_list)
            for filename in file_list:
                utterances.append(os.path.join(spk,filename))

            # add to dataset
            for utterance in utterances:
                dataset.append((spk, spk_emb, utterance))

    with open(os.path.join(root_dir, 'dataset.pkl'), 'wb') as handle:
        pickle.dump(dataset, handle)