import os
import sys
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from utils import *

def make_spect_f0(config):
    root_dir = os.path.join(config.root_dir, config.mode)
    src_dir = os.path.join(root_dir, config.src_dir)
    wav_dir = os.path.join(root_dir, config.wav_dir)
    spmel_dir = os.path.join(root_dir, config.spmel_dir)
    spmel_filt_dir = os.path.join(root_dir, config.spmel_filt_dir)
    spenv_dir = os.path.join(root_dir, config.spenv_dir)
    f0_dir = os.path.join(root_dir, config.f0_dir)
    spk2gen = pickle.load(open('spk2gen.pkl', "rb"))

    dir_name, sub_dir_list, _ = next(os.walk(src_dir))
    state_count = 1
    print('Found directory: %s' % dir_name)

    for sub_dir in sorted(sub_dir_list):
        print(sub_dir)
        
        if not os.path.exists(os.path.join(wav_dir, sub_dir)):
            os.makedirs(os.path.join(wav_dir, sub_dir))
        if not os.path.exists(os.path.join(spmel_dir, sub_dir)):
            os.makedirs(os.path.join(spmel_dir, sub_dir))
        if not os.path.exists(os.path.join(spmel_filt_dir, sub_dir)):
            os.makedirs(os.path.join(spmel_filt_dir, sub_dir))
        if not os.path.exists(os.path.join(spenv_dir, sub_dir)):
            os.makedirs(os.path.join(spenv_dir, sub_dir))
        if not os.path.exists(os.path.join(f0_dir, sub_dir)):
            os.makedirs(os.path.join(f0_dir, sub_dir))    
        _,_, file_list = next(os.walk(os.path.join(dir_name,sub_dir)))
        
        if spk2gen[sub_dir] == 'M':
            lo, hi = 50, 250
        elif spk2gen[sub_dir] == 'F':
            lo, hi = 100, 600
        else:
            continue

        prng = RandomState(state_count) 
        for filename in sorted(file_list):
            # read audio file
            x, fs = sf.read(os.path.join(dir_name,sub_dir,filename))
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            assert fs == 16000
            if config.mode == 'train': 
                wav = filter_wav(x, prng)
            else:
                wav = x
            
            # compute spectrogram
            spmel = get_spmel(wav)

            # compute filtered spectrogram
            spmel_filt = get_spmel_filt(spmel)

            # get spectral envelope
            spenv = get_spenv(wav)
            
            # extract f0
            f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)
            
            assert len(spmel) == len(f0_rapt)

            np.save(os.path.join(wav_dir, sub_dir, os.path.splitext(filename)[0]),
                    wav.astype(np.float32), allow_pickle=False)     
            np.save(os.path.join(spmel_dir, sub_dir, os.path.splitext(filename)[0]),
                    spmel.astype(np.float32), allow_pickle=False) 
            np.save(os.path.join(spmel_filt_dir, sub_dir, os.path.splitext(filename)[0]),
                    spmel_filt.astype(np.float32), allow_pickle=False) 
            np.save(os.path.join(spenv_dir, sub_dir, os.path.splitext(filename)[0]),
                    spenv.astype(np.float32), allow_pickle=False) 
            np.save(os.path.join(f0_dir, sub_dir, os.path.splitext(filename)[0]),
                    f0_norm.astype(np.float32), allow_pickle=False)

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