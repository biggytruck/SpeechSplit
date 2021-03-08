import os
import sys
import pickle
import numpy as np
import soundfile as sf
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from utils import *

def make_spect_f0(config):
    rootDir = config.root_dir
    sourceDir = os.path.join(rootDir, config.src_dir)
    targetDir_wav = os.path.join(rootDir, config.wav_dir)
    targetDir_spmel = os.path.join(rootDir, config.spmel_dir)
    targetDir_spmel_filt = os.path.join(rootDir, config.spmel_filt_dir)
    targetDir_f0 = os.path.join(rootDir, config.f0_dir)
    spk2gen = pickle.load(open(os.path.join(config.root_dir, 'assets/spk2gen.pkl'), "rb"))

    dirName, subdirList, _ = next(os.walk(sourceDir))
    state_count = 1
    print('Found directory: %s' % dirName)

    for subdir in sorted(subdirList):
        print(subdir)
        
        if not os.path.exists(os.path.join(targetDir_wav, subdir)):
            os.makedirs(os.path.join(targetDir_wav, subdir))
        if not os.path.exists(os.path.join(targetDir_spmel, subdir)):
            os.makedirs(os.path.join(targetDir_spmel, subdir))
        if not os.path.exists(os.path.join(targetDir_spmel_filt, subdir)):
            os.makedirs(os.path.join(targetDir_spmel_filt, subdir))
        if not os.path.exists(os.path.join(targetDir_f0, subdir)):
            os.makedirs(os.path.join(targetDir_f0, subdir))    
        _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
        
        if spk2gen[subdir] == 'M':
            lo, hi = 50, 250
        elif spk2gen[subdir] == 'F':
            lo, hi = 100, 600
        else:
            continue

        prng = RandomState(state_count) 
        for fileName in sorted(fileList):
            # read audio file
            x, fs = sf.read(os.path.join(dirName,subdir,fileName))
            assert fs == 16000
            wav = filter_wav(x, prng)
            
            # compute spectrogram
            S = get_spmel(wav)

            # compute filtered spectrogram
            S_filt = smooth_spmel(S)
            S_filt = make_filt_spect(S_filt)
            S_filt = spectral_subtraction(S_filt)
            S_filt = np.abs(S_filt)
            S_filt = zero_one_norm(S_filt)
            
            # extract f0
            f0_rapt, f0_norm = extract_f0(wav, fs, lo, hi)
            
            assert len(S) == len(f0_rapt)

            np.save(os.path.join(targetDir_wav, subdir, os.path.splitext(fileName)[0]),
                    wav.astype(np.float32), allow_pickle=False)     
            np.save(os.path.join(targetDir_spmel, subdir, os.path.splitext(fileName)[0]),
                    S.astype(np.float32), allow_pickle=False) 
            np.save(os.path.join(targetDir_spmel_filt, subdir, os.path.splitext(fileName)[0]),
                    S_filt.astype(np.float32), allow_pickle=False)   
            np.save(os.path.join(targetDir_f0, subdir, os.path.splitext(fileName)[0]),
                    f0_norm.astype(np.float32), allow_pickle=False)

def make_metadata(config):
    targetDir_spmel = os.path.join(config.root_dir, config.spmel_dir)
    targetDir_txt = os.path.join(config.root_dir, config.txt_dir)
    targetDir_meta = os.path.join(config.root_dir, config.meta_dir)

    if not os.path.exists(targetDir_meta):
        os.mkdir(targetDir_meta)

    dirName, subdirList, _ = next(os.walk(targetDir_spmel))
    print('Found directory: %s' % dirName)
    test_wav_ids = get_common_wav_ids(targetDir_txt)

    train_dataset = []
    val_dataset = []
    test_dataset = []
    train_plot_dataset = []
    val_plot_dataset = []
    test_plot_dataset = []

    # use the first 20 speakers only
    for i, speaker in enumerate(sorted(subdirList)[:20]):
        print('Processing speaker: %s' % speaker)

        utterances = []
        utterances.append(speaker)
        _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
        
        # may use generalized speaker embedding for zero-shot conversion
        spkid = np.zeros((82,), dtype=np.float32)
        spkid[i] = 1.0
        utterances.append(spkid)

        # create file list
        fileList = sorted(fileList)
        for fileName in fileList:
            utterances.append(os.path.join(speaker,fileName))

        # train/val/test split
        spk_id = utterances[0]
        spk_emb = utterances[1]
        train_val_utterances, test_utterances = [], []
        for uttr in utterances[2:]:
            if any(test_wav_id in uttr for test_wav_id in test_wav_ids):
                test_utterances.append((spk_id, spk_emb, uttr))
            else:
                train_val_utterances.append((spk_id, spk_emb, uttr))
        train_utterances, val_utterances = train_test_split(train_val_utterances, test_size=0.1, random_state=42)
        train_dataset += train_utterances
        val_dataset += val_utterances
        test_dataset += test_utterances

        # spectrogram for plots
        if speaker == 'p225':
            train_plot_dataset += [train_utterances[0]]
            val_plot_dataset += [val_utterances[0]]
            test_plot_dataset += [test_utterances[0]]
            
    pickleList = ['train.pkl', 'val.pkl', 'test.pkl', 'train_plot.pkl', 'val_plot.pkl', 'test_plot.pkl']
    datasetList = [train_dataset, val_dataset, test_dataset, train_plot_dataset, val_plot_dataset, test_plot_dataset]
    
    for pickleName, dataset in zip(pickleList, datasetList):
        with open(os.path.join(targetDir_meta, pickleName), 'wb') as handle:
            pickle.dump(dataset, handle)