import os 
import torch
import pickle  
import numpy as np
from random import choice

from torch.utils import data
from torch.utils.data.sampler import Sampler
from torchaudio.sox_effects import apply_effects_tensor

from utils import get_spmel, get_spenv


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = os.path.join(config.root_dir, config.mode)
        self.wav_dir = os.path.join(self.root_dir, config.wav_dir)
        self.spmel_dir = os.path.join(self.root_dir, config.spmel_dir)
        self.spmel_filt_dir = os.path.join(self.root_dir, config.spmel_filt_dir)
        self.spenv_dir = os.path.join(self.root_dir, config.spenv_dir+str(config.cutoff))
        self.f0_dir = os.path.join(self.root_dir, config.f0_dir)
        self.mode = config.mode
        self.step = 300
        print('Currently processing {} dataset'.format(config.mode))

        metaname = os.path.join(self.root_dir, 'dataset.pkl')
        meta = pickle.load(open(metaname, "rb"))
        
        dataset = [None] * len(meta)
        self.load_data(meta, dataset, 0)
        self.dataset = list(dataset)
        self.num_tokens = len(self.dataset)

    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]
            uttrs[1] = sbmt[1]
            # fill in data
            wav_tmp = np.load(os.path.join(self.wav_dir, sbmt[2])) 
            sp_tmp = np.load(os.path.join(self.spmel_dir, sbmt[2]))
            sp_filt_tmp = np.load(os.path.join(self.spmel_filt_dir, sbmt[2]))
            f0_tmp = np.load(os.path.join(self.f0_dir, sbmt[2]))
            uttrs[2] = ( wav_tmp, sp_tmp, sp_filt_tmp, f0_tmp )
            dataset[idx_offset+k] = uttrs  
        

    def __getitem__(self, index):
        list_uttrs = self.dataset[index]
        spk_id_org = list_uttrs[0]
        emb_org = list_uttrs[1]
        wav_tmp, melsp, melsp_filt, f0_org = list_uttrs[2]
        
        return wav_tmp, spk_id_org, melsp, melsp_filt, emb_org, f0_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    


class Collator(object):
    def __init__(self, config):
        self.min_len_seq = config.min_len_seq
        self.max_len_seq = config.max_len_seq
        self.max_len_pad = config.max_len_pad
        self.mode = config.mode

    def __call__(self, batch):
        cent = choice([np.random.randint(-400, -200), np.random.uniform(200, 400)])
        effects = [
            ['pitch', str(cent)],
            ['rate', '16000']
        ]
        wav_tmps = torch.FloatTensor([token[0] for token in batch])
        wav_shifts = apply_effects_tensor(wav_tmps, 16000, effects, channels_first=True)[0].numpy()
        
        new_batch = []
        for wav_shift, token in zip(wav_shifts, batch):

            spk_id_org, melsp, melsp_filt, emb_org, f0_org = token[1:]
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1) if self.mode == 'train' else  self.max_len_pad # 1.5s ~ 3s
            left = np.random.randint(0, len(melsp)-len_crop) if self.mode == 'train' else 0

            melsp = melsp[left:left+len_crop, :] # [Lc, F]
            melsp_filt = melsp_filt[left:left+len_crop, :] # [Lc, 1]
            melse = get_spenv(wav_shift)[left:left+len_crop, :] # [Lc, F]
            melsp_R = np.hstack((melsp_filt, melse))
            melsp_C = get_spmel(wav_shift)[left:left+len_crop, :] # [Lc, F]
            f0_org = f0_org[left:left+len_crop] # [Lc, ]
            
            melsp = np.clip(melsp, 0, 1)
            melsp_R = np.clip(melsp_R, 0, 1)
            melsp_C = np.clip(melsp_C, 0, 1)
            
            melsp_pad = np.pad(melsp, ((0,self.max_len_pad-melsp.shape[0]),(0,0)), 'constant')
            melsp_R_pad = np.pad(melsp_R, ((0,self.max_len_pad-melsp_R.shape[0]),(0,0)), 'constant')
            melsp_C_pad = np.pad(melsp_C, ((0,self.max_len_pad-melsp_C.shape[0]),(0,0)), 'constant')
            f0_org_pad = np.pad(f0_org[:,np.newaxis], ((0,self.max_len_pad-f0_org.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (spk_id_org, melsp_pad, melsp_R_pad, melsp_C_pad, emb_org, f0_org_pad, len_crop) ) 
            
        batch = new_batch  
        
        spk_id_org, melsp_pad, melsp_R_pad, melsp_C_pad, emb_org, f0_org_pad, len_crop = zip(*batch)
        spk_id_org = list(spk_id_org)
        melsp = torch.FloatTensor(np.stack(melsp_pad, axis=0))
        melsp_R = torch.FloatTensor(np.stack(melsp_R_pad, axis=0))
        melsp_C = torch.FloatTensor(np.stack(melsp_C_pad, axis=0))
        spk_emb = torch.FloatTensor(np.stack(emb_org, axis=0))
        pitch = torch.FloatTensor(np.stack(f0_org_pad, axis=0))
        len_org = torch.LongTensor(np.stack(len_crop, axis=0))
        
        return spk_id_org, melsp, melsp_R, melsp_C, spk_emb, pitch, len_org
    

    
class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    """
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)        



def get_loader(config):
    """Build and return a data loader list."""

    dataset = Utterances(config)
    collator = Collator(config)
    sampler = MultiSampler(len(dataset), config.samplier, shuffle=config.shuffle)
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    
    data_loader = data.DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 sampler=sampler,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True,
                                 worker_init_fn=worker_init_fn,
                                 collate_fn=collator)

    return data_loader
