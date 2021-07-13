import os 
import torch
import pickle  
import numpy as np

from torch.utils import data
from torch.utils.data.sampler import Sampler

from utils import quantize_f0_torch

torch.multiprocessing.set_sharing_strategy('file_system')

class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = os.path.join(config.root_dir, config.mode)
        self.wav_dir = os.path.join(self.root_dir, config.wav_dir)
        self.spmel_dir = os.path.join(self.root_dir, config.spmel_dir)
        self.spmel_filt_dir = os.path.join(self.root_dir, config.spmel_filt_dir)
        self.spenv_dir = os.path.join(self.root_dir, config.spenv_dir)
        self.spmel_mono_dir = os.path.join(self.root_dir, config.spmel_mono_dir)
        self.mfcc_dir = os.path.join(self.root_dir, config.mfcc_dir)
        self.f0_dir = os.path.join(self.root_dir, config.f0_dir)
        self.experiment = config.experiment
        self.model_type = config.model_type
        self.mode = config.mode
        print('Currently processing {} dataset'.format(self.mode))

        metaname = os.path.join(self.root_dir, 'dataset.pkl')
        metadata = pickle.load(open(metaname, "rb"))
        
        dataset = [None] * len(metadata)
        self.load_data(metadata, dataset)
        self.dataset = list(dataset)
        self.num_tokens = len(self.dataset)

    def load_data(self, metadata, dataset):  
        for k, sbmt in enumerate(metadata):    
            uttrs = len(sbmt)*[None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]
            uttrs[1] = sbmt[1]
            # fill in data
            wav = np.load(os.path.join(self.wav_dir, sbmt[2])) 
            spmel = np.load(os.path.join(self.spmel_dir, sbmt[2]))
            spmel_filt = np.load(os.path.join(self.spmel_filt_dir, sbmt[2]))
            spmel_mono = np.load(os.path.join(self.spmel_mono_dir, sbmt[2]))
            spenv = np.load(os.path.join(self.spenv_dir, sbmt[2]))
            mfcc = np.load(os.path.join(self.mfcc_dir, sbmt[2]))
            f0 = np.load(os.path.join(self.f0_dir, sbmt[2]))
            uttrs[2] = ( wav, spmel, spmel_filt, spenv, spmel_mono, mfcc, f0 )
            dataset[k] = uttrs  
        

    def __getitem__(self, index):
        list_uttrs = self.dataset[index]
        spk_id_org = list_uttrs[0]
        emb_org = list_uttrs[1]
        wav, spmel, spmel_filt, spenv, spmel_mono, mfcc, f0 = list_uttrs[2]
        if self.experiment == 'spsp1':
            rhythm_input = spmel
            content_input = spmel
            pitch_input = f0
            timbre_input = emb_org
        else:
            rhythm_input = spenv if self.model_type == 'G' else spmel_mono
            content_input = spmel_mono
            pitch_input = f0
            timbre_input = emb_org
        
        return wav, spk_id_org, spmel, rhythm_input, content_input, pitch_input, timbre_input
    

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
        new_batch = []
        for token in batch:

            wav, spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1) if self.mode == 'train' else  self.max_len_pad # 1.5s ~ 3s
            left = np.random.randint(0, len(spmel_gt)-len_crop) if self.mode == 'train' else 0

            spmel_gt = spmel_gt[left:left+len_crop, :] # [Lc, F]
            rhythm_input = rhythm_input[left:left+len_crop, :] # [Lc, F]
            content_input = content_input[left:left+len_crop, :] # [Lc, F]
            pitch_input = pitch_input[left:left+len_crop] # [Lc, ]
            
            spmel_gt = np.clip(spmel_gt, 0, 1)
            rhythm_input = np.clip(rhythm_input, 0, 1)
            content_input = np.clip(content_input, 0, 1)
            
            spmel_gt = np.pad(spmel_gt, ((0,self.max_len_pad-spmel_gt.shape[0]),(0,0)), 'constant')
            rhythm_input = np.pad(rhythm_input, ((0,self.max_len_pad-rhythm_input.shape[0]),(0,0)), 'constant')
            content_input = np.pad(content_input, ((0,self.max_len_pad-content_input.shape[0]),(0,0)), 'constant')
            pitch_input = np.pad(pitch_input[:,np.newaxis], ((0,self.max_len_pad-pitch_input.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop) ) 
            
        batch = new_batch  
        
        spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop = zip(*batch)
        spk_id_org = list(spk_id_org)
        spmel_gt = torch.FloatTensor(np.stack(spmel_gt, axis=0))
        rhythm_input = torch.FloatTensor(np.stack(rhythm_input, axis=0))
        content_input = torch.FloatTensor(np.stack(content_input, axis=0))
        pitch_input = torch.FloatTensor(np.stack(pitch_input, axis=0))
        timbre_input = torch.FloatTensor(np.stack(timbre_input, axis=0))
        len_crop = torch.LongTensor(np.stack(len_crop, axis=0))
        
        return spk_id_org, spmel_gt, rhythm_input, content_input, pitch_input, timbre_input, len_crop
    

    
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
