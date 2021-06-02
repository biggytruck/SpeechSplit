import os
import pickle
from collections import OrderedDict

import jiwer
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from soundfile import read, write
from google.cloud import speech

from config import get_config
from model import D_VECTOR
from utils import *
from sylnet import get_speaking_rate

class Evaluator(object):

    def __init__(self):
        self.invalid_f0 = 0


    # def get_vde(self, f0s, pred_f0s):
    #     Nerr = 0
    #     for f0, pred_f0 in zip(f0s, pred_f0s):
    #         if f0 > 1e-4 and pred_f0 <= 1e-4:
    #             Nerr += 1
    #         elif f0 <= 1e-4 and pred_f0 > 1e-4:
    #             Nerr += 1
        
    #     return Nerr / len(f0s)


    # def get_gpe(self, f0s, pred_f0s, delta=0.2):
    #     Nerr = 0
    #     Nvv = 0
    #     for f0, pred_f0 in zip(f0s, pred_f0s):
    #         if f0 > 1e-4 and pred_f0 > 1e-4:
    #             Nvv += 1
    #             if abs(pred_f0/f0 - 1) > delta:
    #                 Nerr += 1
    #     if Nvv == 0:
    #         self.invalid_f0 += 1
        
    #     return Nerr / Nvv if Nvv else 0


    # def get_ffe(self, f0s, pred_f0s, delta=0.2):
    #     Nerr = 0
    #     for f0, pred_f0 in zip(f0s, pred_f0s):
    #         if f0 > 1e-4 and pred_f0 <= 1e-4:
    #             Nerr += 1
    #         elif f0 <= 1e-4 and pred_f0 > 1e-4:
    #             Nerr += 1
    #         elif f0 > 1e-4 and pred_f0 > 1e-4:
    #             if abs(pred_f0/f0 - 1) > delta:
    #                 Nerr += 1
        
    #     return Nerr / len(f0s)
    def get_vde(self, f0s, pred_f0s):
        Nerr = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 != 0 and pred_f0 == 0:
                Nerr += 1
            elif f0 == 0 and pred_f0 != 0:
                Nerr += 1
        
        return Nerr / len(f0s)


    def get_gpe(self, f0s, pred_f0s, delta=0.8):
        Nerr = 0
        Nvv = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 != 0 and pred_f0 != 0:
                Nvv += 1
                if abs(pred_f0/f0 - 1) > delta:
                    Nerr += 1
        if Nvv == 0:
            self.invalid_f0 += 1
        
        return Nerr / Nvv if Nvv else 0


    def get_ffe(self, f0s, pred_f0s, delta=0.8):
        Nerr = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 != 0 and pred_f0 == 0:
                Nerr += 1
            elif f0 == 0 and pred_f0 != 0:
                Nerr += 1
            elif f0 != 0 and pred_f0 != 0:
                if abs(pred_f0/f0 - 1) > delta:
                    Nerr += 1
        
        return Nerr / len(f0s)


    def evaluate_rhythm(self, fname_dir, fname_list):
        speaking_rate = get_speaking_rate(fname_dir) # key: file name; value: speaking rate(num_syls / voiced_duration)
        src_cnt, tgt_cnt = 0, 0
        for fname in fname_list:
            (src_uttr, src_dur) = speaking_rate[fname+'_s.wav']
            (tgt_uttr, tgt_dur) = speaking_rate[fname+'_t.wav']
            (cvt_uttr, cvt_dur) = speaking_rate[fname+'_c.wav']
            avg_uttr = (src_uttr+tgt_uttr) / 2
            src_rate = avg_uttr / src_dur
            tgt_rate = avg_uttr / tgt_dur
            cvt_rate = cvt_uttr / cvt_dur
            if abs(src_rate-cvt_rate)<abs(tgt_rate-cvt_rate):
                src_cnt += 1
            else:
                tgt_cnt += 1
    
        return {'src_cnt': src_cnt, \
                'tgt_cnt': tgt_cnt}

    def evaluate_pitch(self, fname_dir, fname_list):

        src_vde = 0
        src_gpe = 0
        src_ffe = 0

        for fname in fname_list:

            src_gen = spk2gen[fname.split('_')[0]]

            src_name = os.path.join(fname_dir, fname+'_s.wav')
            src_wav, fs = read(src_name)
            src_f0 = extract_f0(src_wav, fs, lo[src_gen], hi[src_gen])[1]
            src_f0 = quantize_f0_numpy(src_f0)[0]
            src_f0 = inverse_quantize_f0_numpy(src_f0)
            src_f0 = np.pad(src_f0, (0, 192-len(src_f0)), 'constant')

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[src_gen], hi[src_gen])[1]
            cvt_f0 = quantize_f0_numpy(cvt_f0)[0]
            cvt_f0 = inverse_quantize_f0_numpy(cvt_f0)
            cvt_f0 = np.pad(cvt_f0, (0, 192-len(cvt_f0)), 'constant')

            src_vde += self.get_vde(src_f0, cvt_f0)
            src_gpe += self.get_gpe(src_f0, cvt_f0)
            src_ffe += self.get_ffe(src_f0, cvt_f0)

        src_vde /= (len(fname_list)-self.invalid_f0)
        src_gpe /= (len(fname_list)-self.invalid_f0)
        src_ffe /= (len(fname_list)-self.invalid_f0)
        self.invalid_f0 = 0

        tgt_vde = 0
        tgt_gpe = 0
        tgt_ffe = 0
        
        debug = []
        for fname in fname_list:

            tgt_gen = spk2gen[fname.split('_')[0]]

            tgt_name = os.path.join(fname_dir, fname+'_t.npy')
            # tgt_f0_idx = np.load(tgt_name)
            # tgt_f0 = np.zeros((tgt_f0_idx.size, 257))
            # tgt_f0[np.arange(tgt_f0_idx.size),tgt_f0_idx] = 1
            tgt_f0 = np.load(tgt_name)
            tgt_f0 = quantize_f0_numpy(tgt_f0)[0]
            tgt_f0 = inverse_quantize_f0_numpy(tgt_f0)
            tgt_f0 = np.pad(tgt_f0, (0, 192-len(tgt_f0)), 'constant')

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[tgt_gen], hi[tgt_gen])[1]
            cvt_f0 = quantize_f0_numpy(cvt_f0)[0]
            cvt_f0 = inverse_quantize_f0_numpy(cvt_f0)
            cvt_f0 = np.pad(cvt_f0, (0, 192-len(cvt_f0)), 'constant')

            tgt_vde += self.get_vde(tgt_f0, cvt_f0)
            tgt_gpe += self.get_gpe(tgt_f0, cvt_f0)
            tgt_ffe += self.get_ffe(tgt_f0, cvt_f0)

            debug.append([os.path.join(fname_dir, fname), self.get_vde(tgt_f0, cvt_f0), self.get_gpe(tgt_f0, cvt_f0), self.get_ffe(tgt_f0, cvt_f0)])

        tgt_vde /= (len(fname_list)-self.invalid_f0)
        tgt_gpe /= (len(fname_list)-self.invalid_f0)
        tgt_ffe /= (len(fname_list)-self.invalid_f0)
        self.invalid_f0 = 0

        # for x in debug:
        #     print(x)

        return {'src_vde': src_vde, \
                'src_gpe': src_gpe, \
                'src_ffe': src_ffe, \
                'tgt_vde': tgt_vde, \
                'tgt_gpe': tgt_gpe, \
                'tgt_ffe': tgt_ffe}


    def evaluate_timbre(self, fname_dir, fname_list):

        src_cos_sim = 0

        for fname in fname_list:

            src_name = os.path.join(fname_dir, fname+'_s.wav')
            src_wav, fs = read(src_name)
            src_spmel = get_spmel(src_wav).astype(np.float32)
            src_spmel = np.pad(src_spmel, ((0,192-src_spmel.shape[1]), (0,0)), 'constant')
            src_spmel = torch.from_numpy(src_spmel[np.newaxis, :, :]).to(device)
            src_emb = C(src_spmel).detach().cpu().numpy()

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_spmel = get_spmel(cvt_wav).astype(np.float32)
            cvt_spmel = np.pad(cvt_spmel, ((0,192-cvt_spmel.shape[1]), (0,0)), 'constant')
            cvt_spmel = torch.from_numpy(cvt_spmel[np.newaxis, :, :]).to(device)
            cvt_emb = C(cvt_spmel).detach().cpu().numpy()

            src_cos_sim += cosine_similarity(src_emb, cvt_emb)[0]
        
        tgt_cos_sim = 0

        for fname in fname_list:

            tgt_name = os.path.join(fname_dir, fname+'_t.wav')
            tgt_wav, fs = read(tgt_name)
            tgt_spmel = get_spmel(tgt_wav).astype(np.float32)
            tgt_spmel = np.pad(tgt_spmel, ((0,192-tgt_spmel.shape[1]), (0,0)), 'constant')
            tgt_spmel = torch.from_numpy(tgt_spmel[np.newaxis, :, :]).to(device)
            tgt_emb = C(tgt_spmel).detach().cpu().numpy()

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_spmel = get_spmel(cvt_wav).astype(np.float32)
            cvt_spmel = np.pad(cvt_spmel, ((0,192-cvt_spmel.shape[1]), (0,0)), 'constant')
            cvt_spmel = torch.from_numpy(cvt_spmel[np.newaxis, :, :]).to(device)
            cvt_emb = C(cvt_spmel).detach().cpu().numpy()

            tgt_cos_sim += cosine_similarity(tgt_emb, cvt_emb)[0]

        return {'src_cos_smi': src_cos_sim.item()/len(fname_list), \
                'tgt_cos_smi': tgt_cos_sim.item()/len(fname_list)}


if __name__ == '__main__':

    config = get_config()
    config.mode = 'test'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    root_dir = config.root_dir
    src_dir = os.path.join(root_dir, config.mode, config.src_dir)
    spmel_dir = os.path.join(root_dir, config.mode, config.spmel_dir)
    spmel_filt_dir = os.path.join(root_dir, config.mode, config.spmel_filt_dir)
    f0_dir = os.path.join(root_dir, config.mode, config.f0_dir)

    model_save_dir = os.path.join(root_dir, config.model_save_dir)
    result_dir = os.path.join(root_dir, config.result_dir)
    plot_dir = os.path.join(root_dir, config.plot_dir)

    test_data_by_ctype = pickle.load(open(os.path.join(root_dir, config.mode, 'test_data_by_ctype.pkl'), 'rb'))
    spk2gen = pickle.load(open('./spk2gen.pkl', 'rb'))
    lo = {'M': 50, 'F': 100}
    hi = {'M': 250, 'F': 600}

    C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().to(device)
    c_checkpoint = torch.load(os.path.join(result_dir, '3000000-BL.ckpt'), map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for key, val in c_checkpoint['model_b'].items():
        new_key = key[7:]
        new_state_dict[new_key] = val
    C.load_state_dict(new_state_dict)

    E = Evaluator()

    model_type_list = [
        'spsp1',
        'spsp2',
    ]

    settings = {
                'R_8_1': [8,8,8,8,1,32],
                # 'R_1_1': [8,1,8,8,1,32],
                # 'R_8_32': [8,8,8,8,32,32],
                # 'R_1_32': [8,1,8,8,32,32],
                }

    ctype_list = [
        'F',
        # 'C',
        'R',
        'U',
    ]

    # initialize metrics
    metrics = {}
    for model_type in model_type_list:
        metrics[model_type] = {}
        for model_name in settings.keys():
            metrics[model_type][model_name] = {}
            for ctype in ctype_list:
                metrics[model_type][model_name][ctype] = {}

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

                for ctype in ctype_list:
                    pairs = test_data_by_ctype[ctype]
                    fname_list = []
                    for (src_name, src_id), (tgt_name, tgt_id) in pairs:
                        fname_list.append(src_name.split('/')[-1]+'_'+tgt_name.split('/')[-1])
                    fname_dir = os.path.join(result_dir, model_type, model_name, ctype)
                    metrics[model_type][model_name][ctype] = {
                        'pitch_metrics': E.evaluate_pitch(fname_dir, fname_list),
                        'rhythm_metrics': E.evaluate_rhythm(fname_dir, fname_list),
                        'timbre_metrics': E.evaluate_timbre(fname_dir, fname_list),
                    }

    dict2json(metrics, 'metrics.json')