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
from utils import quantize_f0_numpy, inverse_quantize_f0_numpy, tensor2onehot

class Converter(object):

    def __init__(self) -> None:
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = get_config()
        self.config.mode = 'test'

        self.root_dir = self.config.root_dir
        self.mode = self.config.mode
        self.src_dir = os.path.join(self.root_dir, self.mode, self.config.src_dir)
        self.spmel_dir = os.path.join(self.root_dir, self.mode, self.config.spmel_dir)
        self.spmel_filt_dir = os.path.join(self.root_dir, self.mode, self.config.spmel_filt_dir)
        self.spmel_mono_dir = os.path.join(self.root_dir, self.mode, self.config.spmel_mono_dir)
        self.spenv_dir = os.path.join(self.root_dir, self.mode, self.config.spenv_dir)
        self.f0_dir = os.path.join(self.root_dir, self.mode, self.config.f0_dir)

        self.model_save_dir = os.path.join(self.root_dir, self.config.model_save_dir)
        self.result_dir = os.path.join(self.root_dir, self.config.result_dir)
        self.plot_dir = os.path.join(self.root_dir, self.config.plot_dir)

        self.metadata = pickle.load(open(os.path.join(self.root_dir, self.config.mode, 'test_data_by_ctype.pkl'), 'rb'))
        self.fs = 16000
        self.S = Synthesizer(self.device)
        self.S.load_ckpt(os.path.join(self.result_dir, 'wavenet_vocoder.pth'))

    def load_model(self):
        self.G = Generator(self.config).eval().to(self.device)
        ckpt = torch.load(os.path.join(self.model_save_dir, self.config.experiment, self.config.model_name+'-G-best.ckpt'))
        try:
            self.G.load_state_dict(ckpt['model'])
        except:
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                new_state_dict[k[7:]] = v
            self.G.load_state_dict(new_state_dict)

        # self.F = F0_Converter(self.config).eval().to(self.device)
        # ckpt = torch.load(os.path.join(self.model_save_dir, self.config.experiment, self.config.model_name+'-G-best.ckpt'))
        # try:
        #     self.F.load_state_dict(ckpt['model'])
        # except:
        #     new_state_dict = OrderedDict()
        #     for k, v in ckpt['model'].items():
        #         new_state_dict[k[7:]] = v
        #     self.F.load_state_dict(new_state_dict)

    def _load_fea(self, fea_dir, fea_name):
        fea = np.load(os.path.join(fea_dir, fea_name))
        if fea.ndim == 1:
            fea = quantize_f0_numpy(fea)[0]
        if len(fea)%8 != 0:
            len_pad = 8 - (len(fea)%8)
            fea = np.pad(fea, ((0,len_pad), (0,0)), 'constant')
        fea = fea[np.newaxis, :, :]
        
        return fea

    def _get_spk_emb(self, spk_id):
        spk_emb = np.zeros((1, 82))
        spk_emb[0, spk_id] = 1

        return spk_emb

    def _convert_pitch(self, rhythm_input, pitch_input):
        T = 192
        rhythm_input = np.pad(rhythm_input, ((0,0), (0,T-rhythm_input.shape[1]), (0,0)), 'constant')
        pitch_input = np.pad(pitch_input, ((0,0), (0,T-pitch_input.shape[1]), (0,0)), 'constant')
        rhythm_input = torch.from_numpy(rhythm_input).float().to(self.device)
        pitch_input = torch.from_numpy(pitch_input).float().to(self.device)
        pitch_output = self.F(rhythm_input, pitch_input, rr=False)
        pitch_output = tensor2onehot(pitch_output)
        
        return pitch_output.cpu().numpy()

    def load_fea(self, ctype, src_path, tgt_path, src_id, tgt_id):
        path_order = {
            'R': [tgt_path, src_path, src_path, src_id],
            'F': [src_path, src_path, tgt_path, src_id],
            'U': [src_path, src_path, src_path, tgt_id],
        }
        if self.config.experiment == 'spsp1':
            rhythm_input = self._load_fea(self.spmel_dir, path_order[ctype][0])
            content_input = self._load_fea(self.spmel_dir, path_order[ctype][1])
            pitch_input = self._load_fea(self.f0_dir, path_order[ctype][2])
            timbre_input = self._get_spk_emb(path_order[ctype][3])
        else:
            rhythm_input1 = self._load_fea(self.spmel_filt_dir, path_order[ctype][0])
            rhythm_input2 = self._load_fea(self.spenv_dir, path_order[ctype][0])
            rhythm_input = np.concatenate((rhythm_input1, rhythm_input2), axis=-1)
            content_input = self._load_fea(self.spmel_mono_dir, path_order[ctype][1])
            pitch_input = self._load_fea(self.f0_dir, path_order[ctype][2])
            if ctype == 'F':
                pitch_input = self._convert_pitch(rhythm_input[:, :, 1:], pitch_input)
            timbre_input = self._get_spk_emb(path_order[ctype][3])
        
        return rhythm_input, content_input, pitch_input, timbre_input

    def convert_spmel(self, rhythm_input, content_input, pitch_input, timbre_input):
        T = 192
        rhythm_input = np.pad(rhythm_input, ((0,0), (0,T-rhythm_input.shape[1]), (0,0)), 'constant')
        content_input = np.pad(content_input, ((0,0), (0,T-content_input.shape[1]), (0,0)), 'constant')
        pitch_input = np.pad(pitch_input, ((0,0), (0,T-pitch_input.shape[1]), (0,0)), 'constant')
        pitch_input_1d = inverse_quantize_f0_numpy(pitch_input[0])

        rhythm_input = torch.FloatTensor(rhythm_input).to(self.device)
        content_input = torch.FloatTensor(content_input).to(self.device)
        pitch_input = torch.FloatTensor(pitch_input).to(self.device)
        timbre_input = torch.FloatTensor(timbre_input).to(self.device)
        content_pitch_input = torch.cat((content_input, pitch_input), dim=-1)

        rhythm_input_numpy = rhythm_input[0].cpu().numpy()
        content_input_numpy = content_input[0].cpu().numpy()
        pitch_input_numpy = pitch_input[0].cpu().numpy()

        rhythm_output = self.G.rhythm(rhythm_input)
        content_output, pitch_output = self.G.content_pitch(content_pitch_input, rr=False)
        spmel_output = self.G.decode(content_output, rhythm_output, pitch_output, timbre_input, T).cpu().numpy()[0]
        
        return spmel_output, pitch_input_1d, [rhythm_input_numpy, content_input_numpy, pitch_input_numpy]

    def draw_plot(self, images, names, plot_path):
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

    def convert(self, experiments, settings, ctypes):
        spmel_list = []
        spmel_path_list = []

        # use spsp1 pitch converter
        from script import Generator_6 as F0_Converter_ORI
        self.F = F0_Converter_ORI(self.config).eval().to(self.device)
        ckpt = torch.load(os.path.join(self.model_save_dir, 'spsp1/R_8_1-F-best.ckpt'))
        try:
            self.F.load_state_dict(ckpt['model'])
        except:
            new_state_dict = OrderedDict()
            for k, v in ckpt['model'].items():
                new_state_dict[k[7:]] = v
            self.F.load_state_dict(new_state_dict)

        with torch.no_grad():
            for experiment in experiments:
                for model_name, params in settings.items():

                    self.config.model_name = model_name
                    self.config.experiment = experiment
                    self.config.freq = params[0]
                    self.config.freq_2 = params[1]
                    self.config.freq_3 = params[2]
                    self.config.dim_neck = params[3]
                    self.config.dim_neck_2 = params[4]
                    self.config.dim_neck_3 = params[5]

                    self.load_model()
    
                    for ctype in ctypes:
                        pairs = self.metadata[ctype]
                        for (src_name, src_id), (tgt_name, tgt_id) in pairs[:40]:
                            fname = src_name.split('/')[-1]+'_'+tgt_name.split('/')[-1]
                            result_path = os.path.join(self.result_dir, experiment, model_name, ctype)
                            if not os.path.exists(result_path):
                                os.makedirs(result_path)

                            # save source waveform
                            src_wav, _ = read(os.path.join(self.src_dir, src_name+'.wav'))
                            src_save_path = os.path.join(result_path, fname+'_s.wav')
                            write(src_save_path, src_wav, self.fs)

                            # save target waveform
                            tgt_wav, _ = read(os.path.join(self.src_dir, tgt_name+'.wav'))
                            tgt_save_path = os.path.join(result_path, fname+'_t.wav')
                            write(tgt_save_path, tgt_wav, self.fs)

                            # get conversion results
                            cvt_spmel_path = os.path.join(result_path, fname+'_c.wav')
                            rhythm_input, content_input, pitch_input, timbre_input = self.load_fea(ctype, src_name+'.npy', tgt_name+'.npy', src_id, tgt_id)
                            cvt_spmel, pitch_input_1d, inputs_numpy = self.convert_spmel(rhythm_input, content_input, pitch_input, timbre_input)
                            spmel_list.append(torch.from_numpy(cvt_spmel))
                            spmel_path_list.append(cvt_spmel_path)
                            if ctype == 'F':
                                pitch_input_1d_path = os.path.join(result_path, fname+'_t.npy')
                                np.save(pitch_input_1d_path, pitch_input_1d)
                            
                            # make plots
                            if not os.path.exists(os.path.join(self.plot_dir, experiment, model_name, ctype)):
                                os.makedirs(os.path.join(self.plot_dir, experiment, model_name, ctype))

                            # plot input
                            src_f0 = np.load(os.path.join(self.f0_dir, src_name+'.npy'))
                            src_f0 = quantize_f0_numpy(src_f0)[0]
                            tgt_f0 = np.load(os.path.join(self.f0_dir, tgt_name+'.npy'))
                            tgt_f0 = quantize_f0_numpy(tgt_f0)[0]
                            images = inputs_numpy + [src_f0, tgt_f0]
                            names = ['Rhythm Input', 'Content Input', 'Pitch Input', 'Source Pitch', 'Target Pitch']
                            plot_path = os.path.join(self.plot_dir, experiment, model_name, ctype, fname+'_input.png')
                            self.draw_plot(images, names, plot_path)

                            # plot output
                            src_spmel = np.load(os.path.join(self.spmel_dir, src_name+'.npy'))
                            tgt_spmel = np.load(os.path.join(self.spmel_dir, tgt_name+'.npy'))
                            images = [src_spmel, tgt_spmel, cvt_spmel]
                            names = ['Source Mel-Spectrogram', 'Target Mel-Spectrogram', 'Converted Mel-Spectrogram']
                            plot_path = os.path.join(self.plot_dir, experiment, model_name, ctype, fname+'_output.png')
                            self.draw_plot(images, names, plot_path)

                i = 0
                batch_size = 120
                print(len(spmel_path_list), len(spmel_list))
                while i<len(spmel_list):
                    cvt_spmel_batch = torch.nn.utils.rnn.pad_sequence(spmel_list[i:i+batch_size], batch_first=True)
                    cvt_wav_batch = self.S.batch_spect2wav(c=cvt_spmel_batch)
                    for path, wav in zip(spmel_path_list[i:i+batch_size], cvt_wav_batch):
                        print(path)
                        write(path, wav, self.fs)
                    i += batch_size
                spmel_list = []
                spmel_path_list = []


if __name__ == '__main__':
    converter = Converter()

    experiments = [
        # 'spsp1',
        'spsp2',
    ]
    settings = {
                # 'R_8_1': [8,8,8,8,1,32],
                'R_1_32': [1,1,1,32,32,32],
                }

    ctypes = [
        'F',
        'R',
        'U',
    ]
    converter.convert(experiments, settings, ctypes)