import os
import pickle
from collections import OrderedDict

from soundfile import read
import jiwer
from google.cloud import speech
from utils import extract_f0, dict2json, filter_wav, get_spmel
from model import D_VECTOR
import numpy as np
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
import torch

from run_SylNet import get_speaking_rate


wav_dir = 'eval/assets/wavs/'
spmel_dir = 'eval/assets/spmel/'
spmel_filt_dir = 'eval/assets/spmel_filt/'
raptf0_dir = 'eval/assets/raptf0/'
txt_dir = 'eval/assets/txt/'
model_dir = 'run/models/speech-split2-find-content-optimal/'
result_dir = 'eval/results'
plot_dir = 'eval/plots'
spk2gen = pickle.load(open('eval/assets/spk2gen.pkl', 'rb'))
C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('eval/assets/3000000-BL.ckpt')
new_state_dict = OrderedDict()
for key, val in c_checkpoint['model_b'].items():
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)
prng = RandomState(1)
txt_dict = {
    '001': 'Please call Stella',
    '010': 'People look, but no one ever finds it',
    '003001': 'Six spoons of fresh snow peas',
    '003002': 'five thick slabs of blue cheese',
    '005002': 'and we will go meet her Wednesday',
    '006001': 'When the sunlight strikes raindrops in the air',
    '008001': 'These take the shape of a long round arch',
    '024001': 'This is a very common type of bow',
    '024002': 'one showing mainly red and yellow',
    '024003': 'with little or no green or blue'
}

class Evaluator(object):

    def __init__(self):

        """Initialize GCP speech recognizer"""
        self.sr_client = speech.SpeechClient()
        self.sr_config = speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                            sample_rate_hertz=16000,
                            language_code="en-US",
                        )
        self.wer_transform = jiwer.Compose([
                                jiwer.ToLowerCase(),
                                jiwer.RemoveMultipleSpaces(),
                                jiwer.RemovePunctuation(),
                                jiwer.RemoveWhiteSpace(replace_by_space=False)
                                # jiwer.SentencesToListOfWords(word_delimiter=" ")
                            ]) 
        self.invalid_f0 = 0


    def get_asr_result(self, content):
        """
        Note that transcription is limited to a 60 seconds audio file.
        Use a GCS file for audio longer than 1 minute.
        """
        audio = speech.RecognitionAudio(content=content)
        operation = self.sr_client.long_running_recognize(config=self.sr_config, audio=audio)
        response = operation.result(timeout=90)
        text = []

        # Each result is for a consecutive portion of the audio. Iterate through
        # them to get the transcripts for the entire audio file.
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            text.append(result.alternatives[0].transcript)

        return " ".join(text)


    def get_wer(self, txt, pred_txt):

        return jiwer.wer(txt, pred_txt, truth_transform=self.wer_transform, hypothesis_transform=self.wer_transform)


    def get_vde(self, f0s, pred_f0s):
        Nerr = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 > 1e-6 and pred_f0 <= 1e-6:
                Nerr += 1
            elif f0 <= 1e-6 and pred_f0 > 1e-6:
                Nerr += 1
        
        return Nerr / len(f0s)


    def get_gpe(self, f0s, pred_f0s, delta=0.2):
        Nerr = 0
        Nvv = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 > 1e-6 and pred_f0 > 1e-6:
                Nvv += 1
                if abs(pred_f0/f0 - 1) > delta:
                    Nerr += 1
        if Nvv == 0:
            self.invalid_f0 += 1
        
        return Nerr / Nvv if Nvv else 0


    def get_ffe(self, f0s, pred_f0s, delta=0.2):
        Nerr = 0
        for f0, pred_f0 in zip(f0s, pred_f0s):
            if f0 > 1e-6 and pred_f0 <= 1e-6:
                Nerr += 1
            elif f0 <= 1e-6 and pred_f0 > 1e-6:
                Nerr += 1
            elif f0 > 1e-6 and pred_f0 > 1e-6:
                if abs(pred_f0/f0 - 1) > delta:
                    Nerr += 1
        
        return Nerr / len(f0s)

    
    def _get_content_from_file(self, fname):
        with open(fname, "rb") as audio_file:
            content = audio_file.read()

        return content

    def evaluate_rhythm(self, fname_dir, fname_list):
        speaking_rate = get_speaking_rate(fname_dir) # key: file name; value: speaking rate(num_syls / voiced_duration)
        src_cnt, tgt_cnt = 0, 0
        for fname in fname_list:
            (src_rate, src_dur) = speaking_rate[fname+'_s.wav']
            (tgt_rate, tgt_dur) = speaking_rate[fname+'_t.wav']
            (cvt_rate, cvt_dur) = speaking_rate[fname+'_c.wav']
            avg_rate = (src_rate+tgt_rate) / 2
            src = avg_rate / src_dur
            tgt = avg_rate / tgt_dur
            cvt = cvt_rate / cvt_dur
            if abs(tgt-cvt)<abs(src-cvt):
                tgt_cnt += 1
            else:
                src_cnt += 1
    
        return {'src_cnt': src_cnt, \
                'tgt_cnt': tgt_cnt}


    def evaluate_content(self, fname_dir, fname_list):

        src_wer = 0
        cvt_wer = 0

        for fname in fname_list:

            src_name = os.path.join(fname_dir, fname+'_s.wav')
            src_content = self._get_content_from_file(src_name)
            src_txt = self.get_asr_result(src_content)

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_content = self._get_content_from_file(cvt_name)
            cvt_txt = self.get_asr_result(cvt_content)

            tgt_wav_id = fname.split('_')[3]
            tgt_txt = txt_dict[tgt_wav_id]

            src_wer += self.get_wer(tgt_txt, src_txt)
            cvt_wer += self.get_wer(tgt_txt, cvt_txt)

        src_wer /= len(fname_list)
        cvt_wer /= len(fname_list)
        rel_wer = (cvt_wer - src_wer) / src_wer

        return {'src_wer': src_wer, \
                'cvt_wer': cvt_wer, \
                'rel_wer': rel_wer}

    def evaluate_pitch(self, fname_dir, fname_list):

        lo = {'M': 50, 'F': 100}
        hi = {'M': 250, 'F': 600}

        src_vde = 0
        src_gpe = 0
        src_ffe = 0
        
        for fname in fname_list:

            src_gen = spk2gen[fname.split('_')[0]]

            src_name = os.path.join(fname_dir, fname+'_s.wav')
            src_wav, fs = read(src_name)
            src_f0 = extract_f0(src_wav, fs, lo[src_gen], hi[src_gen])[1]
            src_f0 = np.pad(src_f0, (0, 192-len(src_f0)), 'constant')

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[src_gen], hi[src_gen])[1]
            cvt_f0 = np.pad(cvt_f0, (0, 192-len(cvt_f0)), 'constant')

            src_vde += self.get_vde(src_f0, cvt_f0)
            src_gpe += self.get_gpe(src_f0, cvt_f0)
            src_ffe += self.get_ffe(src_f0, cvt_f0)

        src_vde /= len(fname_list)-self.invalid_f0
        src_gpe /= len(fname_list)-self.invalid_f0
        src_ffe /= len(fname_list)-self.invalid_f0
        self.invalid_f0 = 0

        tgt_vde = 0
        tgt_gpe = 0
        tgt_ffe = 0
        
        for fname in fname_list:

            tgt_gen = spk2gen[fname.split('_')[2]]

            tgt_name = os.path.join(fname_dir, fname+'_t.wav')
            tgt_wav, fs = read(tgt_name)
            tgt_f0 = extract_f0(tgt_wav, fs, lo[tgt_gen], hi[tgt_gen])[1]
            tgt_f0 = np.pad(tgt_f0, (0, 192-len(tgt_f0)), 'constant')

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[tgt_gen], hi[tgt_gen])[1]
            cvt_f0 = np.pad(cvt_f0, (0, 192-len(cvt_f0)), 'constant')

            tgt_vde += self.get_vde(tgt_f0, cvt_f0)
            tgt_gpe += self.get_gpe(tgt_f0, cvt_f0)
            tgt_ffe += self.get_ffe(tgt_f0, cvt_f0)

        tgt_vde /= len(fname_list)-self.invalid_f0
        tgt_gpe /= len(fname_list)-self.invalid_f0
        tgt_ffe /= len(fname_list)-self.invalid_f0
        self.invalid_f0 = 0

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
            src_wav = filter_wav(src_wav, prng)
            src_spmel = get_spmel(src_wav).astype(np.float32)
            src_spmel = torch.from_numpy(src_spmel[np.newaxis, :, :]).cuda()
            src_emb = C(src_spmel).detach().cpu().numpy()

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_wav = filter_wav(cvt_wav, prng)
            cvt_spmel = get_spmel(cvt_wav).astype(np.float32)
            cvt_spmel = torch.from_numpy(cvt_spmel[np.newaxis, :, :]).cuda()
            cvt_emb = C(cvt_spmel).detach().cpu().numpy()

            src_cos_sim += cosine_similarity(src_emb, cvt_emb)[0]
        
        tgt_cos_sim = 0

        for fname in fname_list:

            tgt_name = os.path.join(fname_dir, fname+'_t.wav')
            tgt_wav, fs = read(tgt_name)
            tgt_wav = filter_wav(tgt_wav, prng)
            tgt_spmel = get_spmel(tgt_wav).astype(np.float32)
            tgt_spmel = torch.from_numpy(tgt_spmel[np.newaxis, :, :]).cuda()
            tgt_emb = C(tgt_spmel).detach().cpu().numpy()

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_wav = filter_wav(cvt_wav, prng)
            cvt_spmel = get_spmel(cvt_wav).astype(np.float32)
            cvt_spmel = torch.from_numpy(cvt_spmel[np.newaxis, :, :]).cuda()
            cvt_emb = C(cvt_spmel).detach().cpu().numpy()

            tgt_cos_sim += cosine_similarity(tgt_emb, cvt_emb)[0]

        return {'src_cos_smi': src_cos_sim.item()/len(fname_list), \
                'tgt_cos_smi': tgt_cos_sim.item()/len(fname_list)}

if __name__ == '__main__':

    e = Evaluator()
    test_data_by_ctype = pickle.load(open('eval/assets/test_data_by_ctype.pkl', 'rb'))
    model_type_list = [
        'spsp1',
        # 'spsp2',
    ]

    model_name_list = {
        'R_8_1',
        # 'R_1_1',
        # 'R_8_32',
        # 'R_1_32',
        # 'wide_CR_8_8',
    }

    ctype_list = [
        'F',
        # 'C',
        # 'R',
        # 'U',
    ]

    # initialize metrics
    metrics = {}
    for model_type in model_type_list:
        metrics[model_type] = {}
        for model_name in model_name_list:
            metrics[model_type][model_name] = {}
            for ctype in ctype_list:
                metrics[model_type][model_name][ctype] = {}

    # get metrics
    for model_type in model_type_list:
        for model_name in model_name_list:
            for ctype in ctype_list:
                pairs = test_data_by_ctype[ctype]
                fname_list = []
                for (src_name, src_id), (tgt_name, tgt_id) in pairs:
                    fname_list.append(src_name.split('/')[-1]+'_'+tgt_name.split('/')[-1])
                fname_dir = os.path.join(result_dir, model_type, model_name, ctype)

                if ctype in ['F', 'R', 'U', 'C']:
                    metrics[model_type][model_name][ctype] = {
                        'pitch_metrics': e.evaluate_pitch(fname_dir, fname_list),
                        # 'content_metrics': e.evaluate_content(fname_dir, fname_list),
                        'rhythm_metrics': e.evaluate_rhythm(fname_dir, fname_list),
                        'timbre_metrics': e.evaluate_timbre(fname_dir, fname_list),
                    }

    dict2json(metrics, 'metrics.json')