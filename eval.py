import os
import pickle

from soundfile import read
import jiwer
from google.cloud import speech
from utils import extract_f0, dict2json
import numpy as np

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

class Evaluator(object):

    def __init__(self):

        """Initialize GCP speech recognizer"""
        # self.sr_client = speech.SpeechClient()
        # self.sr_config = speech.RecognitionConfig(
        #                     encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        #                     sample_rate_hertz=16000,
        #                     language_code="en-US",
        #                 )
        self.wer_transform = jiwer.Compose([
                                jiwer.ToLowerCase(),
                                jiwer.RemoveMultipleSpaces(),
                                jiwer.RemovePunctuation(),
                                jiwer.RemoveWhiteSpace(replace_by_space=False)
                                # jiwer.SentencesToListOfWords(word_delimiter=" ")
                            ]) 


    # def get_asr_result(self, content):
    #     """
    #     Note that transcription is limited to a 60 seconds audio file.
    #     Use a GCS file for audio longer than 1 minute.
    #     """
    #     audio = speech.RecognitionAudio(content=content)
    #     operation = self.sr_client.long_running_recognize(config=self.sr_config, audio=audio)
    #     response = operation.result(timeout=90)
    #     text = []

    #     # Each result is for a consecutive portion of the audio. Iterate through
    #     # them to get the transcripts for the entire audio file.
    #     for result in response.results:
    #         # The first alternative is the most likely one for this portion.
    #         text.append(result.alternatives[0].transcript)

    #     return " ".join(text)


    def get_wer(self, txt, pred_txt):

        return jiwer.wer(txt, pred_txt, truth_transform=self.wer_transform, hypothesis_transform=self.wer_transform)


    def get_vde(self, f0s, pred_f0s):
        Nerr = 0
        for f0, pred_f0 in zip(f0, pred_f0s):
            if f0 > 1e-6 and pred_f0 <= 1e-6:
                Nerr += 1
            elif f0 <= 1e-6 and pred_f0 > 1e-6:
                Nerr += 1
        
        return Nerr / len(f0s)


    def get_gpe(self, f0s, pred_f0s, delta=0.2):
        Nerr = 0
        Nvv = 0
        for f0, pred_f0 in zip(f0, pred_f0s):
            if f0 > 1e-6 and pred_f0 > 1e-6:
                Nvv += 1
                if abs(pred_f0/f0 - 1) > delta:
                    Nerr += 1
        
        return Nerr / Nvv


    def get_ffe(self, f0s, pred_f0s, delta=0.2):
        Nerr = 0
        for f0, pred_f0 in zip(f0, pred_f0s):
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

    def evaluate_rhythm(self, fname_dir):
        speaking_rate = get_speaking_rate(fname_dir) # key: file name; value: speaking rate(num_syls / voiced_duration)
        fname_list = set([key[:-6] for key in speaking_rate.keys()])
        src_cnt, tgt_cnt = 0, 0
        for fname in sorted(fname_list):
            src, tgt, cvt = speaking_rate[fname+'_s.wav'], speaking_rate[fname+'_t.wav'], speaking_rate[fname+'_c.wav']
            if abs(tgt-cvt)<=abs(src-cvt):
                tgt_cnt += 1
            else:
                src_cnt += 1
    
        return {'src_cnt': src_cnt, \
                'tgt_cnt': tgt_cnt}


    # def evaluate_content(self, cvt_wav_dir, src_txt_dir, tgt_txt_dir):
    #     src_wer, tgt_wer = 0, 0
    #     for fname in os.listdir(cvt_wav_dir):
    #         cvt_name = os.path.join(cvt_wav_dir, fname)
    #         cvt_content = self._get_content_from_file(cvt_name)
    #         cvt_txt = self.get_asr_result(cvt_content)

    #         with open(os.path.join(src_txt_dir, os.path.splitext(fname)[0]+'.txt'), 'r') as f:
    #             src_txt = f.read().strip()
    #         src_wer += self.get_wer(src_txt, cvt_txt)

    #         with open(os.path.join(tgt_txt_dir, os.path.splitext(fname)[0]+'.txt'), 'r') as f:
    #             tgt_txt = f.read().strip()
    #         tgt_wer += self.get_wer(tgt_txt, cvt_txt)
    #     src_wer /= len(os.listdir(cvt_wav_dir))
    #     tgt_wer /= len(os.listdir(cvt_wav_dir))

    #     return src_wer, tgt_wer

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
            src_f0 = extract_f0(src_wav, fs, lo[src_gen], hi[src_gen])

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[src_gen], hi[src_gen])

            src_vde += self.get_vde(src_f0, cvt_f0)
            src_gpe += self.get_gpe(src_f0, cvt_f0)
            src_ffe += self.get_ffe(src_f0, cvt_f0)

        src_vde /= len(fname_list)
        src_gpe /= len(fname_list)
        src_ffe /= len(fname_list)

        tgt_vde = 0
        tgt_gpe = 0
        tgt_ffe = 0
        
        for fname in fname_list:

            tgt_gen = spk2gen[fname.split('_')[2]]

            tgt_name = os.path.join(fname_dir, fname+'_t.wav')
            tgt_wav, fs = read(tgt_name)
            tgt_f0 = extract_f0(tgt_wav, fs, lo[tgt_gen], hi[tgt_gen])

            cvt_name = os.path.join(fname_dir, fname+'_c.wav')
            cvt_wav, fs = read(cvt_name)
            cvt_f0 = extract_f0(cvt_wav, fs, lo[tgt_gen], hi[tgt_gen])

            tgt_vde += self.get_vde(tgt_f0, cvt_f0)
            tgt_gpe += self.get_gpe(tgt_f0, cvt_f0)
            tgt_ffe += self.get_ffe(tgt_f0, cvt_f0)

        tgt_vde /= len(fname_list)
        tgt_gpe /= len(fname_list)
        tgt_ffe /= len(fname_list)

        return {'src_vde': src_vde, \
                'src_gpe': src_gpe, \
                'src_ffe': src_ffe, \
                'tgt_vde': tgt_vde, \
                'tgt_gpe': tgt_gpe, \
                'tgt_ffe': tgt_ffe}


if __name__ == '__main__':

    e = Evaluator()
    test_data_by_ctype = pickle.load(open('eval/assets/test_data_by_ctype.pkl', 'rb'))
    model_type_list = [
        # 'spsp1',
        'spsp2',
    ]

    model_name_list = {
        # 'R_8_1',
        # 'R_1_1',
        # 'R_8_32',
        'R_1_32',
    }

    ctype_list = [
        # 'F',
        'R',
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
                for (src_name, src_id), (tgt_name, tgt_id) in pairs[:5]:
                    fname_list.append(src_name.split('/')[-1]+'_'+tgt_name.split('/')[-1])
                fname_dir = os.path.join(result_dir, model_type, model_name, ctype)

                if ctype == 'F':
                    metrics[model_type][model_name][ctype] = e.evaluate_pitch(fname_dir, fname_list)
                elif ctype == 'R':
                    metrics[model_type][model_name][ctype] = e.evaluate_rhythm(fname_dir)

    dict2json(metrics, 'metrics.json')