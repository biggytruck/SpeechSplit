import os

from soundfile import read
import jiwer
from google.cloud import speech
from utils import extract_f0
import numpy as np

class Evaluator(object):

    def __init__(self):

        """Initialize GCP speech recognizer"""
        self.sr_client = speech.SpeechClient()
        self.sr_config = speech.RecognitionConfig(
                            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
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

    def evaluate(self, src_dir, tgt_dir, metrics={}, gen=None):
        for key in metrics.keys():

            if key == 'content':
                wer = 0
                for fname in os.listdir(src_dir):
                    src_name = os.path.join(src_dir, fname)
                    src_content = self._get_content_from_file(src_name)
                    src_text = self.get_asr_result(src_content)

                    tgt_name = os.path.join(tgt_dir, fname)
                    tgt_content = self._get_content_from_file(tgt_name)
                    tgt_text = self.get_asr_result(tgt_content)
            
                    wer += self.get_wer(src_text, tgt_text)
                wer /= len(os.listdir(src_dir))
                metrics['content'] = {'wer': wer}

            if key == 'pitch':
                if gen == 'M':
                    lo, hi = 50, 250
                elif gen == 'F':
                    lo, hi = 100, 600
                else:
                    continue

                vde = 0
                gpe = 0
                ffe = 0
                for fname in os.listdir(src_dir):
                    src_name = os.path.join(src_dir, fname)
                    src_wav, fs = read(src_name)
                    src_f0 = extract_f0(src_wav, fs, lo, hi)

                    tgt_name = os.path.join(tgt_dir, fname)
                    tgt_wav, fs = read(tgt_name)
                    tgt_f0 = extract_f0(tgt_wav, fs, lo, hi)

                    vde += self.get_vde(src_f0, tgt_f0)
                    gpe += self.get_gpe(src_f0, tgt_f0)
                    ffe += self.get_ffe(src_f0, tgt_f0)
                vde /= len(os.listdir(src_dir))
                gpe /= len(os.listdir(src_dir))
                ffe /= len(os.listdir(src_dir))
                metrics['pitch'] = {'vde': vde, 'gpe': gpe, 'ffe': ffe}


if __name__ == '__main__':
    gen = 'M'
    e = Evaluator()
    wav_dir = './assets/wavs/p225/'
    txt_dir = './assets/txt/p225/'
    wer = []
    for fname in os.listdir(wav_dir):
        wav_file = os.path.join(wav_dir, fname)
        txt_file = os.path.join(txt_dir, os.path.splitext(fname)[0][:8]+'.txt')
        with open(txt_file, "r") as f:
            src_txt = f.read()
        tgt_content = e._get_content_from_file(wav_file)
        tgt_txt = e.get_asr_result(tgt_content)
        wer.append(e.get_wer(src_txt, tgt_txt))
        print(src_txt)
        print(tgt_txt)
        print(wer[-1])

    # {'max': 0.9, 'min': 0.0, 'mean': 0.12630339567164464, 'std': 0.1540352880369122, 'median': 0.08}
    print({'max': max(wer), 'min': min(wer), 'mean': np.mean(wer), 'std': np.std(wer), 'median': np.median(wer)})