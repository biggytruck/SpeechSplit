# demo conversion
import torch
import pickle
import numpy as np
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
import torch
import soundfile
import pickle
import os
from synthesis import build_model
from synthesis import wavegen


device = 'cuda:0'
G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/660000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

P = F0_Converter(hparams).eval().to(device)
p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])


metadata = pickle.load(open('assets/demo.pkl', "rb"))


sbmt_i = metadata[0]
emb_org = torch.from_numpy(sbmt_i[1]).to(device)
x_org, f0_org, len_org, uid_org = sbmt_i[2]        
uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis,:,:], 192)
uttr_org_pad = torch.from_numpy(uttr_org_pad).to(device)
f0_org_pad = np.pad(f0_org, (0, 192-len_org), 'constant', constant_values=(0, 0))
f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)
uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

sbmt_j = metadata[1]
emb_trg = torch.from_numpy(sbmt_j[1]).to(device)
x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]        
uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis,:,:], 192)
uttr_trg_pad = torch.from_numpy(uttr_trg_pad).to(device)
f0_trg_pad = np.pad(f0_trg, (0, 192-len_trg), 'constant', constant_values=(0, 0))
f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
f0_trg_onehot = torch.from_numpy(f0_trg_onehot).to(device)

with torch.no_grad():
    f0_pred = P(uttr_org_pad, f0_trg_onehot)[0]
    f0_pred_quantized = f0_pred.argmax(dim=-1).squeeze(0)
    f0_con_onehot = torch.zeros((1, 192, 257), device=device)
    f0_con_onehot[0, torch.arange(192), f0_pred_quantized] = 1
uttr_f0_trg = torch.cat((uttr_org_pad, f0_con_onehot), dim=-1)    


conditions = ['R', 'F', 'U', 'RF', 'RU', 'FU', 'RFU']
spect_vc = []
with torch.no_grad():
    for condition in conditions:
        if condition == 'R':
            x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_org)
        if condition == 'F':
            x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_org)
        if condition == 'U':
            x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_trg)
        if condition == 'RF':
            x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_org)
        if condition == 'RU':
            x_identic_val = G(uttr_f0_org, uttr_trg_pad, emb_trg)
        if condition == 'FU':
            x_identic_val = G(uttr_f0_trg, uttr_org_pad, emb_trg)
        if condition == 'RFU':
            x_identic_val = G(uttr_f0_trg, uttr_trg_pad, emb_trg)
            
        if 'R' in condition:
            uttr_trg = x_identic_val[0, :len_trg, :].cpu().numpy()
        else:
            uttr_trg = x_identic_val[0, :len_org, :].cpu().numpy()
                
        spect_vc.append( ('{}_{}_{}_{}'.format(sbmt_i[0], sbmt_j[0], uid_org, condition), uttr_trg ) )       

# spectrogram to waveform
if not os.path.exists('results'):
    os.makedirs('results')

model = build_model().to(device)
checkpoint = torch.load("assets/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

for spect in spect_vc:
    name = spect[0]
    c = spect[1]
    print(name)
    waveform = wavegen(model, c=c)   
    soundfile.write('results/'+name+'.wav', waveform, 16000)