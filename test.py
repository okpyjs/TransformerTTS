# Set up the paths
from pathlib import Path
MelGAN_path = 'melgan/'
TTS_path = 'TransformerTTS/'

import sys
sys.path.append(TTS_path)

# Load pretrained model
from model.factory import tts_ljspeech
from data.audio import Audio

model, config = tts_ljspeech()
audio = Audio(config)

# Synthesize text
sentence = 'Scientists at the CERN laboratory, say they have discovered a new particle.'
out_normal = model.predict(sentence)

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out_normal['mel'].numpy().T)

import IPython.display as ipd

ipd.display(ipd.Audio(wav, rate=config['sampling_rate']))

# 20% faster
sentence = 'Scientists at the CERN laboratory, say they have discovered a new particle.'
out = model.predict(sentence, speed_regulator=1.20)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
ipd.display(ipd.Audio(wav, rate=config['sampling_rate']))

# 10% slower
sentence = 'Scientists at the CERN laboratory, say they have discovered a new particle.'
out = model.predict(sentence, speed_regulator=.9)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
ipd.display(ipd.Audio(wav, rate=config['sampling_rate']))

# Do some sys cleaning
sys.path.remove(TTS_path)
sys.modules.pop('model')

sys.path.append(MelGAN_path)
import torch
import numpy as np

vocoder = torch.hub.load('seungwonpark/melgan', 'melgan')
vocoder.eval()

mel = torch.tensor(out_normal['mel'].numpy().T[np.newaxis,:,:])

if torch.cuda.is_available():
    vocoder = vocoder.cuda()
    mel = mel.cuda()

with torch.no_grad():
    audio = vocoder.inference(mel)

# Display audio
ipd.display(ipd.Audio(audio.cpu().numpy(), rate=22050))
