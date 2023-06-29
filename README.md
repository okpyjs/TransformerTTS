# Intergrating Transformer TTS to any project

## Pre requirements
- Python >= 3.9
- Clone git repository in your project root directory
```bash
git clone https://github.com/okpyjs/TransformerTTS.git
```

## Installation
- Install espeak as phonemizer backend
```bash
sudo apt-get install 
```
- Install virtual environment
```bash
virtualenv venv
```
```bash
source venv/bin/activate
```
- Install dependencies
```bash
pip install -r TransformerTTS/requirements.txt
```

## Usage
- Add following code to your script
```python
import sys
sys.path.append('TransformerTTS/')

from data.audio import Audio
from model.factory import tts_ljspeech
model = tts_ljspeech()
audio = Audio.from_config(model.config)
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
```

## Use pre-trained model from command line with
```bash
cd TransformerTTS
```
```bash
python predict_tts.py -t "Please, say something."
```
