from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from model.factory import tts_ljspeech
from data.audio import Audio
from model.models import ForwardTransformer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', '-p', dest='path', default=None, type=str)
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--single', '-s', dest='single', action='store_true')
    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()
    # load the appropriate model
    outdir = Path(args.outdir) if args.outdir is not None else Path('.')
    if args.path is not None:
        print(f'Loading model from {args.path}')
        model = ForwardTransformer.load_model(args.path)
        file_name = f"{fname}_{model.config['data_name']}_{model.config['git_hash']}_{model.config['step']}"
    else:
        model, conf = tts_ljspeech()
        file_name = f'{fname}_ljspeech_v1'
    
    outdir = outdir / 'outputs' / f'{file_name}'
    outdir.mkdir(exist_ok=True, parents=True)
    audio = Audio.from_config(model.config)
    print(f'Output wav under {outdir}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: "{phons}"')
            print(f'Tokens: "{tokens}"')
        out = model.predict(tokens, encode=False, phoneme_max_duration=None)
        mel = out['mel'].numpy().T
        wav = audio.reconstruct_waveform(mel)
        wavs.append(wav)
        if args.store_mel:
            np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
        if args.single:
            audio.save_wav(wav, (outdir / (file_name + f'_{i}')).with_suffix('.wav'))
    audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))
