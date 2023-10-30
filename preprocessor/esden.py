import os
import re

import librosa
import numpy as np
from tqdm import tqdm
from pypinyin import pinyin, Style
from scipy.io import wavfile


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    emo_dir = config["path"]["emotion_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    speaker_list = os.listdir(in_dir)
    for speaker in speaker_list:
        print("Processing speaker: " + speaker)
        filename = speaker + ".txt"
        with open(os.path.join(in_dir, speaker, filename), encoding="utf-8") as f:
            for line in tqdm(f):
                wav_name, text, emo_name = line.strip("\n").split("\t")
                wav_file_name = wav_name + ".wav"
                wav_path = os.path.join(in_dir, speaker, "wav", wav_file_name)
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    os.makedirs(os.path.join(emo_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, wav_file_name),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}.lab".format(wav_name)),
                        "w",
                        encoding="utf-8"
                    ) as f1:
                        f1.write(text)
                    with open(
                        os.path.join(emo_dir, speaker, "{}.txt".format(wav_name)),
                        "w",
                        encoding="utf-8"
                    ) as f2:
                        f2.write(emo_name)

