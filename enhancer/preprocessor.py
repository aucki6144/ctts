import os
import librosa
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_path_clean = config["path"]["out_path_clean"]
        self.out_path_noisy = config["path"]["out_path_noisy"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        self.n_fft = config["preprocessing"]["stft"]["filter_length"]
        self.n_mels = config["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config["preprocessing"]["mel"]["mel_fmax"]
        os.makedirs(self.out_path_clean, exist_ok=True)
        os.makedirs(self.out_path_noisy, exist_ok=True)

    def build_from_path(self):
        print("Processing Data (Enhancement Model)...")
        out = list()

        for wav_file in tqdm(os.listdir(self.in_dir)):
            if ".wav" not in wav_file:
                continue
            basename = os.path.splitext(wav_file)[0]
            wav_path = os.path.join(self.in_dir, wav_file)
            mel_path_clean = os.path.join(self.out_path_clean, f"{basename}.npy")
            mel_path_noisy = os.path.join(self.out_path_noisy, f"{basename}.npy")

            # Process each file
            mel_spec, noisy_mel_spec = self.process_utterance(wav_path)
            np.save(mel_path_clean, mel_spec)
            np.save(mel_path_noisy, noisy_mel_spec)
            out.append(mel_path_clean)

    def process_utterance(self, wav_path):
        y, sr = librosa.load(wav_path, sr=self.sampling_rate)
        mel_spec = self.get_mel_spectrogram(y)

        # Add noise to the mel spectrogram
        noisy_mel_spec = self.add_noise_to_spectrogram(mel_spec)
        return mel_spec, noisy_mel_spec

    def get_mel_spectrogram(self, audio):
        S = librosa.feature.melspectrogram(audio, sr=self.sampling_rate, n_fft=self.n_fft,
                                           hop_length=self.hop_length, n_mels=self.n_mels,
                                           fmin=self.mel_fmin, fmax=self.mel_fmax)
        S_DB = librosa.power_to_db(S, ref=np.max)
        return S_DB

    @staticmethod
    def add_noise_to_spectrogram(self, spec, noise_level=0.005):
        noise = np.random.randn(*spec.shape) * noise_level * np.amax(spec)
        noisy_spec = spec + noise
        return noisy_spec
