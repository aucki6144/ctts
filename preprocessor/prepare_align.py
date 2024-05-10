import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


class AISHELL3:

    @staticmethod
    def prepare_align(config):
        in_dir = config["path"]["corpus_path"]
        out_dir = config["path"]["raw_path"]
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        for dataset in ["train", "test"]:
            print("Processing {}ing set...".format(dataset))
            with open(os.path.join(in_dir, dataset, "content.txt"), encoding="utf-8") as f:
                for line in tqdm(f):
                    wav_name, text = line.strip("\n").split("\t")
                    speaker = wav_name[:7]
                    text = text.split(" ")[1::2]
                    wav_path = os.path.join(in_dir, dataset, "wav", speaker, wav_name)
                    if os.path.exists(wav_path):
                        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                        wav, _ = librosa.load(wav_path, sampling_rate)
                        wav = wav / max(abs(wav)) * max_wav_value
                        wavfile.write(
                            os.path.join(out_dir, speaker, wav_name),
                            sampling_rate,
                            wav.astype(np.int16),
                        )
                        with open(
                                os.path.join(out_dir, speaker, "{}.lab".format(wav_name[:11])),
                                "w",
                        ) as f1:
                            f1.write(" ".join(text))


class ESDEN:

    @staticmethod
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


class ESDZH:
    @staticmethod
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
                    wav_name, zh_text, emo_name = line.strip("\n").split("\t")
                    wav_file_name = wav_name + ".wav"
                    pinyin_str = to_pinyin(zh_text)
                    pinyin_str = re.sub(r'[^\w\s]', '', pinyin_str)
                    zh_text = re.sub(r'[^\w\s]', '', zh_text)
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
                                "w"
                        ) as f1:
                            f1.write(pinyin_str)
                        with open(
                                os.path.join(emo_dir, speaker, "{}.txt".format(wav_name)),
                                "w"
                        ) as f2:
                            f2.write(emo_name)

    @staticmethod
    def to_pinyin(chinese_str):
        pinyin_list = pinyin(chinese_str, style=Style.TONE3)
        return ' '.join([word[0] for word in pinyin_list])


class LJSpeech:

    @staticmethod
    def prepare_align(config):
        in_dir = config["path"]["corpus_path"]
        out_dir = config["path"]["raw_path"]
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        cleaners = config["preprocessing"]["text"]["text_cleaners"]
        speaker = "LJSpeech"
        with open(os.path.join(in_dir, "metadata.csv"), encoding="utf-8") as f:
            for line in tqdm(f):
                parts = line.strip().split("|")
                base_name = parts[0]
                text = parts[2]
                text = _clean_text(text, cleaners)

                wav_path = os.path.join(in_dir, "wavs", "{}.wav".format(base_name))
                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                            os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                            "w",
                    ) as f1:
                        f1.write(text)


class LibriTTS:
    @staticmethod
    def prepare_align(config):
        in_dir = config["path"]["corpus_path"]
        out_dir = config["path"]["raw_path"]
        sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
        cleaners = config["preprocessing"]["text"]["text_cleaners"]
        for speaker in tqdm(os.listdir(in_dir)):
            for chapter in os.listdir(os.path.join(in_dir, speaker)):
                for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                    if file_name[-4:] != ".wav":
                        continue
                    base_name = file_name[:-4]
                    text_path = os.path.join(
                        in_dir, speaker, chapter, "{}.normalized.txt".format(base_name)
                    )
                    wav_path = os.path.join(
                        in_dir, speaker, chapter, "{}.wav".format(base_name)
                    )
                    with open(text_path) as f:
                        text = f.readline().strip("\n")
                    text = _clean_text(text, cleaners)

                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sampling_rate)
                    wav = wav / max(abs(wav)) * max_wav_value
                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                            os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                            "w",
                    ) as f1:
                        f1.write(text)


class PrepareAlign:

    @staticmethod
    def prepare_align(config):
        """
        prepare align before preprocessing
        Supported datasets: LJSpeech, AISHELL3, LibriTTS, ESD_zh, ESD_en
        @rtype: object
        """
        if "LJSpeech" in config["dataset"]:
            print("Prepare alignment for LJSpeech")
            LJSpeech.prepare_align(config)

        if "AISHELL3" in config["dataset"]:
            print("Prepare alignment for AISHELL3")
            AISHELL3.prepare_align(config)

        if "LibriTTS" in config["dataset"]:
            print("Prepare alignment for LibriTTS")
            LibriTTS.prepare_align(config)

        if "ESD_zh" in config["dataset"]:
            print("Prepare alignment for ESD")
            ESDZH.prepare_align(config)

        if "ESD_en" in config["dataset"]:
            print("Prepare alignment for ESD")
            ESDEN.prepare_align(config)
