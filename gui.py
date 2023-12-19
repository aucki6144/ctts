import argparse
import os
import os.path
import re

import gradio as gr
import numpy as np
import torch
import yaml
from gradio.components import Textbox, Dropdown, Radio, Audio, Image

import synthesize
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize_speech(text, speaker, emotion, pitch_control, energy_control, duration_control, model_name, step):

    # Parse speaker
    match = re.search(r'Speaker(\d+)', speaker)
    if match:
        speaker_id = int(match.group(1)) - 1
    else:
        raise ValueError("Invalid format of speaker")

    # Parse emotion
    emotions = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}
    emotion_id = emotions.get(emotion)

    # Parse restore step
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=step)
    args = parser.parse_args()

    # Parse pitch, energy, duration
    pitch_control = float(pitch_control)
    energy_control = float(energy_control)
    duration_control = float(duration_control)

    # Get path by model name
    preprocess_config_path = os.path.join("./config/{}/preprocess.yaml".format(model_name))
    model_config_path = os.path.join("./config/{}/model.yaml".format(model_name))
    train_config_path = os.path.join("./config/{}/train.yaml".format(model_name))

    # Read Config
    preprocess_config = yaml.load(
        open(preprocess_config_path, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(
        open(model_config_path, "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open(train_config_path, "r"), Loader=yaml.FullLoader
    )
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])
    emotions = np.array([emotion_id])
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([synthesize.preprocess_english(text, preprocess_config)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([synthesize.preprocess_mandarin(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, emotions, texts, text_lens, max(text_lens))]

    path_name = ""

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            path_name = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

    pic_path_name = "{}.png".format(path_name)
    wav_path_name = "{}.wav".format(path_name)

    return pic_path_name, wav_path_name


def gr_interface(text, speaker, emotion, pitch_control, energy_control, duration_control, model_name, step):
    pic_path_name, wav_path_name = (
        synthesize_speech(text, speaker, emotion, pitch_control, energy_control, duration_control, model_name, step))
    if os.path.exists(pic_path_name) and os.path.exists(wav_path_name):
        return pic_path_name, wav_path_name
    else:
        print(text)
        print(speaker)
        raise ValueError("Cannot find path of png/wav file")


if __name__ == "__main__":
    iface = gr.Interface(
        fn=gr_interface,
        inputs=[
            Textbox(lines=3, placeholder="Type in your content...", label="Text"),
            Dropdown(
                choices=["Speaker1", "Speaker2", "Speaker3", "Speaker4", "Speaker5", "Speaker6", "Speaker7", "Speaker8",
                         "Speaker9",
                         "Speaker10", "Speaker11"], label="Speaker"),
            Radio(choices=["Neutral", "Angry", "Happy", "Sad", "Surprise"], label="Emotion"),
            Textbox(lines=1, placeholder="Float number accepted", label="Pitch Control", value="1.0"),
            Textbox(lines=1, placeholder="Float number accepted", label="Energy Control", value="1.0"),
            Textbox(lines=1, placeholder="Float number accepted", label="Duration Control", value="1.0"),
            Dropdown(
                choices=["LJSpeech", "ESD_en", "Cross_LJESD"], label="Database"),
            Textbox(lines=1, placeholder="Int number accepted", label="Checkpoint step", value="5000"),
        ],
        outputs=[
            Image(type="filepath", label="Mel Spectrogram"),
            Audio(type="filepath", label="Model Output Audio"),
        ],
        title="FastSpeech2 with speaker and emotion embedding",
        description="This is a PyTorch implementation of Microsoft's text-to-speech system FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. This project is based on xcmyz's implementation of FastSpeech. Code based on https://github.com/ming024/FastSpeech2. More details can be found in the origin repo."
    )

    iface.launch()
