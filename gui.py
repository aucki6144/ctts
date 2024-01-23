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
from utils.model import get_model, get_vocoder, get_model_from_path
from utils.tools import to_device, synth_samples

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synthesize_speech(text, speaker, emotion, pitch, energy, duration, checkpoint):
    """
    @param text: text to generate
    @param speaker: the id of a speaker
    @param emotion: emotion name
    @param pitch: pitch control parameter
    @param energy: energy control parameter
    @param duration: duration control parameter
    @param checkpoint: file name of the checkpoint in ./gui/checkpoints
    @return: file name of the generated mel-spectrogram picture and the generated .wav file
    """

    # Parse emotion id from emotion name
    emotions = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}
    emotion_id = emotions.get(emotion)

    # Parse speaker id
    speaker_id = int(speaker)

    # Parse dataset name
    dataset_name = checkpoint.split("_")[0]

    # Parse restore step
    restore_step = checkpoint.split("_")[1].split(".")[0]
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=restore_step)
    args = parser.parse_args()

    # Parse pitch, energy, duration
    pitch = float(pitch)
    energy = float(energy)
    duration = float(duration)

    # Get path by model name
    preprocess_config_path = os.path.join("./config/{}/preprocess.yaml".format(dataset_name))
    model_config_path = os.path.join("./config/{}/model.yaml".format(dataset_name))
    train_config_path = os.path.join("./config/{}/train.yaml".format(dataset_name))

    # Read config files
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
    model = get_model_from_path(configs, device, checkpoint_path="./gui/checkpoints/{}".format(checkpoint))

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
    batches = [(ids, raw_texts, speakers, emotions, texts, text_lens, max(text_lens))]

    path_name = ""

    for batch in batches:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch,
                e_control=energy,
                d_control=duration
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


def gr_interface(text, speaker, emotion, pitch, energy, duration, checkpoint):
    pic_path_name, wav_path_name = (
        synthesize_speech(text, speaker, emotion, pitch, energy, duration, checkpoint))
    if os.path.exists(pic_path_name) and os.path.exists(wav_path_name):
        return pic_path_name, wav_path_name
    else:
        raise ValueError("Cannot find path of png/wav file")


theme_css = """
<style>
    :root {
    }
</style>
"""

if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown(theme_css)
        gr.Markdown("<center><h1>FastSpeech2 with speaker and emotion embedding</h1></center>")
        gr.Markdown("This is a PyTorch implementation of Microsoft's text-to-speech system FastSpeech 2: "
                    "Fast and High-Quality End-to-End Text to Speech. This project is based on xcmyz's implementation "
                    "of FastSpeech. Code based on https://github.com/ming024/FastSpeech2. More details can be found in "
                    "the origin repo.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Content Configuration")
                text_input = gr.Textbox(lines=3, placeholder="Type in your content...", label="Text")
                # speaker_dropdown = gr.Dropdown(
                #     choices=["Speaker1", "Speaker2", "Speaker3", "Speaker4", "Speaker5", "Speaker6", "Speaker7",
                #              "Speaker8",
                #              "Speaker9", "Speaker10", "Speaker11"], label="Speaker")
                speaker_control = gr.Textbox(lines=1, placeholder="Select speaker id", label="speaker", value="1")
                pitch_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Pitch Control",
                                           value="1.0")
                energy_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Energy Control",
                                            value="1.0")
                duration_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Duration Control",
                                              value="1.0")
                # database_dropdown = gr.Dropdown(choices=["LJSpeech", "ESD_en", "Cross_LJESD"], label="Database")
                emotion_radio = gr.Radio(choices=["Neutral", "Angry", "Happy", "Sad", "Surprise"], label="Emotion")
                submit_button = gr.Button("Generate")

            with gr.Column():
                gr.Markdown("Model Configuration")
                checkpoint_files = os.listdir('gui/checkpoints')
                checkpoint_file = gr.Dropdown(choices=checkpoint_files, label="Checkpoint file")
                batch_inference = gr.Checkbox(label="Batch Inference")
                strict_mode = gr.Checkbox(label="Strict Mode")
                denoise_mode = gr.Checkbox(label="Denoise Mode")

                mel_spectrogram = gr.Image(type="filepath", label="Mel Spectrogram")
                output_audio = gr.Audio(type="filepath", label="Model Output Audio")

        submit_button.click(
            gr_interface,
            inputs=[text_input,
                    speaker_control,
                    emotion_radio,
                    pitch_control,
                    energy_control,
                    duration_control,
                    checkpoint_file],
            outputs=[mel_spectrogram, output_audio]
        )

    demo.launch(server_port=14523)
