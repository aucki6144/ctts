import argparse
import os
import os.path
import json
import time

import gradio as gr
import numpy as np
import torch
import yaml

import synthesize
from utils.model import get_vocoder, get_model_from_path
from utils.tools import to_device, synth_samples

os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoints_dir = 'gui/checkpoints'
checkpoints_dir = 'gui/ESD_en'


def synthesize_speech(text, speaker, emotion, pitch, energy, duration, checkpoint, dataset, strict):
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
    dataset_name = dataset

    # Parse restore step
    restore_step = checkpoint.split(".")[0]
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
    model = get_model_from_path(configs, device, checkpoint_path=os.path.join(checkpoints_dir, checkpoint))

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])
    emotions = np.array([emotion_id])
    if preprocess_config["preprocessing"]["text"]["language"] == "en":
        texts = np.array([synthesize.preprocess_english(text, preprocess_config, strict=strict)])
    elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
        texts = np.array([synthesize.preprocess_mandarin(text, preprocess_config)])
    text_lens = np.array([len(texts[0])])
    batches = [(ids, raw_texts, speakers, emotions, texts, text_lens, max(text_lens))]

    path_name = ""

    start_time = time.time()

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

    end_time = time.time()

    inference_time = end_time - start_time

    print("Inference time: {}".format(inference_time))

    pic_path_name = "{}.png".format(path_name)
    wav_path_name = "{}.wav".format(path_name)

    return pic_path_name, wav_path_name, inference_time


def gr_interface(text, speaker, emotion, pitch, energy, duration, dataset, checkpoint, strict):
    pic_path_name, wav_path_name, inference_time = (
        synthesize_speech(text, speaker, emotion, pitch, energy, duration, checkpoint, dataset, strict))
    if os.path.exists(pic_path_name) and os.path.exists(wav_path_name):
        return pic_path_name, wav_path_name, inference_time
    else:
        raise ValueError("Cannot find path of png/wav file")


customize_css = """
body, button, input, select, textarea {
    font-size: 16px; /* 增加字号 */
    font-weight: bold; /* 增加字重 */
}
"""

if __name__ == "__main__":
    datasets_file_path = './gui/datasets.json'

    with open(datasets_file_path, 'r') as file:
        datasets_list = json.load(file)

    with gr.Blocks(theme=gr.themes.Soft(), css=customize_css) as demo:
        gr.Markdown(
            "<center><h1>Controllable Text-To-Speech: FastSpeech2 with speaker and emotion embedding</h1></center>")

        with gr.Row():
            with gr.Column():
                gr.Markdown("<h4>Content Configuration</h4>")
                text_input = gr.Textbox(lines=3, placeholder="Type in your content...", label="Text")
                speaker_control = gr.Textbox(lines=1, placeholder="Select speaker id", label="speaker", value="1")
                pitch_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Pitch Control",
                                           value="1.0")
                energy_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Energy Control",
                                            value="1.0")
                duration_control = gr.Textbox(lines=1, placeholder="Float number accepted", label="Duration Control",
                                              value="1.0")
                emotion_radio = gr.Radio(choices=["Neutral", "Angry", "Happy", "Sad", "Surprise"],
                                         label="Emotion", value="Neutral")
                submit_button = gr.Button("Generate")

            with gr.Column():
                gr.Markdown("<h4>Model Configuration</h4>")
                checkpoint_files = os.listdir(checkpoints_dir)
                checkpoint_file = gr.Dropdown(choices=checkpoint_files, label="Checkpoint file")
                # batch_inference = gr.Checkbox(label="Batch Inference")
                strict_mode = gr.Checkbox(label="Strict Mode")
                # denoise_mode = gr.Checkbox(label="Denoise Mode")
                dataset_select = gr.Dropdown(choices=datasets_list, label="Database")

                mel_spectrogram = gr.Image(type="filepath", label="Mel Spectrogram")
                output_audio = gr.Audio(type="filepath", label="Model Output Audio")
                output_time = gr.Text(label="Inference Time")

        submit_button.click(
            gr_interface,
            inputs=[text_input,
                    speaker_control,
                    emotion_radio,
                    pitch_control,
                    energy_control,
                    duration_control,
                    dataset_select,
                    checkpoint_file,
                    strict_mode],
            outputs=[mel_spectrogram, output_audio, output_time]
        )

    demo.launch(server_port=14523)
