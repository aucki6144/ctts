import os
import os.path
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

checkpoints_dir = 'gui/checkpoints'


def tts(text, speaker, emotion, pitch, energy, duration, checkpoint, strict):
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
    dataset_name = "ESD_en"

    # Parse restore step
    # restore_step = checkpoint.split(".")[0]
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--restore_step", type=int, default=restore_step)
    # args = parser.parse_args()

    # Parse pitch, energy, duration
    pitch = float(pitch)
    energy = float(energy)
    duration = float(duration)

    # Get path by models name
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

    # Get models
    model = get_model_from_path(configs, device, checkpoint_path=os.path.join(checkpoints_dir, checkpoint))

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    ids = raw_texts = [text[:100]]
    speakers = np.array([speaker_id])
    emotions = np.array([emotion_id])
    texts = np.array([synthesize.preprocess_phoneme(text, preprocess_config, strict=strict)])
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


if __name__ == "__main__":
    checkpoint_files = os.listdir(checkpoints_dir)
    default_checkpoint = checkpoint_files[0]

    demo_play = gr.Interface(fn=tts,
                             theme=gr.themes.Soft(),
                             inputs=[
                                 gr.Textbox(max_lines=3, label="Input Text",
                                            value=""),
                                 gr.Dropdown(choices=[0], label="Speaker"),
                                 gr.Radio(choices=["Neutral", "Angry", "Happy", "Sad", "Surprise"],
                                          label="Emotion", value="Neutral"),
                                 gr.Slider(0, 2, 1.0, label="pitch"),
                                 gr.Slider(0, 2, 1.0, label="duration"),
                                 gr.Slider(0, 2, 1.0, label="energy"),
                                 gr.Dropdown(choices=checkpoint_files, label="Checkpoint file"),
                                 gr.Checkbox(label="Strict Mode")
                             ],
                             outputs=[
                                 gr.Image(type="filepath", label="Mel Spectrogram"),
                                 gr.Audio(type="filepath", label="Model Output Audio"),
                                 gr.Text(label="Inference_time")
                             ],
                             title='ctts',
                             description='''
                             
                             ''',
                             examples=[
                                 ["Tester", "0", "Neutral", 1.0, 1.0, 1.0, default_checkpoint, True]
                             ]
                             )
    demo_play.launch()
