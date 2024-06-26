import json
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
speaker_list_path = 'gui/speakers.json'
example_list_path = 'gui/examples.json'

article_file = """
<h4 style="text-align: left;">Parameter descriptions</h4>
<p><strong>Input Text:&nbsp;</strong>The text to be synthesised, avoid symbols like "!?..." in the end. Not recommend to be longer than 20 words, for those situations, divide it into multiple shorter sentences to synthesis.</p>
<p><strong>Speaker:&nbsp;</strong>Different speakers have different voice and speech style. Select expected style by trying examples.</p>
<p><strong>Emotion:&nbsp;</strong>Choose a emotion between Neutral, Angry, Happy, Surprise and Sad.</p>
<p><strong>Pitch:&nbsp;</strong>Float number between 0.00 and 2.00, controlling the overall pitch of the whole sentence.</p>
<p><strong>Duration:&nbsp;</strong>Float number between 0.00 and 2.00, controlling the speed of the speech.</p>
<p><strong>Energy:&nbsp;</strong>Float number betwwen 0.00 and 2.00, controlling the variation of the energy of the whole sentence.</p>
<p><strong>Checkpoint File: </strong>Model checkpoint to be loaded, ESD_LJSpeech_8000_eng is the most recommended.</p>
<p><strong>Strict Mode:&nbsp;</strong>Whether synthesis wrong spelled words.</p>
"""


def tts(text, speaker, emotion, pitch, duration, energy, checkpoint, strict):
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

    print(f" pitch:{pitch}, duration:{duration}, energy:{energy}")

    # Parse emotion id from emotion name
    emotions = {"Neutral": 0, "Angry": 1, "Happy": 2, "Sad": 3, "Surprise": 4}
    emotion_id = emotions.get(emotion)

    # Parse speaker id
    speaker_id_dict = load_speaker_dict()
    speaker_id = int(speaker_id_dict.get(speaker, 0))

    # Parse pitch, energy, duration
    pitch = float(pitch)
    energy = float(energy)
    duration = float(duration)

    # Get path by models name
    preprocess_config_path = os.path.join("./config/ESD_en/preprocess.yaml")
    model_config_path = os.path.join("./config/ESD_en/model.yaml")
    train_config_path = os.path.join("./config/ESD_en/train.yaml")

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


def load_speaker_dict():
    # Read speaker list from file
    with open(speaker_list_path, 'r') as file:
        speaker_json = json.load(file)
    raw_dict = {}
    for item in speaker_json['speaker_list']:
        raw_dict.update(item)
    return raw_dict


def load_example_list():
    with open(example_list_path, 'r') as file:
        example_json = json.load(file)
    example_list = []
    for item in example_json['example_list']:
        example_list.append(list(dict(item).values()))
    return example_list


if __name__ == "__main__":
    checkpoint_files = os.listdir(checkpoints_dir)
    default_checkpoint = checkpoint_files[0]

    speaker_dict = load_speaker_dict()

    demo_play = gr.Interface(fn=tts,
                             theme=gr.themes.Soft(),
                             inputs=[
                                 gr.Textbox(max_lines=3, label="Input Text",
                                            value=""),
                                 gr.Dropdown(choices=list(speaker_dict.keys()), label="Speaker"),
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
                             title='CTTS: Controllable Text To Speech',
                             description='''
                             <center><b>Controllable Text-To-Speech: FastSpeech2 with speaker and emotion embedding, 
                             bring FastSpeech2 to the next level of its controllability while remaining its 
                             train/inference efficiency.</b></center>
                             ''',
                             article=article_file,
                             examples=load_example_list(),
                             allow_flagging="never"
                             )
    demo_play.launch()
