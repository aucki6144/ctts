# CTTS: Controllable Text-To-Speech Based on Fastspeech2

## Status

> **NEW PROGRESS WILL BE UPDATED IN THIS REPOSITORY**

### Current progress:

* GUI code structure updated

* Standard grading pipeline (Due to CUDA version confliction, maybe moved to a new repo later)

* Preprocess and test on a new dataset combining LJSpeech and ESD(en part).

* Condition Layer Norm added for emotion control.

* ~~Added support for ESD(Mandarin)~~ Temporarly removed for adjustment on the model itself.

* ESD(Mandarin) MFA

* Preprocess of ESD, emotion label included.

**Todo:**

* Train model for mandrain tts

* Reconstruct the Naive model (without conditional layer norm) and evaluate it on the new standard grader

* Working on this project: [Speech-Backbones/Grad-TTS at main · huawei-noah/Speech-Backbones · GitHub](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS)

* Working on this project: https://github.com/sh-lee-prml/hierspeechpp

## Quickstart

### GUI
Start gradio gui with
```commandline
python .\gui.py
```

### Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

### Inference

Arrange config files as the following structure:
```
  .
  ├── config
  │   ├── DATASET_NAME
  └── └── └── model.yaml
          └── preprocess.yaml
          └── train.yaml
```
Take ESD_en dataset as an example. For English single-speaker TTS, run
```commandline
python .\synthesize.py -t "YOUR_CONTENT" -m ESD_en
```
There are optional parameters, you can check out the details by using "help" or read the code in "synthesis.py"
```commandline
python .\synthesize.py --help
```
Here lists some common used parameters:

``-s`` or ``--speaker_id``: specify the emotion id in multi emotion datasets.

``-e`` or ``--emotion_id``: specify the speaker id in multi speaker datasets.

``-r`` or ``--restore_step``: load the model of a particular checkpoint.

The generated utterances will be put in ``output/result/``.
