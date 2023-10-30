# FastSpeech 2 With Emotion Embedding- PyTorch Implementation

## Status
**Current progress:**
- Added support for ESD(Mandarin)
- ESD(Mandarin) MFA
- Preprocess of ESD, emotion label included.
- GUI with gradio

**Todo:**
- rewrite the following parts: batch inference and controllability.
- check if all supported datasets work well

## Introduction of this repository

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1). 
This project is based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Code based on https://github.com/ming024/FastSpeech2. More details can be found in the origin repo.

![](./img/model.png)

# Quickstart

## Gradio Page
Start gui with
```commandline
python .\gui.py
```


## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

If you have read the origin repository (by @ming024), you can see the inference/training command is now simplified. You can now simply use the name of the model like "ESD_en" to refer to all its config files.
To make that work, you must regulate config files in a fixed structure, which should look like this:

```
  .
  ├── config
  │   ├── MODEL_NAME
  └── └── └── model.yaml
          └── preprocess.yaml
          └── train.yaml
``` 
We take ESD_en dataset as an example. For English single-speaker TTS, run
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

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.
- [ESD](https://hltsingapore.github.io/ESD/): ESD is an Emotional Speech Database for voice conversion research. The ESD database consists of 350 parallel utterances spoken by 10 native English and 10 native Chinese speakers and covers 5 emotion categories (neutral, happy, angry, sad and surprise). More than 29 hours of speech data were recorded in a controlled acoustic environment. The database is suitable for multi-speaker and cross-lingual emotional voice conversion studies.


We take AISHELL3 as an example hereafter.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/AISHELL3/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## More details about MFA

~~Alternately, you can align the corpus by yourself. 
Download the official MFA package and run~~
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
~~or~~
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

~~to align the corpus and then run the preprocessing script.~~
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Updated by CTTS: the commands above provided by origin repo cannot work in most cases, especially when the coder is not familiar with MFA tools, you can get more details in the Issues of the origin repo.

The following is the baselien of aligning your raw data with MFA, Attention: dictionarys and acoustic models provided my MFA can work well in languages like ENG (As far as I am concerned), but official dictionaries fail in aligning mandarins, especially in this 
repository, because the preprocess of this repository is based on pinyin in Mandarin.  Therefore, used customized dictionary and dictionary to make sure it works well with Mandarin. Sources can be accessible in the Google Drive URL below:

**Acoustic Model:** [Download In Google Drive](https://drive.google.com/file/d/1Kr6iefOAtam2w9KcHWAED3v5MsAPVRMn/view?usp=sharing)

**Dictionary:** [Download in Google Drive](https://drive.google.com/file/d/1yrRNkZM4m4RG-IjoJQO1pxXWvNqzec9Y/view?usp=sharing)

> ### How to align Mandarin raw data (e.g. AISHELL3, ESD) with MFA?
> 1. Install MFA tools, use the command "mfa --help" to check if the environment is set correctly.
> 2. Install Mandarin pretrained acoustic model, the models can be found at: https://mfa-models.readthedocs.io/en/latest/acoustic/index.html
> 3. Get data prepared for alignment
> 4. Make sure a lexicon file is ready, like the "pinyin-lexicon-r.txt" in this project.
> 5. Use the command **"mfa align .\raw_data_path .\lexicon_path ACOUSTIC_MODEL .\target_path"** to start alignment.

> *In this project (ESD mandarin), the speaker "0005" is removed for unknown errors it would cause in MFA alignment"*


**_Attention: The following processes haven't been modified for ESD support yet!_**
## Training

Train your model with
```
python3 train.py -m MODEL_NAME
```

# Implementation Issues

- Following [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Post-Net after the decoder, which is not used in the original FastSpeech 2.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```
