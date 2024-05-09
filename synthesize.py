import argparse
import os.path
import re
import time
import numpy as np
import torch
import yaml

from g2p_en import G2p
from torch.utils.data import DataLoader
from string import punctuation
from dataset import TextDataset
from text import text_to_sequence
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_lexicon(lex_path):
    """
    Reads a pronunciation lexicon from a file and stores it in a dictionary.
    @param lex_path: The file path to the lexicon text file.
    @type lex_path: str
    @rtype: dict
    @return: A dictionary mapping each word (in lowercase) to its list of phonetic transcriptions.
    """
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_phoneme(text, preprocess_config, strict=False):
    """
    Processes a given text into phonemes using a specified lexicon and grapheme-to-phoneme (G2P) conversion.
    @param text: The input text to be converted into phonemes.
    @type text: str
    @param preprocess_config: Preprocess configuration
    @type preprocess_config: dict
    @param strict: Toggle if G2p is on
    @type strict: bool
    @rtype: ndarray
    @return: A numpy array containing the numerical representation of the phoneme sequence generated from the text.
    """

    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    print("Preprocess phoneme")

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif strict is False:
            phones += list(filter(lambda p: p != " ", g2p(w)))
        # phones.append("sp")
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, config, vocoder, batchs, control_values):
    """
    Executes the synthesis process for a batch of inputs using a provided models and vocoder.
    @param model: The text-to-speech synthesis models used for generating speech.
    @type model: PyTorch Module
    @param config: A tuple containing three configuration dictionaries (preprocess_config, model_config, train_config)
    @type config: tuple
    @param vocoder: The vocoder used to convert models output into audible signals.
    @type vocoder: PyTorch Module
    @param batchs: A list of batches where each batch contains data to be processed.
    @type batchs: list
    @param control_values: A tuple containing control values for pitch, energy, and duration which adjust the synthesis characteristics.
    @type control_values: tuple
    @rtype: None
    @return: Does not return anything but prints the total inference time upon completion.
    """

    preprocess_config, model_config, train_config = config
    pitch_control, energy_control, duration_control = control_values
    start_time = time.time()

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
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )

    end_time = time.time()

    print("Inference time: {}".format(end_time - start_time))


def main(input_args, input_configs):
    # Get models
    checkpoint_model = get_model(input_args, input_configs, device, train=False, strict_load=True)

    # Load vocoder
    input_vocoder = get_vocoder(input_model_config, device)

    # Preprocess texts
    if input_args.mode == "batch":
        # Get dataset
        dataset = TextDataset(input_args.source, input_preprocess_config)
        input_batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    elif input_args.mode == "single":
        ids = raw_texts = [input_args.text[:100]]
        speakers = np.array([input_args.speaker_id])
        emotions = np.array([input_args.emotion_id])
        texts = np.array([preprocess_phoneme(input_args.text, input_preprocess_config)])
        text_lens = np.array([len(texts[0])])
        input_batchs = [(ids, raw_texts, speakers, emotions, texts, text_lens, max(text_lens))]
    else:
        print("Unsupported synthesis mode!")
        return

    input_control_values = input_args.pitch_control, input_args.energy_control, input_args.duration_control

    synthesize(checkpoint_model, input_configs, input_vocoder, input_batchs, input_control_values)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        default="batch",
        help="Synthesize a whole dataset or a single sentence",
    )

    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        type=str,
        default="./output/base_checkpoint.pth.tar",
        help="The path of checkpoint file, for example ./output/base_checkpoint.pth.tar",
    )

    parser.add_argument(
        "--source",
        type=str,
        default="./output/source.txt",
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )

    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "-s",
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )

    parser.add_argument(
        "-e",
        "--emotion_id",
        type=int,
        default=0,
        help="emotion ID for multi-emotional synthesis, for single-sentence mode only",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ESD_en",
        help="Name of the models used"
    )

    parser.add_argument(
        "-pc",
        "--pitch_control",
        type=float,
        default=1.0,
        help="Control the pitch of the whole utterance"
    )

    parser.add_argument(
        "-ec",
        "--energy_control",
        type=float,
        default=1.0,
        help="Control the energy of the whole utterance"
    )

    parser.add_argument(
        "-dc",
        "--duration_control",
        type=float,
        default=1.0,
        help="Control the speed of the whole utterance"
    )

    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Get path by models name
    preprocess_config_path = os.path.join("./config/", args.model, "preprocess.yaml")
    model_config_path = os.path.join("./config/", args.model, "model.yaml")
    train_config_path = os.path.join("./config/", args.model, "train.yaml")

    # Read Config
    input_preprocess_config = yaml.load(
        open(preprocess_config_path, "r"), Loader=yaml.FullLoader
    )
    input_model_config = yaml.load(
        open(model_config_path, "r"), Loader=yaml.FullLoader
    )
    input_train_config = yaml.load(
        open(train_config_path, "r"), Loader=yaml.FullLoader
    )
    configs = (input_preprocess_config, input_model_config, input_train_config)

    main(args, configs)
