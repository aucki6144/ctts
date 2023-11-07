import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, esdzh, esden


def main(config):
    if "LJSpeech" in config["dataset"]:
        print("Prepare alignment for LJSpeech")
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        print("Prepare alignment for AISHELL3")
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        print("Prepare alignment for LibriTTS")
        libritts.prepare_align(config)
    if "ESD_zh" in config["dataset"]:  # Add support for ESD dataset
        print("Prepare alignment for ESD")
        esdzh.prepare_align(config)
    if "ESD_en" in config["dataset"]:
        print("Prepare alignment for ESD")
        esden.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
