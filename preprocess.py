import argparse
import os
import yaml

from preprocessor.preprocessor import Preprocessor
from preprocessor.prepare_align import PrepareAlign


if __name__ == "__main__":
    config_root = "./config"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="ESD_en",
        help="Name of the models used. Supported: LJSpeech, AISHELL3, LibriTTS, ESD_zh, ESD_en(DEFAULT)"
    )

    input_args = parser.parse_args()
    model = input_args.model

    # Get path by models name
    preprocess_config_path = os.path.join("./config/", model, "preprocess.yaml")

    # Load config files
    input_preprocess_config = yaml.load(
        open(preprocess_config_path, "r"), Loader=yaml.FullLoader
    )

    print(f"Start preparing align for dataset {model}")
    PrepareAlign.prepare_align(input_preprocess_config)

    # Verify if TextGrid Exits
    text_grid_path = os.path.join(input_preprocess_config["path"]["preprocessed_path"], "TextGrid")

    if os.path.exists(text_grid_path) and os.path.isdir(text_grid_path):
        print(f"Start preprocessing dataset {model}")
        preprocessor = Preprocessor(input_preprocess_config)
        preprocessor.build_from_path()
    else:
        print(f"\033[32mCannot detect TextGrid in this path: {text_grid_path}\n"
              f"Please put TextGrid in this path and run preprocess.py again.")
