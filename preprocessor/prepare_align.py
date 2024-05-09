import aishell3
import esden
import esdzh
import libritts
import ljspeech


class PrepareAlign:

    @staticmethod
    def prepare_align(config):
        """
        prepare align before preprocessing
        Supported datasets: LJSpeech, AISHELL3, LibriTTS, ESD_zh, ESD_en
        @rtype: object
        """
        if "LJSpeech" in config["dataset"]:
            print("Prepare alignment for LJSpeech")
            ljspeech.prepare_align(config)

        if "AISHELL3" in config["dataset"]:
            print("Prepare alignment for AISHELL3")
            aishell3.prepare_align(config)

        if "LibriTTS" in config["dataset"]:
            print("Prepare alignment for LibriTTS")
            libritts.prepare_align(config)

        if "ESD_zh" in config["dataset"]:
            print("Prepare alignment for ESD")
            esdzh.prepare_align(config)

        if "ESD_en" in config["dataset"]:
            print("Prepare alignment for ESD")
            esden.prepare_align(config)
