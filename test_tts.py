import os
import re
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "libs")))

from CoquiTTS.api import TTS
import sounddevice as sd
import yaml


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["coqui_tts_model_path"]
config_path = config["coqui_tts_model_config_path"]
vocoder_path = config["coqui_tts_vocoder_model_path"]
vocoder_config_path = config["coqui_tts_vocoder_model_config_path"]
tts_gpu = config["coqui_tts_gpu"]

class CoquiTTS:
    def __init__(self):
        self.synthesizer = TTS(model_path=model_path, 
                                config_path=config_path, 
                                vocoder_path=vocoder_path, 
                                vocoder_config_path=vocoder_config_path,
                                gpu=tts_gpu).synthesizer

    def say(self, text):
        if text == "":
            return
        
        chunks = self._split_text(text)
        # chunks = [text]
        print(chunks)
        for chunk in chunks:
            wav = self.synthesizer.tts(chunk)
            sd.stop()
            sd.play(wav, samplerate=22050)
            sd.wait()

    def _split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS processing"""
        sentences = re.split(r'(?<=[.!?]) +|\n+', text)
        chunks, current = [], ''
        for sent in sentences:
            sent = sent.strip()
            if len(current) + len(sent) + 1 <= 200:
                current += sent + ' '
            else:
                chunks.append(current.strip())
                current = sent + ' '
        if current:
            chunks.append(current.strip())
        return chunks 

tts = CoquiTTS()
tts.say("ten percent!")
tts.say("ten percent and twenty percent is thirty percent.")


