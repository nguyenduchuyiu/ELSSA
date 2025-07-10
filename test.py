# create processing chime sound: signify end of listening, start of processing

import numpy as np
import scipy.io.wavfile
import os

def create_processing_chime(
    filename="assets/audio/processing_chime.wav",
    duration=0.7,
    sr=24000,
    freq1=880,
    freq2=1320,
    gap=0.08
):
    # "Ding-ding" chime: two short tones, second higher, with a small gap
    t1 = np.linspace(0, 0.18, int(sr * 0.18), False)
    t2 = np.linspace(0, 0.13, int(sr * 0.13), False)
    gap_silence = np.zeros(int(sr * gap))

    tone1 = 0.6 * np.cos(2 * np.pi * freq1 * t1) * np.exp(-6 * t1)
    tone2 = 0.5 * np.cos(2 * np.pi * freq2 * t2) * np.exp(-7 * t2)

    audio = np.concatenate([tone1, gap_silence, tone2])

    # Pad or trim to duration
    if len(audio) < int(sr * duration):
        audio = np.pad(audio, (0, int(sr * duration) - len(audio)))
    else:
        audio = audio[:int(sr * duration)]

    # Normalize to int16
    audio = audio / np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scipy.io.wavfile.write(filename, sr, audio_int16)
    print(f"Processing chime saved to {filename}")

create_processing_chime()
