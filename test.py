import numpy as np
import sounddevice as sd

fs = 16000  # Sample rate
duration = 2.0  # seconds
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

# Tạo sóng sine có biên độ giống Elssa TTS
tone = 0.4 * np.sin(2 * np.pi * 440 * t)  # 440Hz là nốt La (A4)
print("Range:", tone.min(), tone.max())  # -0.4 ~ 0.4

sd.play(tone, samplerate=fs, blocking=True)
