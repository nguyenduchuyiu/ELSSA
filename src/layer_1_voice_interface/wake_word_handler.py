import sys
import threading
import time
from typing import Callable, List, Optional
from collections import deque
import numpy as np

import openwakeword
from openwakeword.model import Model
from .audio_manager import AudioManager

class WakeWordHandler:
    """
    Wake word detection using OpenWakeWord (open-source).
    Listens on microphone for specified keywords and triggers callbacks.
    """
    def __init__(
        self,
        wakeword_models: List[str] = ["models/openwakeword/elssa_v0.1.tflite"],
        sample_rate: int = AudioManager.DEFAULT_SAMPLE_RATE,
        block_size: int = 512,
        threshold: float = 0.9
    ):
        openwakeword.utils.download_models()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold


        # Load OpenWakeWord model
        self.model = Model(wakeword_models=wakeword_models)


        # Audio manager
        self.audio_manager = AudioManager()

        # Callbacks
        self._callbacks: List[Callable[[], None]] = []

        # Threading
        self._stop_event = threading.Event()
        self._stream_thread = None

        # Frame buffer
        self.audio_buffer = deque(maxlen=self.block_size)

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Wake word audio status: {status}")
        self.audio_buffer.extend(indata[:, 0].tolist())

    def _stream_loop(self):
        self.audio_manager.start_stream(self._audio_callback)
        print("🎙️ Listening for wake words...")

        while not self._stop_event.is_set():
            if len(self.audio_buffer) >= self.block_size:
                samples = np.array([self.audio_buffer.popleft() for _ in range(self.block_size)])
                scores = self.model.predict(samples, threshold=self.threshold)
                for name, score in scores.items():
                    if score > self.threshold:
                        print(f"🔊 Wake word '{name}' detected! (score={score:.2f})")
                        for cb in self._callbacks:
                            threading.Thread(target=cb, daemon=True).start()
            time.sleep(0.01)

    def register_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to invoke when wake word is detected"""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Begin wake word detection"""
        self._stop_event.clear()
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()
        print("✅ Wake word detection started")

    def stop(self) -> None:
        """Stop wake word detection"""
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join()
        self.audio_manager.close()
        print("✅ Wake word detection stopped")
