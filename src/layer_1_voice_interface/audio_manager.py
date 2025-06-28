import numpy as np
import sounddevice as sd
from typing import Callable, Optional, List, Tuple

class AudioManager:
    """
    Core audio interface for handling audio input/output operations.
    Provides unified interface for recording, playback, and stream management.
    """
    
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_DTYPE = 'int16'
    
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        dtype: str = DEFAULT_DTYPE,
        buffer_size: int = 1024
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.buffer_size = buffer_size
        
        self._input_stream = None
        self._output_stream = None
    
    def start_recording(self, callback: Callable) -> None:
        """
        Start recording audio with the provided callback function
        """
        if self._input_stream is not None:
            self.stop_recording()
            
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.buffer_size,
            callback=callback
        )
        self._input_stream.start()
    
    def stop_recording(self) -> None:
        """Stop the recording stream if active."""
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
    
    def play_audio(self, audio_data: np.ndarray, blocking: bool = True) -> None:
        """
        Play audio data through the default output device.
        """
        sd.play(audio_data, samplerate=self.sample_rate, blocking=blocking)
    
    def start_stream(self, callback: Callable) -> None:
        """
        Start a continuous audio stream with the provided callback.
        """
        if self._input_stream is not None:
            self.stop_recording()
            
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=callback
        )
        self._input_stream.start()
    
    def record_audio(self, duration: float) -> np.ndarray:
        """
        Record audio for the specified duration.
        """
        return sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocking=True
        )
    
    def apply_fade(self, audio: np.ndarray, fade_ms: int = 10) -> np.ndarray:
        """
        Apply fade in/out to audio to prevent clicks.
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        if fade_samples <= 0 or audio.shape[0] < 2 * fade_samples:
            return audio
            
        fade_in = np.linspace(0.0, 1.0, fade_samples)
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        
        audio_copy = audio.copy()
        audio_copy[:fade_samples] *= fade_in
        audio_copy[-fade_samples:] *= fade_out
        
        return audio_copy
    
    def close(self) -> None:
        """Clean up all audio resources."""
        self.stop_recording()
        
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

