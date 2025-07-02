import numpy as np
import sounddevice as sd
import asyncio
import threading
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
        
        # Playback control
        self._playback_stop_event = asyncio.Event()
        self._current_playback = None
        self._is_playing = False
    
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
        # Normalize and amplify audio for proper volume
        amplified_audio = self.normalize_and_amplify_audio(audio_data, target_amplitude=0.9)
        
        # Convert to int16 for better compatibility if needed
        if amplified_audio.dtype == np.float32:
            audio_int16 = (amplified_audio.astype(np.float64) * 32767.0).astype(np.int16)
        else:
            audio_int16 = amplified_audio
            
        self._current_playback = sd.play(audio_int16, samplerate=self.sample_rate, blocking=blocking)
    
    def stop_playback(self, fade_out_ms: int = 200) -> None:
        """
        Stop current audio playback immediately or with fade-out.
        
        Args:
            fade_out_ms: Fade-out duration in milliseconds. 0 = immediate stop.
        """
        print(f"🛑 Setting stop event (fade_out: {fade_out_ms}ms)")
        self._playback_stop_event.set()
        
        # For OutputStream-based playback, the abort will be handled in the playback loop
        # This just sets the signal - the actual stopping happens in _play_audio_interruptible
    
    async def play_audio_async(
        self, 
        audio_data: np.ndarray, 
        blocking: bool = True,
        interruptible: bool = False
    ) -> bool:
        """
        Async version of play_audio. If blocking=True, runs in executor to avoid blocking event loop.
        
        Args:
            audio_data: Audio data to play
            blocking: Whether to wait for playback completion
            interruptible: Whether this playback can be interrupted
            
        Returns:
            True if completed normally, False if interrupted
        """
        self._playback_stop_event.clear()
        
        # Normalize and amplify audio for proper volume
        amplified_audio = self.normalize_and_amplify_audio(audio_data, target_amplitude=0.9)
        
        # Convert to float32 for OutputStream
        if amplified_audio.dtype != np.float32:
            if amplified_audio.dtype == np.int16:
                audio_float32 = amplified_audio.astype(np.float32) / 32767.0
            else:
                audio_float32 = amplified_audio.astype(np.float32)
        else:
            audio_float32 = amplified_audio
            
        print(f"🔊 Playing audio - Shape: {audio_float32.shape}, Peak: {np.max(np.abs(audio_float32)):.6f}")
        
        if not blocking:
            # Non-blocking, just start playback (use old method for non-interruptible)
            self._current_playback = sd.play(audio_float32, samplerate=self.sample_rate, blocking=False)
            return True
        else:
            # Blocking version
            if interruptible:
                # Interruptible playback using OutputStream
                return await self._play_audio_interruptible(audio_float32)
            else:
                # Regular blocking playback
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: sd.play(audio_float32, samplerate=self.sample_rate, blocking=True)
                )
                return True
    
    async def _play_audio_interruptible(self, audio_data: np.ndarray) -> bool:
        """
        Play audio with ability to be interrupted via stop_playback()
        """
        audio_with_fade = self.apply_fade_in(audio_data, fade_ms=50)
        expected_duration = len(audio_with_fade) / self.sample_rate
        print(f"🎵 Starting interruptible playback - Duration: {expected_duration:.2f}s")

        self._playback_stop_event.clear()
        self._is_playing = True

        try:
            # Create OutputStream for proper interrupt control
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )

            with stream:
                stream.write(audio_with_fade)
                start_time = asyncio.get_event_loop().time()

                # Monitor stream while it's active
                while stream.active:
                    # Check for interrupt signal
                    if self._playback_stop_event.is_set():
                        print("🔄 Interrupt detected - stopping playback")
                        stream.abort()
                        await asyncio.sleep(0.2)  # Allow fade-out to apply if needed
                        self._is_playing = False
                        return False

                    # Timeout safety
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > expected_duration + 1.0:
                        print(f"⚠️ Playback timeout after {elapsed:.2f}s")
                        break

                    await asyncio.sleep(0.01)  # Check every 10ms

            print("✅ Interruptible playback completed successfully")
            self._is_playing = False
            return True

        except Exception as e:
            print(f"⚠️ Error in interruptible playback: {e}")
            self._is_playing = False
            return False

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
    
    async def record_audio_async(self, duration: float) -> np.ndarray:
        """
        Async version of record_audio. Runs in executor to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocking=True
            )
        )
    
    def apply_fade(self, audio: np.ndarray, fade_ms: int = 10) -> np.ndarray:
        """
        Apply fade in/out to audio to prevent clicks.
        """
        # Use existing fade in/out methods to avoid code duplication
        audio_faded = self.apply_fade_in(audio, fade_ms)
        audio_faded = self.apply_fade_out(audio_faded, fade_ms)
        return audio_faded

    def apply_fade_in(self, audio: np.ndarray, fade_ms: int = 50) -> np.ndarray:
        """
        Apply fade-in to audio for smooth start.
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        if fade_samples <= 0 or audio.shape[0] < fade_samples:
            return audio

        # Handle different data types
        original_dtype = audio.dtype
        if original_dtype == np.int16:
            # Convert to float for processing
            audio_float = audio.astype(np.float32) / 32767.0
        else:
            audio_float = audio.astype(np.float32)

        fade_in = np.linspace(0.0, 1.0, fade_samples)
        audio_copy = audio_float.copy()
        audio_copy[:fade_samples] *= fade_in
        
        # Convert back to original dtype if needed
        if original_dtype == np.int16:
            return (audio_copy * 32767.0).astype(np.int16)
        else:
            return audio_copy

    def apply_fade_out(self, audio: np.ndarray, fade_ms: int = 200) -> np.ndarray:
        """
        Apply fade-out to audio for smooth ending.
        """
        fade_samples = int(self.sample_rate * fade_ms / 1000)
        if fade_samples <= 0 or audio.shape[0] < fade_samples:
            return audio

        # Handle different data types
        original_dtype = audio.dtype
        if original_dtype == np.int16:
            # Convert to float for processing
            audio_float = audio.astype(np.float32) / 32767.0
        else:
            audio_float = audio.astype(np.float32)

        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio_copy = audio_float.copy()
        audio_copy[-fade_samples:] *= fade_out
        
        # Convert back to original dtype if needed
        if original_dtype == np.int16:
            return (audio_copy * 32767.0).astype(np.int16)
        else:
            return audio_copy
    
    def normalize_and_amplify_audio(self, audio: np.ndarray, target_amplitude: float = 0.8) -> np.ndarray:
        """
        Normalize and amplify audio to ensure proper volume levels.
        
        Args:
            audio: Input audio array
            target_amplitude: Target peak amplitude (0.0 to 1.0)
            
        Returns:
            Amplified audio array
        """
        if len(audio) == 0:
            return audio
            
        # Remove DC offset
        audio_centered = audio - np.mean(audio)
        
        # Find current peak amplitude
        current_peak = np.max(np.abs(audio_centered))
        
        if current_peak > 0:
            # Calculate amplification factor
            amplification_factor = target_amplitude / current_peak
            
            # Apply amplification but prevent clipping
            amplified = audio_centered * amplification_factor
            
            # Ensure no clipping
            amplified = np.clip(amplified, -1.0, 1.0)
            
            print(f"🔊 Audio amplified: {current_peak:.6f} -> {np.max(np.abs(amplified)):.6f}")
            return amplified
        else:
            print("⚠️ Audio is silent (no peak detected)")
            return audio_centered
    
    def close(self) -> None:
        """Clean up all audio resources."""
        self.stop_recording()
        self.stop_playback()
        
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

