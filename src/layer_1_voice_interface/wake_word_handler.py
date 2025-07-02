import sys
import threading
import time
import asyncio
from typing import Callable, List, Optional, Union, Awaitable
from collections import deque
import numpy as np

import openwakeword
from openwakeword.model import Model
from .audio_manager import AudioManager

import warnings
# Suppress UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

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

        # Callbacks - support both sync and async
        self._callbacks: List[Union[Callable[[], None], Callable[[], Awaitable[None]]]] = []

        # Threading
        self._stop_event = threading.Event()
        self._stream_thread = None

        # Frame buffer
        self.audio_buffer = deque(maxlen=self.block_size)
        
        # Store reference to main event loop for async callbacks
        self._main_loop = None

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
                        self._trigger_callbacks()
            time.sleep(0.01)

    def _trigger_callbacks(self):
        """Trigger all registered callbacks, handling both sync and async versions"""
        for cb in self._callbacks:
            if asyncio.iscoroutinefunction(cb):
                # Async callback - submit to main event loop
                if self._main_loop and not self._main_loop.is_closed():
                    try:
                        asyncio.run_coroutine_threadsafe(cb(), self._main_loop)
                        print("✅ Async callback submitted to main loop")
                    except Exception as e:
                        print(f"⚠️ Error submitting async callback: {e}")
                else:
                    print("⚠️ Async callback skipped - no main event loop available")
            else:
                # Sync callback - run in daemon thread
                threading.Thread(target=cb, daemon=True).start()

    def register_callback(self, callback: Union[Callable[[], None], Callable[[], Awaitable[None]]]) -> None:
        """Register a callback to invoke when wake word is detected. Supports both sync and async callbacks."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Begin wake word detection"""
        # Capture reference to current event loop if available
        try:
            self._main_loop = asyncio.get_running_loop()
            print("✅ Captured reference to main event loop")
        except RuntimeError:
            print("⚠️ No event loop running - async callbacks will be skipped")
            self._main_loop = None
            
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
        self._main_loop = None
        print("✅ Wake word detection stopped")


class InterruptWakeWordHandler:
    """
    Specialized wake word handler for interrupt detection during TTS playback.
    Lightweight, fast detection with immediate callback on wake word.
    """
    def __init__(
        self,
        wakeword_models: List[str] = ["models/openwakeword/elssa_v0.1.tflite"],
        sample_rate: int = AudioManager.DEFAULT_SAMPLE_RATE,
        block_size: int = 256,  # Smaller for faster response
        threshold: float = 0.8  # Slightly lower for more sensitivity during TTS
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.threshold = threshold
        
        # Load OpenWakeWord model (reuse download if already done)
        try:
            openwakeword.utils.download_models()
        except:
            pass  # Models might already be downloaded
        self.model = Model(wakeword_models=wakeword_models)
        
        # Dedicated audio manager for interrupt detection
        self.audio_manager = AudioManager(sample_rate=sample_rate)
        
        # Interrupt callback
        self._interrupt_callback: Optional[Callable[[], None]] = None
        
        # Threading
        self._stop_event = threading.Event()
        self._interrupt_thread = None
        
        # Frame buffer
        self.audio_buffer = deque(maxlen=self.block_size)
        
        # State
        self._is_active = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback optimized for interrupt detection"""
        if status:
            return  # Skip logging during TTS to avoid noise
            
        # Only process if actively listening for interrupts
        if self._is_active:
            self.audio_buffer.extend(indata[:, 0].tolist())

    def _interrupt_loop(self):
        """Optimized loop for fast interrupt detection"""
        self.audio_manager.start_stream(self._audio_callback)
        
        while not self._stop_event.is_set():
            if self._is_active and len(self.audio_buffer) >= self.block_size:
                samples = np.array([self.audio_buffer.popleft() for _ in range(self.block_size)])
                scores = self.model.predict(samples, threshold=self.threshold)
                
                for name, score in scores.items():
                    if score > self.threshold:
                        print(f"⚡ INTERRUPT: Wake word '{name}' detected during TTS! (score={score:.2f})")
                        self._trigger_interrupt()
                        return  # Exit immediately on interrupt
                        
            time.sleep(0.005)  # Faster polling for interrupt

    def _trigger_interrupt(self):
        """Trigger interrupt callback immediately"""
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
            except Exception as e:
                print(f"⚠️ Error in interrupt callback: {e}")

    def set_interrupt_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when wake word detected during TTS"""
        self._interrupt_callback = callback

    def start_interrupt_monitoring(self) -> None:
        """Start monitoring for wake word interrupts"""
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            return  # Already running
            
        self._stop_event.clear()
        self._is_active = True
        self._interrupt_thread = threading.Thread(target=self._interrupt_loop, daemon=True)
        self._interrupt_thread.start()
        print("🎯 Interrupt monitoring started")

    def stop_interrupt_monitoring(self) -> None:
        """Stop monitoring for wake word interrupts"""
        self._is_active = False
        self._stop_event.set()
        
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            self._interrupt_thread.join(timeout=1.0)  # Don't wait too long
            
        self.audio_manager.close()
        print("🎯 Interrupt monitoring stopped")

    def is_monitoring(self) -> bool:
        """Check if currently monitoring for interrupts"""
        return self._is_active and not self._stop_event.is_set()
