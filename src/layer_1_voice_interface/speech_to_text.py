import threading
import queue
import numpy as np
import time
import re
import asyncio
from typing import List, Optional, Callable, Union, Awaitable
from whispercpp import Whisper

from .audio_manager import AudioManager

class SpeechToText:
    """
    Speech-to-text engine using Whisper model.
    Supports real-time transcription with silence detection and noise filtering.
    """
    
    DEFAULT_MODEL = "tiny.en"
    DEFAULT_MODEL_DIR = "models"
    
    # Whisper noise patterns to filter out
    NOISE_PATTERNS = [
        r'\(mumble\)',
        r'\(noise\)',
        r'\(music\)',
        r'\(applause\)',
        r'\(laughter\)',
        r'\(coughing\)',
        r'\(sneezing\)',
        r'\(breathing\)',
        r'\(silence\)',
        r'\(inaudible\)',
        r'\[BLANK_AUDIO\]',
        r'\[MUSIC\]',
        r'\[NOISE\]',
        r'\[SILENCE\]',
        r'\[.*?\]',  # Any text in square brackets
        r'\(.*?\)',  # Any text in parentheses that might be noise
    ]
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_dir: str = DEFAULT_MODEL_DIR,
        sample_rate: int = AudioManager.DEFAULT_SAMPLE_RATE,
        chunk_duration: float = 3.0,
        overlap_duration: float = 0.5,
        num_threads: int = 4,
        queue_maxsize: int = 10,
        silence_threshold: float = 3.0,  # 3 seconds of silence
        silence_callback: Optional[Union[Callable, Callable[[], Awaitable[None]]]] = None,
        text_callback: Optional[Union[Callable[[str], None], Callable[[str], Awaitable[None]]]] = None,
    ):
        # Configuration
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.num_threads = num_threads

        # Derived values
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples

        # Audio manager
        self.audio_manager = AudioManager(sample_rate=self.sample_rate)

        # Whisper ASR
        self.asr = Whisper.from_pretrained(model_name=model_name, basedir=model_dir)
        self.asr.params.num_threads = self.num_threads

        # Threading primitives
        self.audio_queue = queue.Queue(maxsize=queue_maxsize)
        self.stop_event = threading.Event()
        self.buffer_lock = threading.Lock()
        self.audio_buffer = []

        # Store all recognized text
        self._all_text = []
        self._all_text_lock = threading.Lock()

        # Threads
        self._recorder_thread = threading.Thread(target=self._record_stream, daemon=True)
        self._worker_thread = threading.Thread(target=self._asr_worker, daemon=True)
        
        # Status
        self.is_running = False

        self.silence_threshold = silence_threshold
        self.silence_callback = silence_callback
        self.text_callback = text_callback
        
        # Silence detection
        self.last_text_time = None
        self.silence_timer = threading.Timer(self.silence_threshold, self._on_silence)
        self.silence_detected = threading.Event()

        # Compile noise patterns for efficiency
        self.noise_regex = re.compile('|'.join(self.NOISE_PATTERNS), re.IGNORECASE)

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback for incoming audio blocks"""
        if status:
            print(f"⚠️ Audio status: {status}")
        data = indata[:, 0]  # mono
        with self.buffer_lock:
            self.audio_buffer.extend(data.tolist())
            if len(self.audio_buffer) >= self.chunk_samples:
                chunk = self.audio_buffer[: self.chunk_samples]
                try:
                    self.audio_queue.put_nowait(chunk)
                except queue.Full:
                    pass
                del self.audio_buffer[: self.step_samples]

    def _record_stream(self):
        """Starts non-blocking audio input stream"""
        self.audio_manager.start_stream(self._audio_callback)
        while not self.stop_event.is_set():
            time.sleep(0.1)

    def _filter_whisper_noise(self, text: str) -> str:
        """
        Filter out Whisper noise tokens like (mumble), (noise), [BLANK_AUDIO], etc.
        
        Args:
            text: Raw text from Whisper
            
        Returns:
            Cleaned text with noise tokens removed
        """
        if not text:
            return ""
            
        # Remove noise patterns
        cleaned = self.noise_regex.sub('', text)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Additional filtering for very short or meaningless content
        if len(cleaned) <= 2:  # Single characters or very short strings
            return ""
            
        # Filter common meaningless whisper outputs
        meaningless_phrases = [
            'uh', 'um', 'ah', 'eh', 'mm', 'hmm', 'hm',
            'thank you', 'thanks', 'bye', 'goodbye'  # Sometimes whisper hallucinates these
        ]
        
        if cleaned.lower().strip() in meaningless_phrases:
            return ""
            
        return cleaned

    def _is_meaningful_text(self, text: str) -> bool:
        """
        Check if text contains meaningful speech content
        
        Args:
            text: Text to check
            
        Returns:
            True if text is meaningful, False otherwise
        """
        if not text or len(text.strip()) == 0:
            return False
            
        # Filter out noise first
        cleaned = self._filter_whisper_noise(text)
        
        if not cleaned or len(cleaned.strip()) == 0:
            return False
            
        # Check for minimum meaningful length
        if len(cleaned.strip()) < 3:
            return False
            
        # Check if it's mostly punctuation or special characters
        alpha_chars = sum(c.isalpha() for c in cleaned)
        if alpha_chars < len(cleaned) * 0.5:  # Less than 50% alphabetic
            return False
            
        return True

    def _asr_worker(self):
        """Consumes audio chunks and performs ASR with silence detection and noise filtering"""
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            audio = np.array(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            start = time.time()
            raw_text = self.asr.transcribe(audio.tolist())
            latency = time.time() - start
            
            # Filter noise before processing
            cleaned_text = self._filter_whisper_noise(raw_text.strip())
            
            print(f"Raw: '{raw_text.strip()}' -> Cleaned: '{cleaned_text}' (⏱️ {latency:.2f}s)", flush=True)
            
            # Check if we got meaningful text after filtering
            if self._is_meaningful_text(cleaned_text):
                # Reset silence timer on meaningful text
                self._reset_silence_timer()
                self.last_text_time = time.time()
                
                # Store recognized text
                with self._all_text_lock:
                    self._all_text.append(cleaned_text)
                
                print(f"✅ Meaningful text detected: '{cleaned_text}'")
                
                # Start new silence timer after meaningful text
                self.silence_timer = threading.Timer(self.silence_threshold, self._on_silence)
                self.silence_timer.start()
                print(f"🔄 Silence timer restarted - waiting {self.silence_threshold}s for next input")
                
                # Notify callback if provided (support both sync and async)
                if self.text_callback:
                    if asyncio.iscoroutinefunction(self.text_callback):
                        # Async callback - run in new task
                        try:
                            loop = asyncio.get_event_loop()
                            loop.create_task(self.text_callback(cleaned_text))
                        except RuntimeError:
                            # No event loop, just call sync version if possible
                            pass
                    else:
                        # Sync callback
                        self.text_callback(cleaned_text)
            else:
                # Start silence timer if no meaningful text and no timer running
                if self.last_text_time is None:
                    self.last_text_time = time.time()
                    self.silence_timer = threading.Timer(self.silence_threshold, self._on_silence)
                    self.silence_timer.start()
                    print(f"🔇 Starting silence timer - no meaningful text detected")
                
            self.audio_queue.task_done()

    def _reset_silence_timer(self):
        """Reset the silence detection timer"""
        if hasattr(self, 'silence_timer') and self.silence_timer.is_alive():
            self.silence_timer.cancel()
        self.silence_detected.clear()
        
    def _on_silence(self):
        """Called when silence threshold is reached"""
        print(f"\n🔇 {self.silence_threshold}s silence detected")
        self.silence_detected.set()
        if self.silence_callback:
            if asyncio.iscoroutinefunction(self.silence_callback):
                # Async callback - run in new task
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(self.silence_callback())
                except RuntimeError:
                    # No event loop, just skip async callback
                    pass
            else:
                # Sync callback
                self.silence_callback()

    async def wait_for_silence_or_text_async(self, timeout: float = 100) -> tuple[bool, str]:
        """
        Async version of wait_for_silence_or_text.
        Wait for either silence detection or timeout
        
        Returns:
            (silence_detected, final_cleaned_text)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.silence_detected.is_set():
                with self._all_text_lock:
                    # Join all meaningful text and apply final filtering
                    final_text = " ".join(self._all_text).strip()
                    final_cleaned = self._filter_whisper_noise(final_text)
                return True, final_cleaned
            await asyncio.sleep(0.1)  # Non-blocking sleep
        
        # Timeout reached
        with self._all_text_lock:
            final_text = " ".join(self._all_text).strip()
            final_cleaned = self._filter_whisper_noise(final_text)
        return False, final_cleaned

    def wait_for_silence_or_text(self, timeout: float = 10.0) -> tuple[bool, str]:
        """
        Wait for either silence detection or timeout
        
        Returns:
            (silence_detected, final_cleaned_text)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.silence_detected.is_set():
                with self._all_text_lock:
                    # Join all meaningful text and apply final filtering
                    final_text = " ".join(self._all_text).strip()
                    final_cleaned = self._filter_whisper_noise(final_text)
                return True, final_cleaned
            time.sleep(0.1)
        
        # Timeout reached
        with self._all_text_lock:
            final_text = " ".join(self._all_text).strip()
            final_cleaned = self._filter_whisper_noise(final_text)
        return False, final_cleaned

    def reset_session(self):
        """Reset for new ASR session"""
        self._reset_silence_timer()
        with self._all_text_lock:
            self._all_text.clear()
        self.last_text_time = None
        print("🔄 ASR session reset")

    def start(self):
        """Starts ASR service"""
        if self.is_running:
            return
            
        print("▶️ Starting SpeechToText...")
        
        # Reset stop event for new session
        self.stop_event.clear()
        
        # Create new threads if previous ones have finished
        if not self._recorder_thread.is_alive():
            self._recorder_thread = threading.Thread(target=self._record_stream, daemon=True)
        if not self._worker_thread.is_alive():
            self._worker_thread = threading.Thread(target=self._asr_worker, daemon=True)
        
        self.is_running = True
        self._recorder_thread.start()
        self._worker_thread.start()
        print("✅ SpeechToText started")

    def stop(self) -> str:
        """
        Stops ASR service and returns all recognized text concatenated.
        Keeps model weights in memory, only stops detection threads.
        
        Returns:
            All recognized text joined as a single string, filtered
        """
        if not self.is_running:
            return ""
            
        print("\n❌ Stopping SpeechToText...")
        self.stop_event.set()
        
        # Get final text before cleanup
        with self._all_text_lock:
            final_text = " ".join(self._all_text).strip()
            final_cleaned = self._filter_whisper_noise(final_text)
        
        # Stop threads
        if self._recorder_thread.is_alive():
            self._recorder_thread.join()
        if self._worker_thread.is_alive():
            self._worker_thread.join()
            
        # Close audio stream
        self.audio_manager.close()
        
        # Clear buffers (but keep ASR model in memory)
        with self.buffer_lock:
            self.audio_buffer.clear()
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Clear text storage
        with self._all_text_lock:
            self._all_text.clear()
        
        self.is_running = False
        print("✅ SpeechToText stopped (model weights preserved)")
        
        # Return the saved final text
        return final_cleaned

    async def stop_async(self) -> str:
        """
        Async version of stop. Stops ASR service and returns all recognized text.
        Keeps model weights in memory, only stops detection threads.
        
        Returns:
            All recognized text joined as a single string, filtered
        """
        if not self.is_running:
            return ""
            
        print("\n❌ Stopping SpeechToText...")
        self.stop_event.set()
        
        # Get final text before cleanup
        with self._all_text_lock:
            final_text = " ".join(self._all_text).strip()
            final_cleaned = self._filter_whisper_noise(final_text)
        
        # Run thread joins in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        
        # Stop threads
        if self._recorder_thread.is_alive():
            await loop.run_in_executor(None, self._recorder_thread.join)
        if self._worker_thread.is_alive():
            await loop.run_in_executor(None, self._worker_thread.join)
        
        # Close audio stream
        self.audio_manager.close()
        
        # Clear buffers (but keep ASR model in memory)
        with self.buffer_lock:
            self.audio_buffer.clear()
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # Clear text storage
        with self._all_text_lock:
            self._all_text.clear()
        
        self.is_running = False
        print("✅ SpeechToText stopped (model weights preserved)")
        
        # Return the saved final text
        return final_cleaned
            
    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe audio from a file
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        # Implementation would depend on the Whisper API
        # This is a placeholder
        return self.asr.transcribe_file(audio_file_path)