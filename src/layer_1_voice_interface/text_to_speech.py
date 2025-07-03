import torch
import json
import numpy as np
import re
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Callable
import threading

from openvoice import se_extractor  # type: ignore
from openvoice.api import BaseSpeakerTTS, ToneColorConverter  # type: ignore

from .audio_manager import AudioManager
from .wake_word_handler import InterruptWakeWordHandler

INTERRUPT_WAKE_WORD_MODEL = ["models/openwakeword/alexa_v0.1.tflite"]

class TextToSpeech:
    """
    Text-to-Speech engine using OpenVoice, supports text splitting, voice conversion,
    and direct audio playback. Loads resources lazily in separate background thread.
    Now supports interrupt detection during playback.
    """
    
    DEFAULT_BASE_DIR = 'openvoice/checkpoints/base_speakers/EN'
    DEFAULT_CONVERTER_DIR = 'openvoice/checkpoints/converter'
    DEFAULT_REFERENCE_PATH = 'assets/audio/ref_voice.mp3'
    DEFAULT_OUTPUT_DIR = 'openvoice/outputs'
    
    def __init__(
        self,
        ckpt_base_dir: str = DEFAULT_BASE_DIR,
        ckpt_converter_dir: str = DEFAULT_CONVERTER_DIR,
        reference_speaker_path: str = DEFAULT_REFERENCE_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        device: Optional[str] = None,
        max_len: int = 200,
        fade_ms: int = 10,
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ckpt_base = Path(ckpt_base_dir)
        self.ckpt_conv = Path(ckpt_converter_dir)
        self.ref_path = Path(reference_speaker_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_len = max_len
        self.fade_ms = fade_ms

        # Audio manager with correct sample rate for TTS
        self.audio_manager = AudioManager(sample_rate=48000) # Sample rate for output stream

        # Executor for synthesis tasks
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.ready = False

        # Attributes to initialize
        self.base_tts = None
        self.converter = None
        self.source_se = None
        self.target_se = None

        # Start dedicated thread for loading (non-daemon to ensure scheduling)
        self._load_thread = threading.Thread(target=self._lazy_init)
        self._load_thread.start()
        
        # Interrupt handling
        self._interrupt_handler: Optional[InterruptWakeWordHandler] = None
        self._interrupt_callback: Optional[Callable[[], None]] = None
        self._is_interrupted = threading.Event()

    def _lazy_init(self):
        try:
            print("🔄 Loading TTS model in background...")
            # Load base TTS model config
            with open(self.ckpt_base / 'config.json', 'r') as f:
                cfg = json.load(f)

            # Instantiate and load checkpoints
            self.base_tts = BaseSpeakerTTS(str(self.ckpt_base / 'config.json'), device=self.device)
            self.base_tts.load_ckpt(str(self.ckpt_base / 'checkpoint.pth'))
            self.converter = ToneColorConverter(str(self.ckpt_conv / 'config.json'), device=self.device)
            self.converter.load_ckpt(str(self.ckpt_conv / 'checkpoint.pth'))

            # Speaker embeddings
            self.source_se = torch.load(self.ckpt_base / 'en_default_se.pth').to(self.device)
            self.target_se = self._extract_target_se()

            self.ready = True
            print("✅ TTS model loaded and ready.")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")

    def _extract_target_se(self):
        se, _ = se_extractor.get_se(
            str(self.ref_path),
            self.converter,
            target_dir=str(self.output_dir / 'processed'),
            vad=True
        )
        return se

    def _split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS processing"""
        sentences = re.split(r'(?<=[.!?]) +|\n+', text)
        chunks, current = [], ''
        for sent in sentences:
            sent = sent.strip()
            if len(current) + len(sent) + 1 <= self.max_len:
                current += sent + ' '
            else:
                chunks.append(current.strip())
                current = sent + ' '
        if current:
            chunks.append(current.strip())
        return chunks

    def _generate_chunk(
        self,
        text_chunk: str,
        idx: int,
        speaker: str,
        language: str,
        speed: float,
        emotion: str
    ) -> np.ndarray:
        tmp = self.output_dir / f'chunk_{idx}.wav'
        if not self.ready:
            raise RuntimeError("TTS model not loaded yet.")
        self.base_tts.tts(text_chunk, str(tmp), speaker=speaker, language=language, speed=speed)
        audio = self.converter.convert(
            audio_src_path=str(tmp),
            src_se=self.source_se,
            tgt_se=self.target_se,
            message=emotion
        )
        
        # Debug TTS audio properties
        print(f"🎵 TTS chunk {idx} - Shape: {audio.shape}, Type: {audio.dtype}, Range: {audio.min():.6f} to {audio.max():.6f}")
        
        faded_audio = self.audio_manager.apply_fade(audio, self.fade_ms)
        
        print(f"🎵 TTS chunk {idx} after fade - Range: {faded_audio.min():.6f} to {faded_audio.max():.6f}")
        
        return faded_audio

    async def speak_async(
        self,
        text: str,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell',
        play_audio: bool = True,
        interruptible: bool = False,
        interrupt_callback: Optional[Callable[[], None]] = None
    ) -> dict:
        """
        Async version of speak. Convert text to speech and optionally play it.
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice to use
            language: Language for synthesis
            speed: Speech rate multiplier
            emotion: Emotion style for voice
            play_audio: Whether to play audio immediately
            interruptible: Whether this speech can be interrupted by wake word
            interrupt_callback: Callback to call if interrupted
            
        Returns:
            Dict with 'completed': bool, 'interrupted': bool, 'audio': np.ndarray
        """
        print(f"🎯 TTS speak_async started - Text: '{text[:50]}...', Play: {play_audio}, Interruptible: {interruptible}")
        
        result = {
            'completed': False,
            'interrupted': False,
            'audio': np.array([], dtype=np.float32)
        }
        
        # Ensure load thread has opportunity to run early
        if not self.ready and self._load_thread.is_alive():
            print("🕐 Waiting briefly for TTS to load...")
            # Run thread join in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._load_thread.join(timeout=1.0))
        if not self.ready:
            # If still not ready, block fully
            print("🕐 Blocking until TTS ready...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_thread.join)
            
        chunks = self._split_text(text)
        if not chunks:
            result['completed'] = True
            print("🎯 TTS speak_async completed - No chunks to process")
            return result

        print(f"🎯 TTS processing {len(chunks)} chunks")

        # Setup interrupt monitoring if requested (setup EARLY before synthesis)
        if interruptible:
            print("🎯 Setting up interrupt monitoring...")
            await self._setup_interrupt_monitoring(interrupt_callback)

        try:
            # SOLUTION 1: Generate ALL chunks in parallel first
            loop = asyncio.get_event_loop()
            futures: List[asyncio.Future] = []
            for i, chunk_text in enumerate(chunks):
                future = loop.run_in_executor(
                    self.executor,
                    self._generate_chunk, chunk_text, i, speaker, language, speed, emotion
                )
                futures.append(future)

            # Wait for ALL chunks to complete in parallel
            print(f"🎯 Generating all {len(futures)} chunks in parallel...")
            chunk_audios = await asyncio.gather(*futures)
            
            # SOLUTION 2: Concatenate all chunks into one continuous audio stream
            if chunk_audios:
                # Apply smooth transitions between chunks
                combined_audio_parts = []
                for i, audio in enumerate(chunk_audios):
                    if i == 0:
                        # First chunk: apply fade-in only
                        combined_audio_parts.append(self.audio_manager.apply_fade_in(audio, fade_ms=50))
                    elif i == len(chunk_audios) - 1:
                        # Last chunk: apply fade-out only  
                        combined_audio_parts.append(self.audio_manager.apply_fade_out(audio, fade_ms=100))
                    else:
                        # Middle chunks: no additional fading (already has basic fade from _generate_chunk)
                        combined_audio_parts.append(audio)
                
                # Concatenate all chunks into one continuous stream
                combined_audio = np.concatenate(combined_audio_parts)
                result['audio'] = combined_audio
                                
                # SOLUTION 3: Play as ONE continuous audio stream (no gaps between chunks)
                if play_audio:
                    print(f"🎯 Playing continuous audio stream...")
                    completed = await self.audio_manager.play_audio_async(
                        combined_audio, 
                        blocking=True, 
                        interruptible=interruptible
                    )
                    
                    if not completed:
                        # Interrupted during playback
                        print("🔄 TTS interrupted during continuous playback")
                        result['interrupted'] = True
                    else:
                        print("✅ Continuous audio playback completed successfully")
                        result['completed'] = True
                else:
                    # Not playing, just return audio
                    result['completed'] = True
            else:
                print("⚠️ No audio chunks generated")
                result['completed'] = True

        except Exception as e:
            print(f"⚠️ Error in speak_async: {e}")
            result['interrupted'] = True
            
        finally:
            # Always cleanup interrupt monitoring
            if interruptible:
                print("🎯 Cleaning up interrupt monitoring...")
                await self._cleanup_interrupt_monitoring()

        print(f"🎯 TTS speak_async returning - Completed: {result['completed']}, Interrupted: {result['interrupted']}")
        return result
    
    async def _setup_interrupt_monitoring(self, interrupt_callback: Optional[Callable[[], None]]) -> None:
        """IMPROVED: Setup interrupt wake word monitoring during TTS - can be called multiple times"""
        try:
            self._interrupt_callback = interrupt_callback
            self._is_interrupted.clear()
            
            # FIXED: Always create fresh interrupt handler for reliable detection
            # This ensures clean state for each TTS session
            if self._interrupt_handler is not None:
                # Clean up existing handler first
                try:
                    if self._interrupt_handler.is_monitoring():
                        self._interrupt_handler.stop_interrupt_monitoring()
                except:
                    pass
            
            # Create new interrupt handler for this session
            self._interrupt_handler = InterruptWakeWordHandler(
                wakeword_models=INTERRUPT_WAKE_WORD_MODEL
            )
            
            # Set interrupt callback
            self._interrupt_handler.set_interrupt_callback(self._on_interrupt_detected)
            
            # Start monitoring
            self._interrupt_handler.start_interrupt_monitoring()
            print("✅ Interrupt monitoring setup completed")
            
        except Exception as e:
            print(f"⚠️ Error setting up interrupt monitoring: {e}")
    
    async def _cleanup_interrupt_monitoring(self) -> None:
        """IMPROVED: Cleanup interrupt monitoring - ensures clean state for next session"""
        try:
            if self._interrupt_handler and self._interrupt_handler.is_monitoring():
                print("🧹 Cleaning up interrupt monitoring...")
                self._interrupt_handler.stop_interrupt_monitoring()
                print("✅ Interrupt monitoring cleaned up successfully")
            
            # FIXED: Reset interrupt handler reference to None for clean state
            # This ensures fresh setup for next TTS session
            self._interrupt_handler = None
            self._interrupt_callback = None
            self._is_interrupted.clear()
            
        except Exception as e:
            print(f"⚠️ Error cleaning up interrupt monitoring: {e}")
            # Force cleanup even on error
            self._interrupt_handler = None
            self._interrupt_callback = None
    
    def _on_interrupt_detected(self) -> None:
        """IMPROVED: Called when wake word detected during TTS - provides immediate audio stop"""
        print("⚡ TTS INTERRUPT DETECTED - STOPPING IMMEDIATELY!")
        self._is_interrupted.set()
        
        # CRITICAL: Stop current audio playback IMMEDIATELY with minimal fade
        self.audio_manager.stop_playback(fade_out_ms=50)  # Very quick fade to avoid clicks
        
        # Call user-provided interrupt callback
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
                print("✅ User interrupt callback executed")
            except Exception as e:
                print(f"⚠️ Error in user interrupt callback: {e}")
        
        print("🚨 AUDIO STOPPED - Interrupt complete")

    def speak(
        self,
        text: str,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell',
        play_audio: bool = True
    ) -> np.ndarray:
        """
        Convert text to speech and optionally play it
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice to use
            language: Language for synthesis
            speed: Speech rate multiplier
            emotion: Emotion style for voice
            play_audio: Whether to play audio immediately
            
        Returns:
            Audio data as numpy array
        """
        # Ensure load thread has opportunity to run early
        if not self.ready and self._load_thread.is_alive():
            print("🕐 Waiting briefly for TTS to load...")
            self._load_thread.join(timeout=0)
        if not self.ready:
            # If still not ready, block fully
            print("🕐 Blocking until TTS ready...")
            self._load_thread.join()
            
        chunks = self._split_text(text)
        if not chunks:
            return np.array([], dtype=np.float32)

        futures: List[Future] = []
        for i, chunk_text in enumerate(chunks):
            futures.append(self.executor.submit(
                self._generate_chunk, chunk_text, i, speaker, language, speed, emotion
            ))

        combined_audio = []
        for fut in futures:
            audio = fut.result()
            combined_audio.append(audio)
            if play_audio:
                self.audio_manager.play_audio(audio)

        return np.concatenate(combined_audio)

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        if self._load_thread.is_alive():
            self._load_thread.join()
        self.audio_manager.close()