import torch
import json
import numpy as np
import re
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Callable
import threading

from openvoice import se_extractor  
from openvoice.api import BaseSpeakerTTS, ToneColorConverter  

from .audio_manager import AudioManager
from .wake_word_handler import InterruptWakeWordHandler
from .playback_buffer import PlaybackBuffer
from .streaming_components import StreamingOrchestrator


class TextToSpeech:
    """
    Text-to-Speech engine using OpenVoice.
    Supports both standard and streaming TTS with seamless audio playback.
    Provides interrupt detection during playback.
    """
    
    DEFAULT_BASE_DIR = 'models/openvoice/checkpoints/base_speakers/EN'
    DEFAULT_CONVERTER_DIR = 'models/openvoice/checkpoints/converter'
    DEFAULT_REFERENCE_PATH = 'assets/audio/ref_voice.mp3'
    DEFAULT_OUTPUT_DIR = 'models/openvoice/outputs'
    
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

        # TTS model attributes
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
        
        # Streaming orchestrator
        self._streaming_orchestrator = StreamingOrchestrator(self)

    def _lazy_init(self):
        """Initialize TTS models in background thread"""
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
            print("✅ TTS model loaded successfully")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")

    def _extract_target_se(self):
        """Extract target speaker embedding from reference audio"""
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
        """Generate audio chunk from text"""
        if not self.ready:
            raise RuntimeError("TTS model not loaded yet.")
        audio = self.base_tts.tts(text_chunk, speaker=speaker, language=language, speed=speed)
        audio = self.converter.convert(
            raw_audio=audio,
            src_se=self.source_se,
            tgt_se=self.target_se,
            message=emotion
        )
        
        # Apply fade
        faded_audio = self.audio_manager.apply_fade(audio, self.fade_ms)
        
        # Debug info
        print(f"🎵 Generated chunk {idx}: {len(faded_audio)} samples at 22050 Hz")
        
        return faded_audio
    
    async def _generate_audio_chunk_async(self, text: str, idx: int, speaker: str, language: str, speed: float, emotion: str) -> np.ndarray:
        """Async wrapper for audio chunk generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._generate_chunk,
            text, idx, speaker, language, speed, emotion
        )

    def _apply_smooth_transition(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Apply smooth transition for seamless audio chunk connection"""
        # Apply gentle fade-in to reduce potential clicks between chunks
        fade_samples = int(0.005 * 22050)  # 5ms fade at TTS sample rate
        if len(audio_chunk) > fade_samples:
            fade_curve = np.linspace(0.0, 1.0, fade_samples)
            audio_chunk[:fade_samples] *= fade_curve
        return audio_chunk
    
    # ========== Interrupt Handling ==========
    
    async def _setup_interrupt_monitoring(self, interrupt_callback: Optional[Callable[[], None]]) -> None:
        """Setup interrupt wake word monitoring during TTS"""
        try:
            self._interrupt_callback = interrupt_callback
            self._is_interrupted.clear()
            
            # Clean up existing handler first
            if self._interrupt_handler is not None:
                try:
                    if self._interrupt_handler.is_monitoring():
                        self._interrupt_handler.stop_interrupt_monitoring()
                except:
                    pass
            
            # Create new interrupt handler for this session
            self._interrupt_handler = InterruptWakeWordHandler()
            self._interrupt_handler.set_interrupt_callback(self._on_interrupt_detected)
            self._interrupt_handler.start_interrupt_monitoring()
            print("✅ Interrupt monitoring setup completed")
            
        except Exception as e:
            print(f"⚠️ Error setting up interrupt monitoring: {e}")
    
    async def _cleanup_interrupt_monitoring(self) -> None:
        """Cleanup interrupt monitoring"""
        try:
            if self._interrupt_handler and self._interrupt_handler.is_monitoring():
                print("🧹 Cleaning up interrupt monitoring...")
                self._interrupt_handler.stop_interrupt_monitoring()
                print("✅ Interrupt monitoring cleaned up successfully")
            
            self._interrupt_handler = None
            self._interrupt_callback = None
            self._is_interrupted.clear()
            
        except Exception as e:
            print(f"⚠️ Error cleaning up interrupt monitoring: {e}")
            self._interrupt_handler = None
            self._interrupt_callback = None
    
    def _on_interrupt_detected(self) -> None:
        """Called when wake word detected during TTS"""
        print("⚡ TTS INTERRUPT DETECTED - STOPPING IMMEDIATELY!")
        self._is_interrupted.set()
        
        # Stop current audio playback immediately
        self.audio_manager.stop_playback(fade_out_ms=50)
        
        # Call user-provided interrupt callback
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
                print("✅ User interrupt callback executed")
            except Exception as e:
                print(f"⚠️ Error in user interrupt callback: {e}")
        
        print("🚨 AUDIO STOPPED - Interrupt complete")

    # ========== Public API Methods ==========

    async def speak_stream_async(
        self,
        text_stream,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell',
        interruptible: bool = False,
        interrupt_callback: Optional[Callable[[], None]] = None
    ) -> dict:
        """
        Stream text chunks to TTS with continuous playback buffer to eliminate gaps.
        Uses 3-component architecture: Producer, Feeder, and Continuous Player.
        
        Args:
            text_stream: Iterator/generator that yields text chunks
            speaker: Speaker voice to use
            language: Language for synthesis
            speed: Speech rate multiplier
            emotion: Emotion style for voice
            interruptible: Whether this speech can be interrupted by wake word
            interrupt_callback: Callback to call if interrupted
            
        Returns:
            Dict with 'completed': bool, 'interrupted': bool, 'text': str
        """
        print(f"🎯 TTS speak_stream_async started - Interruptible: {interruptible}")
        
        result = {
            'completed': False,
            'interrupted': False,
            'text': '',
            'audio': np.array([], dtype=np.float32)
        }
        
        # Ensure TTS is ready
        if not self.ready and self._load_thread.is_alive():
            print("🕐 Waiting for TTS to load...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_thread.join)
            
        if not self.ready:
            print("❌ TTS not ready")
            return result

        # Setup interrupt monitoring if requested
        if interruptible:
            print("🎯 Setting up interrupt monitoring for streaming...")
            await self._setup_interrupt_monitoring(interrupt_callback)

        # Initialize playback buffer
        playback_buffer = PlaybackBuffer(sample_rate=22050, buffer_duration=60)
        
        # Get audio device configuration
        import yaml
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            output_device = config.get('output_device', None)
        except:
            output_device = None

        try:
            print("🎤 Starting 3-component streaming architecture...")
            
            # Start continuous player immediately
            print("🎵 Starting Continuous Player")
            playback_buffer.start_playback(
                device=output_device,
                channels=1,
                blocksize=512
            )
            
            # Use orchestrator to coordinate streaming
            success = await self._streaming_orchestrator.coordinate_streaming(
                text_stream=text_stream,
                playback_buffer=playback_buffer,
                result=result,
                interruptible=interruptible,
                speaker=speaker,
                language=language,
                speed=speed,
                emotion=emotion
            )
            
            if success and not result['interrupted']:
                result['completed'] = True
                print("✅ All audio played successfully")

        except Exception as e:
            print(f"\n⚠️ Error in speak_stream_async: {e}")
            result['interrupted'] = True
            
        finally:
            # Cleanup
            print("🧹 Cleaning up streaming components...")
            playback_buffer.stop_playback()
            
            if interruptible:
                print("🧹 Cleaning up interrupt monitoring...")
                await self._cleanup_interrupt_monitoring()

        print(f"🎯 TTS speak_stream_async returning - Completed: {result['completed']}, Interrupted: {result['interrupted']}")
        return result

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
        Standard TTS synthesis. Convert text to speech and optionally play it.
        
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
        
        # Ensure TTS is ready
        if not self.ready and self._load_thread.is_alive():
            print("🕐 Waiting for TTS to load...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_thread.join)
            
        if not self.ready:
            print("❌ TTS not ready")
            return result
            
        chunks = self._split_text(text)
        if not chunks:
            result['completed'] = True
            print("🎯 TTS speak_async completed - No chunks to process")
            return result

        print(f"🎯 TTS processing {len(chunks)} chunks")

        # Setup interrupt monitoring if requested
        if interruptible:
            print("🎯 Setting up interrupt monitoring...")
            await self._setup_interrupt_monitoring(interrupt_callback)

        try:
            # Generate all chunks in parallel
            loop = asyncio.get_event_loop()
            futures: List[asyncio.Future] = []
            for i, chunk_text in enumerate(chunks):
                future = loop.run_in_executor(
                    self.executor,
                    self._generate_chunk, chunk_text, i, speaker, language, speed, emotion
                )
                futures.append(future)

            print(f"🎯 Generating all {len(futures)} chunks in parallel...")
            chunk_audios = await asyncio.gather(*futures)
            
            # Concatenate all chunks into one continuous audio stream
            if chunk_audios:
                combined_audio_parts = []
                for i, audio in enumerate(chunk_audios):
                    if i == 0:
                        # First chunk: apply fade-in only
                        combined_audio_parts.append(self.audio_manager.apply_fade_in(audio, fade_ms=50))
                    elif i == len(chunk_audios) - 1:
                        # Last chunk: apply fade-out only  
                        combined_audio_parts.append(self.audio_manager.apply_fade_out(audio, fade_ms=100))
                    else:
                        # Middle chunks: no additional fading
                        combined_audio_parts.append(audio)
                
                combined_audio = np.concatenate(combined_audio_parts)
                result['audio'] = combined_audio
                                
                # Play as one continuous audio stream
                if play_audio:
                    print(f"🎯 Playing continuous audio stream...")
                    completed = await self.audio_manager.play_audio_async(
                        combined_audio, 
                        blocking=True, 
                        interruptible=interruptible
                    )
                    
                    if not completed:
                        print("🔄 TTS interrupted during continuous playback")
                        result['interrupted'] = True
                    else:
                        print("✅ Continuous audio playback completed successfully")
                        result['completed'] = True
                else:
                    result['completed'] = True
            else:
                print("⚠️ No audio chunks generated")
                result['completed'] = True

        except Exception as e:
            print(f"⚠️ Error in speak_async: {e}")
            result['interrupted'] = True
            
        finally:
            # Cleanup interrupt monitoring
            if interruptible:
                print("🎯 Cleaning up interrupt monitoring...")
                await self._cleanup_interrupt_monitoring()

        print(f"🎯 TTS speak_async returning - Completed: {result['completed']}, Interrupted: {result['interrupted']}")
        return result

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        if self._load_thread.is_alive():
            self._load_thread.join()
        self.audio_manager.close()