from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    Provides common functionality like streaming, chunking, and parallel processing.
    Subclasses only need to implement model loading and core audio generation.
    """
    
    def __init__(self, device: Optional[str] = None, max_len: int = 200, fade_ms: int = 10):
        self.device = device
        self.ready = False
        self.max_len = max_len
        self.fade_ms = fade_ms
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Will be initialized in subclasses
        self.audio_manager = None
        self.interrupt_manager = None
        self.streaming_orchestrator = None
        # Removed self.initialize() call - it's handled properly in async methods
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS engine asynchronously"""
        pass
    
    @abstractmethod
    def _generate_chunk_core(self, text: str) -> np.ndarray:
        """
        Core audio generation method that subclasses must implement.
        This is the only method subclasses need to override for basic functionality.
        
        Args:
            text: Text chunk to synthesize
            
        Returns:
            Raw audio data as numpy array
        """
        pass
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS processing"""
        sentences = re.split(r'(?<=[.!?]) +|\n+', text)
        # remove special characters like !, ...
        sentences = [re.sub(r'[^\w\s?!]', '', sent) for sent in sentences]
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
    
    def _generate_chunk(self, text: str) -> np.ndarray:
        """Generate and post-process audio chunk"""
        if not self.ready:
            raise RuntimeError("TTS model not loaded yet.")
            
        # Get raw audio from subclass implementation
        audio = self._generate_chunk_core(text)
        
        # Apply fade if audio manager is available
        if self.audio_manager and hasattr(self.audio_manager, 'apply_fade'):
            audio = self.audio_manager.apply_fade(audio, self.fade_ms)
        
        return audio
    
    async def _generate_chunk_async(self, text: str) -> np.ndarray:
        """Async wrapper for audio chunk generation - runs directly to avoid race conditions"""
        # Run directly instead of using executor to avoid CoquiTTS cache race conditions
        return self._generate_chunk(text)
    
    async def _generate_all_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """Generate all audio chunks sequentially to avoid CoquiTTS cache race conditions"""
        
        chunk_audios = []
        for i, chunk_text in enumerate(chunks):
            try:
                # Generate chunk synchronously to avoid race conditions
                # print(f"🔊 Processing chunk {i+1}/{len(chunks)}: '{chunk_text}...'")
                audio = self._generate_chunk(chunk_text)
                chunk_audios.append(audio)
                
                # Small delay to be nice to the system
                await asyncio.sleep(0.001)
                
            except Exception as e:
                print(f"⚠️ Error generating chunk {i+1}: {e}")
                # Continue with other chunks instead of failing completely
                continue
        
        # print(f"✅ Generated {len(chunk_audios)} audio chunks successfully")
        return chunk_audios
    
    def _combine_audio_chunks(self, chunk_audios: List[np.ndarray]) -> np.ndarray:
        """Combine audio chunks into continuous stream with proper fading"""
        if not chunk_audios:
            return np.array([], dtype=np.float32)
            
        combined_audio_parts = []
        for i, audio in enumerate(chunk_audios):
            if i == 0 and self.audio_manager and hasattr(self.audio_manager, 'apply_fade_in'):
                # First chunk: apply fade-in only
                combined_audio_parts.append(self.audio_manager.apply_fade_in(audio, fade_ms=50))
            elif i == len(chunk_audios) - 1 and self.audio_manager and hasattr(self.audio_manager, 'apply_fade_out'):
                # Last chunk: apply fade-out only  
                combined_audio_parts.append(self.audio_manager.apply_fade_out(audio, fade_ms=100))
            else:
                # Middle chunks: no additional fading
                combined_audio_parts.append(audio)
        
        return np.concatenate(combined_audio_parts)
    
    def generate_audio(self, text: str) -> np.ndarray:
        """
        Generate audio from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Raw audio data as numpy array
        """
        if not self.ready:
            raise RuntimeError("TTS model not loaded yet.")
            
        # chunks = self._split_text(text)
        print(f"Received text: {text}")
        chunks = [text]
        if not chunks:
            return np.array([], dtype=np.float32)
            
        # Generate all chunks
        chunk_audios = []
        for chunk_text in chunks:
            audio = self._generate_chunk(chunk_text)
            chunk_audios.append(audio)
            
        # Combine into continuous audio
        return self._combine_audio_chunks(chunk_audios)
    
    async def speak_async(
        self,
        text: str,
        play_audio: bool = True,
        interruptible: bool = False,
        interrupt_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """
        Standard TTS synthesis. Convert text to speech and optionally play it.
        """
        result = {
            'completed': False,
            'interrupted': False,
            'audio': np.array([], dtype=np.float32)
        }
        
        try:
            # Ensure TTS is ready
            await self.initialize()
            if not self.ready:
                print("❌ TTS not ready")
                return result
                
            chunks = self._split_text(text)
            if not chunks:
                result['completed'] = True
                return result


            # Setup interrupt monitoring if requested and available
            if interruptible and self.interrupt_manager:
                await self.interrupt_manager.setup_monitoring(interrupt_callback)

            try:
                # Generate all chunks in parallel
                chunk_audios = await self._generate_all_chunks(chunks)
                
                # Combine all chunks into one continuous audio stream
                if chunk_audios:
                    combined_audio = self._combine_audio_chunks(chunk_audios)
                    result['audio'] = combined_audio
                                    
                    # Play as one continuous audio stream
                    if play_audio and self.audio_manager:
                        completed = await self.audio_manager.play_audio_async(
                            combined_audio, 
                            blocking=True, 
                            interruptible=interruptible
                        )
                        
                        if not completed:
                            print("🔄 TTS interrupted during continuous playback")
                            result['interrupted'] = True
                        else:
                            # print("✅ Continuous audio playback completed successfully")
                            result['completed'] = True
                    else:
                        result['completed'] = True
                else:
                    print("⚠️ No audio chunks generated")
                    result['completed'] = True

            finally:
                # Cleanup interrupt monitoring
                if interruptible and self.interrupt_manager:
                    await self.interrupt_manager.cleanup_monitoring()
                    
        except Exception as e:
            print(f"⚠️ Error in speak_async: {e}")
            result['interrupted'] = True
            
        return result
    
    async def speak_stream_async(
        self,
        text_stream,
        interruptible: bool = False,
        interrupt_callback: Optional[Callable[[], None]] = None
    ) -> Dict[str, Any]:
        """
        Stream text chunks to TTS with continuous playback buffer.
        Uses streaming orchestrator if available, otherwise falls back to sequential processing.
        """
        result = {
            'completed': False,
            'interrupted': False,
            'text': '',
            'audio': np.array([], dtype=np.float32)
        }
        
        # Ensure TTS is ready
        await self.initialize()
        if not self.ready:
            print("❌ TTS not ready")
            return result

        # Use streaming orchestrator if available
        if self.streaming_orchestrator:
            return await self._stream_with_orchestrator(text_stream, interruptible, interrupt_callback)
        else:
            # Fallback to sequential processing
            return await self._stream_sequential(text_stream, interruptible, interrupt_callback)
    
    async def _stream_with_orchestrator(self, text_stream, interruptible: bool, interrupt_callback: Optional[Callable[[], None]]) -> Dict[str, Any]:
        """Stream using orchestrator for advanced streaming"""
        from ..streaming_component.playback_buffer import PlaybackBuffer
        
        result = {
            'completed': False,
            'interrupted': False,
            'text': '',
            'audio': np.array([], dtype=np.float32)
        }

        # Setup interrupt monitoring if requested
        if interruptible and self.interrupt_manager:
            await self.interrupt_manager.setup_monitoring(interrupt_callback)

        # Initialize playback buffer
        playback_buffer = PlaybackBuffer(
            sample_rate=22050, 
            buffer_duration=config.get('tts_buffer_duration', 5)
        )
        
        # Get audio device configuration
        try:
            output_device = config.get('output_device', None)
        except:
            output_device = None

        try:            
            # Start continuous player immediately
            playback_buffer.start_playback(
                device=output_device,
                channels=1,
                blocksize=512
            )
            
            # Verify playback started successfully
            if not playback_buffer._is_playing:
                print("❌ Failed to start playback stream")
                result['interrupted'] = True
                return result
            
            # Use orchestrator to coordinate streaming
            success = await self.streaming_orchestrator.coordinate_streaming(
                text_stream=text_stream,
                playback_buffer=playback_buffer,
                result=result
            )
                        
            if success and not result['interrupted']:
                result['completed'] = True
                print("✅ All audio played successfully")
            else:
                print(f"⚠️ Streaming incomplete - Success: {success}, Interrupted: {result['interrupted']}")

        except Exception as e:
            print(f"\n⚠️ Error in speak_stream_async: {e}")
            import traceback
            traceback.print_exc()
            result['interrupted'] = True
            
        finally:
            # Enhanced PlaybackBuffer cleanup
            try:
                playback_buffer.cleanup()
            except Exception as e:
                print(f"⚠️ Error cleaning playback buffer: {e}")
            
            if interruptible and self.interrupt_manager:
                await self.interrupt_manager.cleanup_monitoring()

        return result
    
    async def _stream_sequential(self, text_stream, interruptible: bool, interrupt_callback: Optional[Callable[[], None]]) -> Dict[str, Any]:
        """Fallback sequential streaming for basic TTS engines"""
        result = {
            'completed': False,
            'interrupted': False,
            'text': '',
            'audio': np.array([], dtype=np.float32)
        }
        
        try:
            combined_text = ""
            for chunk in text_stream:
                combined_text += str(chunk) + " "
                result['text'] = combined_text.strip()
                
            # Use regular speak_async for sequential processing
            speak_result = await self.speak_async(
                combined_text, 
                play_audio=True, 
                interruptible=interruptible, 
                interrupt_callback=interrupt_callback
            )
            
            result.update(speak_result)
            
        except Exception as e:
            print(f"⚠️ Error in sequential streaming: {e}")
            result['interrupted'] = True
            
        return result
    
    def close(self) -> None:
        """Clean up common resources"""
        try:
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Close audio manager if available
            if self.audio_manager:
                self.audio_manager.close()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            self.ready = False
            print("✅ Base TTS cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in base TTS cleanup: {e}")
    
    def is_ready(self) -> bool:
        """Check if TTS engine is ready to use"""
        return self.ready 