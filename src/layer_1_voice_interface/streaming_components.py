"""
Streaming components for 3-component TTS architecture.
Includes Producer, Feeder, and Monitor tasks for seamless audio streaming.
"""

import asyncio
import numpy as np
from typing import Callable, Optional, List, Tuple
from .playback_buffer import PlaybackBuffer


class StreamingProducer:
    """
    Component 1: Producer Task - Generate audio chunks from text stream
    """
    
    def __init__(self, tts_instance, sample_rate: int = 22050):
        self.tts = tts_instance
        self.sample_rate = sample_rate
        
    async def produce_audio_chunks(
        self,
        text_stream,
        synthesis_queue: asyncio.Queue,
        result: dict,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell'
    ):
        """Generate audio chunks from text stream and put them in synthesis queue"""
        print("🏭 Producer Task started")
        
        sentence_buffer = ""
        sentence_endings = ['.', '!', '?', '\n']
        chunk_count = 0
        generation_tasks = []
        
        try:
            for text_chunk in text_stream:
                if self.tts._is_interrupted.is_set():
                    print("🔄 Producer interrupted")
                    result['interrupted'] = True
                    break
                    
                result['text'] += text_chunk
                sentence_buffer += text_chunk
                print(text_chunk, end="", flush=True)
                
                # Check if we have a complete sentence or enough text
                has_sentence_ending = any(ending in sentence_buffer for ending in sentence_endings)
                has_enough_text = len(sentence_buffer.strip()) > 25
                
                if has_sentence_ending or has_enough_text:
                    sentence = sentence_buffer.strip()
                    if sentence:                        
                        # Create generation task
                        task = asyncio.create_task(
                            self.tts._generate_audio_chunk_async(
                                sentence, chunk_count, speaker, language, speed, emotion
                            )
                        )
                        generation_tasks.append((chunk_count, task))
                        chunk_count += 1
                        
                    sentence_buffer = ""
                    
                await asyncio.sleep(0.001)  # Minimal yield
            
            # Process remaining text
            if sentence_buffer.strip() and not self.tts._is_interrupted.is_set():
                sentence = sentence_buffer.strip()
                print(f"\n🏭 Queuing final audio generation: '{sentence[:50]}...'")
                
                task = asyncio.create_task(
                    self.tts._generate_audio_chunk_async(
                        sentence, chunk_count, speaker, language, speed, emotion
                    )
                )
                generation_tasks.append((chunk_count, task))
            
            # Wait for all generation tasks and put results in synthesis queue
            for chunk_id, task in generation_tasks:
                try:
                    audio_chunk = await task
                    await synthesis_queue.put(audio_chunk)
                    print(f"🏭 Audio chunk {chunk_id} ready")
                except Exception as e:
                    print(f"⚠️ Error generating audio chunk {chunk_id}: {e}")
            
        except Exception as e:
            print(f"⚠️ Error in producer task: {e}")
            result['interrupted'] = True
        finally:
            # Signal end of production
            await synthesis_queue.put(None)
            print("🏭 Producer Task completed")


class StreamingFeeder:
    """
    Component 2: Feeder Task - Feed chunks into playback buffer
    """
    
    def __init__(self, tts_instance):
        self.tts = tts_instance
        
    async def feed_playback_buffer(
        self,
        synthesis_queue: asyncio.Queue,
        playback_buffer: PlaybackBuffer,
        result: dict
    ):
        """Feed audio chunks from synthesis queue into playback buffer"""
        print("🚚 Feeder Task started")
        all_chunks = []
        
        try:
            while True:
                try:
                    audio_chunk = await asyncio.wait_for(synthesis_queue.get(), timeout=5.0)
                    
                    if audio_chunk is None:  # End signal
                        print("🚚 No more chunks to feed")
                        break
                        
                    # Apply fade transitions for smooth playback
                    if len(all_chunks) == 0:
                        # First chunk - apply fade-in
                        processed_chunk = self.tts.audio_manager.apply_fade_in(audio_chunk, fade_ms=25)
                    else:
                        # Subsequent chunks - apply gentle fade-in to prevent clicks
                        processed_chunk = self.tts._apply_smooth_transition(audio_chunk)
                    
                    all_chunks.append(processed_chunk)
                    
                    # Feed chunk into playback buffer
                    success = playback_buffer.write_chunk(processed_chunk)
                    if success:
                        duration = len(processed_chunk) / playback_buffer.sample_rate
                        print(f"🚚 Fed chunk {len(all_chunks)} to buffer ({duration:.2f}s)")
                    else:
                        print(f"⚠️ Buffer full, feeder waiting...")
                        # Retry feeding after brief delay
                        await asyncio.sleep(0.1)
                    
                    # Check for interruption
                    if self.tts._is_interrupted.is_set():
                        print("🔄 Feeder interrupted")
                        result['interrupted'] = True
                        break
                        
                except asyncio.TimeoutError:
                    print("⏰ Feeder timeout - checking for interruption")
                    if self.tts._is_interrupted.is_set():
                        result['interrupted'] = True
                        break
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error in feeder task: {e}")
            result['interrupted'] = True
        finally:
            # Mark buffer as finished and combine chunks for result
            playback_buffer.mark_finished()
            if all_chunks:
                result['audio'] = np.concatenate(all_chunks)
            
            print("🚚 Feeder Task completed")


class StreamingMonitor:
    """
    Component 3: Monitor Task - Handle interruptions and monitoring
    """
    
    def __init__(self, tts_instance):
        self.tts = tts_instance
        
    async def monitor_interrupts(
        self,
        playback_buffer: PlaybackBuffer,
        result: dict
    ):
        """Monitor for interruptions and stop playback buffer immediately"""
        while not result['interrupted'] and not result['completed']:
            if self.tts._is_interrupted.is_set():
                print("⚡ Interrupt detected - stopping playback buffer")
                playback_buffer.set_interrupted()
                result['interrupted'] = True
                break
            await asyncio.sleep(0.01)  # Check every 10ms for responsive interruption
    
    async def monitor_buffer_completion(
        self,
        playback_buffer: PlaybackBuffer,
        result: dict
    ):
        """Monitor buffer completion with periodic status updates"""
        print("⏰ Waiting for playback buffer to empty...")
        
        timeout = 30.0
        check_interval = 2.0
        elapsed = 0.0
        
        while elapsed < timeout:
            status = playback_buffer.get_buffer_status()
            print(f"📊 Buffer status: {status['available_samples']} samples, "
                  f"{status['buffer_fill_ratio']:.1%} full, playing: {status['is_playing']}")
            
            if status['available_samples'] == 0:
                print("✅ Buffer empty - playback completed")
                result['completed'] = True
                return True
            
            if not status['is_playing']:
                print("⚠️ Playback stream not active")
                return False
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        print(f"⏰ Timeout after {timeout}s waiting for buffer to empty")
        return False


class StreamingOrchestrator:
    """
    Orchestrator for managing all streaming components
    """
    
    def __init__(self, tts_instance):
        self.tts = tts_instance
        self.producer = StreamingProducer(tts_instance)
        self.feeder = StreamingFeeder(tts_instance)
        self.monitor = StreamingMonitor(tts_instance)
    
    async def coordinate_streaming(
        self,
        text_stream,
        playback_buffer: PlaybackBuffer,
        result: dict,
        interruptible: bool = False,
        **tts_params
    ) -> bool:
        """Coordinate all streaming components"""
        synthesis_queue = asyncio.Queue()
        
        # Create tasks
        producer_task = asyncio.create_task(
            self.producer.produce_audio_chunks(
                text_stream, synthesis_queue, result, **tts_params
            )
        )
        
        feeder_task = asyncio.create_task(
            self.feeder.feed_playback_buffer(
                synthesis_queue, playback_buffer, result
            )
        )
        
        tasks = [producer_task, feeder_task]
        
        # Add interrupt monitoring if requested
        if interruptible:
            interrupt_task = asyncio.create_task(
                self.monitor.monitor_interrupts(playback_buffer, result)
            )
            tasks.append(interrupt_task)
        
        # Wait for producer and feeder to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Monitor buffer completion if not interrupted
        if not result['interrupted']:
            return await self.monitor.monitor_buffer_completion(playback_buffer, result)
        
        return False 