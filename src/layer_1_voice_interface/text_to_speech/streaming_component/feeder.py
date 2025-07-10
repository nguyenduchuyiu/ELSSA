import asyncio
from .playback_buffer import PlaybackBuffer

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
        
        # Track total audio size instead of keeping all chunks
        total_samples = 0
        chunk_count = 0
        
        try:
            while True:
                try:
                    audio_chunk = await asyncio.wait_for(synthesis_queue.get(), timeout=5.0)
                    
                    if audio_chunk is None:  # End signal
                        print("🚚 No more chunks to feed")
                        break
                        
                    # Apply fade transitions for smooth playback
                    if chunk_count == 0:
                        # First chunk - apply fade-in
                        processed_chunk = self.tts.audio_manager.apply_fade_in(audio_chunk, fade_ms=25)
                    else:
                        # Subsequent chunks - no additional processing needed
                        processed_chunk = audio_chunk
                    
                    chunk_count += 1
                    total_samples += len(processed_chunk)
                    
                    # Feed chunk into playback buffer
                    success = playback_buffer.write_chunk(processed_chunk)
                    if success:
                        duration = len(processed_chunk) / playback_buffer.sample_rate
                        print(f"🚚 Fed chunk {chunk_count} to buffer ({duration:.2f}s)")
                        
                        # Clear processed_chunk immediately to free memory
                        del processed_chunk
                        
                    else:
                        print(f"⚠️ Buffer full, feeder waiting...")
                        # Retry feeding after brief delay
                        await asyncio.sleep(0.1)
                    
                    # Check for interruption using interrupt_manager
                    if hasattr(self.tts, 'interrupt_manager') and self.tts.interrupt_manager.is_interrupted():
                        result['interrupted'] = True
                        break
                        
                except asyncio.TimeoutError:
                    if hasattr(self.tts, 'interrupt_manager') and self.tts.interrupt_manager.is_interrupted():
                        result['interrupted'] = True
                        break
                    # Check if producer has finished - if so, there might be a None signal waiting
                    # Don't set interrupted flag on timeout unless explicitly interrupted
                    continue
                    
        except Exception as e:
            print(f"⚠️ Error in feeder task: {e}")
            result['interrupted'] = True
        finally:
            # Mark buffer as finished
            playback_buffer.mark_finished()
            
            # Create lightweight result summary instead of full audio
            result['total_samples'] = total_samples
            result['chunk_count'] = chunk_count
            result['estimated_duration'] = total_samples / playback_buffer.sample_rate if total_samples > 0 else 0
            
            print(f"🚚 Feeder Task completed - {chunk_count} chunks, {total_samples} samples")
            print(f"🚚 Estimated duration: {result['estimated_duration']:.2f}s")
            print(f"🚚 Final interrupted status: {result['interrupted']}")