"""
Streaming components for 3-component TTS architecture.
Includes Producer, Feeder, and Monitor tasks for seamless audio streaming.
"""

import asyncio
import numpy as np
from .playback_buffer import PlaybackBuffer
from .producer import StreamingProducer
from .feeder import StreamingFeeder
from .monitor import StreamingMonitor

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
        result: dict
    ) -> bool:
        """Coordinate all streaming components"""
        synthesis_queue = asyncio.Queue()
        
        # Create tasks
        producer_task = asyncio.create_task(
            self.producer.produce_audio_chunks(
                text_stream, synthesis_queue, result
            )
        )
        
        feeder_task = asyncio.create_task(
            self.feeder.feed_playback_buffer(
                synthesis_queue, playback_buffer, result
            )
        )
        
        tasks = [producer_task, feeder_task]
        
        # Wait for producer and feeder to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Monitor buffer completion if not interrupted
        if not result['interrupted']:
            return await self.monitor.monitor_buffer_completion(playback_buffer, result)
        
        return False 