import asyncio
from .playback_buffer import PlaybackBuffer

class StreamingMonitor:
    """
    Component 3: Monitor Task - Handle interruptions and monitoring
    """
    
    def __init__(self, tts_instance):
        self.tts = tts_instance
        
    async def monitor_buffer_completion(
        self,
        playback_buffer: PlaybackBuffer,
        result: dict
    ):
        """Monitor buffer completion with periodic status updates"""
        
        timeout = 30.0
        check_interval = 2.0
        elapsed = 0.0
        
        while elapsed < timeout:
            status = playback_buffer.get_buffer_status()
            # print(f"📊 Buffer status: {status['available_samples']} samples, "
            #       f"{status['buffer_fill_ratio']:.1%} full, playing: {status['is_playing']}")
            
            # Check for interruption using interrupt_manager
            if hasattr(self.tts, 'interrupt_manager') and self.tts.interrupt_manager.is_interrupted():
                result['interrupted'] = True
                return False
            
            # Check if buffer is empty
            if status['available_samples'] == 0:
                print("✅ Buffer empty - playback completed")
                result['completed'] = True
                return True
            
            # Check if playback stream is still active
            if not status['is_playing']:
                print("⚠️ Playback stream not active")
                return False
            
            # Check if buffer is finished but still has samples
            # if status['finished'] and status['available_samples'] > 0:
                # print(f"📊 Buffer finished but still has {status['available_samples']} samples - waiting for playback...")
            
            await asyncio.sleep(check_interval)
            elapsed += check_interval
        
        print(f"⏰ Timeout after {timeout}s waiting for buffer to empty")
        return False