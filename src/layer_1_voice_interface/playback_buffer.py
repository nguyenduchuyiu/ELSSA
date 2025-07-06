"""
Continuous playback buffer for seamless audio streaming.
Provides thread-safe circular buffer with real-time audio playback.
"""

import numpy as np
import threading
import sounddevice as sd
from typing import Optional
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class PlaybackBuffer:
    """
    Continuous playback buffer for seamless audio streaming.
    Uses a circular buffer with thread-safe operations to eliminate gaps between chunks.
    """
    
    def __init__(self, sample_rate: int = 22050, buffer_duration: float = 5):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)  # buffer duration in samples
        
        # Circular buffer for audio data
        self._buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self._write_pos = 0
        self._read_pos = 0
        self._available_samples = 0
        self._finished = False
        self._interrupted = False
        
        # Thread synchronization
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        
        # Stream control
        self._stream = None
        self._is_playing = False
        
    def write_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Write audio chunk to buffer.
        Returns True if successful, False if buffer is full.
        """
        with self._condition:
            chunk_size = len(audio_chunk)
            
            # Check if we have enough space
            free_space = self.buffer_size - self._available_samples
            if chunk_size > free_space:
                # Buffer overflow - wait for space to become available
                print(f"⚠️ Buffer overflow: need {chunk_size}, have {free_space} - waiting for space...")
                
                # Wait for buffer to have enough space (with timeout)
                def has_space():
                    return (self.buffer_size - self._available_samples) >= chunk_size or self._interrupted
                
                if not self._condition.wait_for(has_space, timeout=config.get('tts_timeout_buffer_wait', 10)):
                    print(f"❌ Timeout waiting for buffer space")
                    return False
                
                # Check again after waiting
                if self._interrupted:
                    return False
                    
                free_space = self.buffer_size - self._available_samples
                if chunk_size > free_space:
                    print(f"❌ Still not enough space after waiting: need {chunk_size}, have {free_space}")
                    return False
            
            # Write chunk to circular buffer
            end_pos = self._write_pos + chunk_size
            if end_pos <= self.buffer_size:
                # Simple case: no wraparound
                self._buffer[self._write_pos:end_pos] = audio_chunk
            else:
                # Wraparound case
                first_part = self.buffer_size - self._write_pos
                self._buffer[self._write_pos:] = audio_chunk[:first_part]
                self._buffer[:end_pos - self.buffer_size] = audio_chunk[first_part:]
            
            self._write_pos = end_pos % self.buffer_size
            self._available_samples += chunk_size
            
            # Notify readers that data is available
            self._condition.notify_all()
            return True
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """
        Read samples from buffer for audio callback.
        Returns zeros if not enough data available or if interrupted.
        """
        with self._lock:
            # If interrupted, return silence immediately
            if self._interrupted:
                return np.zeros(num_samples, dtype=np.float32)
                
            if self._available_samples < num_samples:
                if self._finished and self._available_samples > 0:
                    # Return remaining samples + zeros
                    remaining = self._available_samples
                    result = np.zeros(num_samples, dtype=np.float32)
                    
                    end_pos = self._read_pos + remaining
                    if end_pos <= self.buffer_size:
                        result[:remaining] = self._buffer[self._read_pos:end_pos]
                    else:
                        first_part = self.buffer_size - self._read_pos
                        result[:first_part] = self._buffer[self._read_pos:]
                        result[first_part:remaining] = self._buffer[:end_pos - self.buffer_size]
                    
                    self._read_pos = end_pos % self.buffer_size
                    self._available_samples = 0
                    
                    # Notify writers that space is available
                    self._condition.notify_all()
                    return result
                else:
                    # Not enough data and not finished - return silence
                    return np.zeros(num_samples, dtype=np.float32)
            
            # Read samples from circular buffer
            result = np.zeros(num_samples, dtype=np.float32)
            end_pos = self._read_pos + num_samples
            
            if end_pos <= self.buffer_size:
                # Simple case: no wraparound
                result = self._buffer[self._read_pos:end_pos].copy()
            else:
                # Wraparound case
                first_part = self.buffer_size - self._read_pos
                result[:first_part] = self._buffer[self._read_pos:]
                result[first_part:] = self._buffer[:end_pos - self.buffer_size]
            
            self._read_pos = end_pos % self.buffer_size
            self._available_samples -= num_samples
            
            # Notify writers that space is available
            self._condition.notify_all()
            return result
    
    def start_playback(self, device: Optional[int] = None, channels: int = 1, blocksize: int = 1024):
        """Start continuous audio playback stream."""
        if self._is_playing:
            return
            
        def audio_callback(outdata, frames, time, status):
            if status:
                print(f"⚠️ Audio callback status: {status}")
            
            # Read samples from buffer
            audio_data = self.read_samples(frames)
            outdata[:, 0] = audio_data
            
            # Fill additional channels if needed
            for i in range(1, channels):
                outdata[:, i] = audio_data
        
        try:
            self._stream = sd.OutputStream(
                device=device,
                samplerate=self.sample_rate,
                channels=channels,
                dtype='float32',
                blocksize=blocksize,
                callback=audio_callback
            )
            self._stream.start()
            self._is_playing = True
            print(f"🎵 Started continuous playback stream (blocksize: {blocksize})")
        except Exception as e:
            print(f"⚠️ Error starting playback stream: {e}")
            self._is_playing = False
    
    def stop_playback(self):
        """Stop continuous audio playback stream."""
        if self._stream and self._is_playing:
            try:
                self._stream.stop()
                self._stream.close()
                self._is_playing = False
                print("🎵 Stopped continuous playback stream")
            except Exception as e:
                print(f"⚠️ Error stopping playback stream: {e}")
        self._stream = None
    
    def set_interrupted(self):
        """Set interrupt flag to stop audio playback immediately."""
        with self._lock:
            self._interrupted = True
            self._condition.notify_all()
            print("⚡ Playback buffer interrupted")
    
    def mark_finished(self):
        """Mark that no more audio will be written to buffer."""
        with self._condition:
            self._finished = True
            self._condition.notify_all()
    
    def get_buffer_status(self) -> dict:
        """Get current buffer status for monitoring."""
        with self._lock:
            return {
                'available_samples': self._available_samples,
                'buffer_fill_ratio': self._available_samples / self.buffer_size,
                'is_playing': self._is_playing,
                'finished': self._finished,
                'interrupted': self._interrupted
            }
    
    def cleanup(self):
        """Enhanced cleanup to free large audio buffer and reset state"""
        print("🧹 PlaybackBuffer cleanup starting...")
        
        # Stop playback stream first
        self.stop_playback()
        
        # Clear the large circular buffer
        with self._lock:
            if hasattr(self, '_buffer') and self._buffer is not None:
                buffer_size_mb = self._buffer.nbytes / (1024 * 1024)
                print(f"🧹 Freeing circular buffer: {buffer_size_mb:.2f}MB")
                del self._buffer
                self._buffer = None
            
            # Reset buffer state
            self._write_pos = 0
            self._read_pos = 0
            self._available_samples = 0
            self._finished = False
            self._interrupted = False
            
            # Notify any waiting threads
            self._condition.notify_all()
        
        print("✅ PlaybackBuffer cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if hasattr(self, '_buffer') and self._buffer is not None:
                self.cleanup()
        except:
            pass 