import threading
from typing import Optional, Callable

from ...wake_word_handler import InterruptWakeWordHandler


class TTSInterruptManager:
    """
    Manages interrupt handling during TTS playback.
    Handles wake word detection and callback execution.
    """
    
    def __init__(self):
        self._interrupt_handler: Optional[InterruptWakeWordHandler] = None
        self._interrupt_callback: Optional[Callable[[], None]] = None
        self._is_interrupted = threading.Event()
        
    async def setup_monitoring(self, interrupt_callback: Optional[Callable[[], None]]) -> None:
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
            # self._interrupt_handler.set_interrupt_callback(self._on_interrupt_detected)
            self._interrupt_handler.set_interrupt_callback(lambda: None) # FIXME: for now, we don't need to detect interrupt
            self._interrupt_handler.start_interrupt_monitoring()
            
        except Exception as e:
            print(f"⚠️ Error setting up interrupt monitoring: {e}")
    
    async def cleanup_monitoring(self) -> None:
        """Cleanup interrupt monitoring"""
        try:
            if self._interrupt_handler and self._interrupt_handler.is_monitoring():
                self._interrupt_handler.stop_interrupt_monitoring()
            
            self._interrupt_handler = None
            self._interrupt_callback = None
            self._is_interrupted.clear()
            
        except Exception as e:
            print(f"⚠️ Error cleaning up interrupt monitoring: {e}")
            self._interrupt_handler = None
            self._interrupt_callback = None
    
    def _on_interrupt_detected(self) -> None:
        """Called when wake word detected during TTS"""
        self._is_interrupted.set()
        
        # Call user-provided interrupt callback
        if self._interrupt_callback:
            try:
                self._interrupt_callback()
            except Exception as e:
                print(f"⚠️ Error in user interrupt callback: {e}")
        
        print("🚨 INTERRUPT DETECTED - Stopping TTS")
        
    def is_interrupted(self) -> bool:
        """Check if interrupt was triggered"""
        return self._is_interrupted.is_set()
        
    def clear_interrupt(self) -> None:
        """Clear interrupt flag"""
        self._is_interrupted.clear() 