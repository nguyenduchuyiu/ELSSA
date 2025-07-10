import threading
from abc import ABC, abstractmethod
from typing import Optional


class BaseTTSModelManager(ABC):
    """
    Abstract base class for TTS model managers.
    Handles common functionality like background loading, state management, and cleanup.
    """
    
    def __init__(self):
        # Loading state
        self.ready = False
        self._load_thread: Optional[threading.Thread] = None
        
    def start_loading(self) -> None:
        """Start loading models in background thread"""
        if self._load_thread is None or not self._load_thread.is_alive():
            self._load_thread = threading.Thread(target=self._load_models_safely)
            self._load_thread.start()
            
    def wait_for_loading(self) -> None:
        """Wait for model loading to complete"""
        if self._load_thread and self._load_thread.is_alive():
            self._load_thread.join()
            
    def is_ready(self) -> bool:
        """Check if models are loaded and ready"""
        return self.ready
        
    def _load_models_safely(self) -> None:
        """Safe wrapper for model loading with error handling"""
        try:
            self._load_models()
            self.ready = True
            print(f"✅ {self._get_engine_name()} models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading {self._get_engine_name()} models: {e}")
            self.ready = False
             
    @abstractmethod
    def _load_models(self) -> None:
        """Load engine-specific models. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _get_engine_name(self) -> str:
        """Get engine name for logging. Must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _cleanup_models(self) -> None:
        """Clean up engine-specific model resources. Must be implemented by subclasses."""
        pass
        
    def cleanup(self) -> None:
        """Clean up model resources"""
        try:
            # Wait for loading to complete
            if self._load_thread and self._load_thread.is_alive():
                self._load_thread.join()
                
            # Clean up engine-specific models
            self._cleanup_models()
            
            # Common GPU memory cleanup
            self._cleanup_gpu_memory()
                    
            self.ready = False
            print(f"✅ {self._get_engine_name()} models cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in {self._get_engine_name()} model cleanup: {e}")
            
    def _cleanup_gpu_memory(self) -> None:
        """Common GPU memory cleanup logic"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"🎮 {self._get_engine_name()} GPU memory cleared")
        except ImportError:
            pass 