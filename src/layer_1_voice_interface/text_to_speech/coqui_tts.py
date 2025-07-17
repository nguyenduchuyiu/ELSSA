import numpy as np
import asyncio
import yaml

from .tts_component.base_tts import BaseTTS
from .tts_component.base_model_manager import BaseTTSModelManager
from .tts_component.interrupt_manager import TTSInterruptManager
from ..audio_manager import AudioManager
from .streaming_component.orchestrator import StreamingOrchestrator

from CoquiTTS.api import TTS

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class CoquiModelManager(BaseTTSModelManager):
    """
    Manages loading and initialization of CoquiTTS models.
    Inherits common functionality from BaseTTSModelManager.
    """
    
    def __init__(self):
        super().__init__()
        
        # Model configuration from config
        self.model_path = config.get("coqui_tts_model_path")
        self.config_path = config.get("coqui_tts_model_config_path")
        self.vocoder_path = config.get("coqui_tts_vocoder_model_path")
        self.vocoder_config_path = config.get("coqui_tts_vocoder_model_config_path")
        self.tts_gpu = config.get("coqui_tts_gpu", False)
        
        # Model instance
        self.synthesizer = None
        
    def _get_engine_name(self) -> str:
        """Get engine name for logging"""
        return "CoquiTTS"
        
    def _load_models(self) -> None:
        """Load CoquiTTS models"""
        
        self.synthesizer = TTS(
            model_path=self.model_path,
            config_path=self.config_path, 
            vocoder_path=self.vocoder_path,
            vocoder_config_path=self.vocoder_config_path,
            gpu=self.tts_gpu
        ).synthesizer
        
    def _cleanup_models(self) -> None:
        """Clean up CoquiTTS-specific model resources"""
        # Clear model references
        if self.synthesizer is not None:
            del self.synthesizer
            self.synthesizer = None


class CoquiTTS(BaseTTS):
    """
    CoquiTTS implementation with full functionality.
    Inherits all common functionality from BaseTTS including:
    - Text chunking and parallel processing
    - Audio combination with fading  
    - Streaming orchestrator support
    - Interrupt management
    - Async speak & stream methods
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize CoquiTTS model manager
        self.model_manager = CoquiModelManager()
        
        # Initialize common components
        self.audio_manager = AudioManager(sample_rate=48000)
        self.interrupt_manager = TTSInterruptManager()
        self.streaming_orchestrator = StreamingOrchestrator(self)
        
        # Start model loading
        self.model_manager.start_loading()
        
    async def initialize(self) -> None:
        """Initialize CoquiTTS engine asynchronously"""
        if not self.ready:
            print("🕐 Waiting for CoquiTTS to load...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.model_manager.wait_for_loading)
            self.ready = self.model_manager.is_ready()
    
    def _generate_chunk_core(self, text: str) -> np.ndarray:
        """CoquiTTS-specific audio generation"""
        if not self.model_manager.is_ready():
            raise RuntimeError("CoquiTTS model not loaded yet.")
            
        if not text.strip():
            return np.array([], dtype=np.float32)
            
        # print(f"🔊 Generating audio for text: {text}")
        # Generate audio using CoquiTTS 
        wav = self.model_manager.synthesizer.tts(text)
        
        # Convert to numpy array if needed
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
            
        return wav
    
    def close(self) -> None:
        """Clean up CoquiTTS-specific resources"""
        try:
            # Clean up CoquiTTS models
            self.model_manager.cleanup()
            
            # Call base cleanup
            super().close()
            
            print("✅ CoquiTTS cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in CoquiTTS cleanup: {e}") 