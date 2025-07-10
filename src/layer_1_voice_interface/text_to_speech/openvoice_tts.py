import torch
import numpy as np
import asyncio
from pathlib import Path

import yaml

from .tts_component.base_tts import BaseTTS
from .tts_component.base_model_manager import BaseTTSModelManager
from .tts_component.interrupt_manager import TTSInterruptManager
from ..audio_manager import AudioManager
from .streaming_component.orchestrator import StreamingOrchestrator

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


class OpenVoiceModelManager(BaseTTSModelManager):
    """
    Manages loading and initialization of OpenVoice TTS models.
    Inherits common functionality from BaseTTSModelManager.
    """
    
    def __init__(
        self,
        ckpt_base_dir: str,
        ckpt_converter_dir: str, 
        reference_speaker_path: str,
        output_dir: str
    ):
        super().__init__()
        
        self.ckpt_base = Path(ckpt_base_dir)
        self.ckpt_conv = Path(ckpt_converter_dir)
        self.ref_path = Path(reference_speaker_path)
        self.output_dir = Path(output_dir)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # Model instances
        self.base_tts = None
        self.converter = None
        self.source_se = None
        self.target_se = None
        
    def _get_engine_name(self) -> str:
        """Get engine name for logging"""
        return "OpenVoice"
        
    def _load_models(self) -> None:
        """Load OpenVoice models"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base TTS model
        self.base_tts = BaseSpeakerTTS(str(self.ckpt_base / 'config.json'), device=self.device)
        self.base_tts.load_ckpt(str(self.ckpt_base / 'checkpoint.pth'))
        
        # Load converter
        self.converter = ToneColorConverter(str(self.ckpt_conv / 'config.json'), device=self.device)
        self.converter.load_ckpt(str(self.ckpt_conv / 'checkpoint.pth'))
        
        # Load speaker embeddings
        self.source_se = torch.load(self.ckpt_base / 'en_default_se.pth').to(self.device)
        self.target_se = self._extract_target_se()
        
    def _extract_target_se(self):
        """Extract target speaker embedding from reference audio"""
        se, _ = se_extractor.get_se(
            str(self.ref_path),
            self.converter,
            target_dir=str(self.output_dir / 'processed'),
            vad=True
        )
        return se
        
    def _cleanup_models(self) -> None:
        """Clean up OpenVoice-specific model resources"""
        # Clear model references
        if self.base_tts is not None:
            del self.base_tts
            self.base_tts = None
            
        if self.converter is not None:
            del self.converter
            self.converter = None
            
        if self.source_se is not None:
            del self.source_se
            self.source_se = None
            
        if self.target_se is not None:
            del self.target_se
            self.target_se = None


class OpenVoiceTTS(BaseTTS):
    """
    OpenVoice Text-to-Speech implementation.
    Inherits all common functionality from BaseTTS including:
    - Text chunking and parallel processing
    - Audio combination with fading  
    - Streaming orchestrator support
    - Interrupt management
    - Async speak & stream methods
    """

    def __init__(
        self,
        ckpt_base_dir: str = config['openvoice_tts_ckpt_base_dir'],
        ckpt_converter_dir: str = config['openvoice_tts_ckpt_converter_dir'],
        reference_speaker_path: str = config['openvoice_tts_reference_speaker_path'],
        output_dir: str = config['openvoice_tts_output_dir'],
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell'
    ):
        super().__init__()
        
        # OpenVoice-specific configuration
        self.speaker = speaker
        self.language = language
        self.speed = speed
        self.emotion = emotion
        
        # Initialize OpenVoice model manager
        self.model_manager = OpenVoiceModelManager(
            ckpt_base_dir, ckpt_converter_dir, reference_speaker_path, output_dir
        )
        
        # Initialize common components
        self.audio_manager = AudioManager(sample_rate=48000)
        self.interrupt_manager = TTSInterruptManager()
        self.streaming_orchestrator = StreamingOrchestrator(self)
        
        # Start model loading
        self.model_manager.start_loading()
        
    async def initialize(self) -> None:
        """Initialize the OpenVoice TTS engine asynchronously"""
        if not self.ready:
            print("🕐 Waiting for OpenVoice TTS to load...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.model_manager.wait_for_loading)
            self.ready = self.model_manager.is_ready()
        
    def _generate_chunk_core(self, text: str) -> np.ndarray:
        """OpenVoice-specific audio generation"""
        if not self.model_manager.is_ready():
            raise RuntimeError("OpenVoice model not loaded yet.")
            
        # Generate using OpenVoice
        audio = self.model_manager.base_tts.tts(
            text, speaker=self.speaker, language=self.language, speed=self.speed
        )
        audio = self.model_manager.converter.convert(
            raw_audio=audio,
            src_se=self.model_manager.source_se,
            tgt_se=self.model_manager.target_se,
            message=self.emotion
        )
        
        return audio

    def close(self):
        """Clean up OpenVoice-specific resources"""
        try:
            # Clean up OpenVoice models
            self.model_manager.cleanup()
            
            # Call base cleanup
            super().close()
            
            print("✅ OpenVoice TTS cleanup completed")
            
        except Exception as e:
            print(f"⚠️ Error in OpenVoice TTS cleanup: {e}") 