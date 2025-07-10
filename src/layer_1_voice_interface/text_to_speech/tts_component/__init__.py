"""
TTS Component Package

This package contains the core components for the modular TTS system:
- BaseTTS: Abstract base class with common functionality
- BaseTTSModelManager: Abstract base class for model managers
- TTSInterruptManager: Handles wake word interrupts
- TTSFactory: Factory pattern for creating TTS instances
"""

from .base_tts import BaseTTS
from .base_model_manager import BaseTTSModelManager
from .interrupt_manager import TTSInterruptManager
from .tts_factory import TTSFactory

__all__ = [
    'BaseTTS',
    'BaseTTSModelManager',
    'TTSInterruptManager', 
    'TTSFactory'
] 