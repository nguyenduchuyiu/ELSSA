"""
Text-to-Speech module providing modular and extensible TTS engines.

Main classes:
- BaseTTS: Abstract base class for all TTS implementations
- OpenVoiceTTS: OpenVoice implementation with voice cloning  
- CoquiTTS: CoquiTTS implementation with neural vocoders
- TTSFactory: Factory for creating TTS instances

For backward compatibility:
- TextToSpeech: Alias to OpenVoiceTTS
- create_text_to_speech(): Function to create TTS instance
"""

# Abstract base class
from .tts_component.base_tts import BaseTTS

# Concrete implementations  
from .openvoice_tts import OpenVoiceTTS
from .coqui_tts import CoquiTTS

# Factory and utilities
from .tts_component.tts_factory import TTSFactory, create_text_to_speech

# Component classes (for advanced usage)
from .tts_component.interrupt_manager import TTSInterruptManager

__all__ = [
    # Main interfaces
    'BaseTTS',
    'OpenVoiceTTS',
    'CoquiTTS',
    'TTSFactory',
    
    # Convenience functions and legacy compatibility
    'create_text_to_speech',    
    # Component classes
    'TTSInterruptManager',
] 