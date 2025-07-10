from typing import Optional, Dict, Any
from .base_tts import BaseTTS
from ..openvoice_tts import OpenVoiceTTS
from ..coqui_tts import CoquiTTS


class TTSFactory:
    """
    Factory class for creating TTS instances.
    Provides easy instantiation and maintains backward compatibility.
    """
    
    @staticmethod
    def create_tts(
        engine: str = "openvoice",
        **kwargs
    ) -> BaseTTS:
        """
        Create a TTS instance based on the specified engine.
        
        Args:
            engine: TTS engine type ("openvoice", "coqui")
            **kwargs: Engine-specific configuration parameters
            
        Returns:
            BaseTTS instance
            
        Raises:
            ValueError: If engine type is not supported
        """
        
        engine = engine.lower()
        
        if engine == "openvoice":
            return OpenVoiceTTS(**kwargs)
        elif engine == "coqui":
            return CoquiTTS(**kwargs)
        else:
            available = ", ".join(TTSFactory.get_available_engines().keys())
            raise ValueError(f"Unsupported TTS engine: {engine}. Available: {available}")
    
    @staticmethod
    def get_available_engines() -> Dict[str, str]:
        """
        Get list of available TTS engines.
        
        Returns:
            Dict mapping engine names to descriptions
        """
        return {
            "openvoice": "OpenVoice multi-language TTS with voice cloning support",
            "coqui": "CoquiTTS open-source TTS with neural vocoders"
        }


# Convenience function for backward compatibility
def create_text_to_speech(engine: str = "openvoice", **kwargs) -> BaseTTS:
    """
    Convenience function to create TTS instance.
    Maintains backward compatibility with existing code.
    
    Args:
        engine: TTS engine to use (default: "openvoice")
        **kwargs: Engine-specific configuration
    """
    return TTSFactory.create_tts(engine, **kwargs)


# Legacy alias for backward compatibility
TextToSpeech = OpenVoiceTTS 