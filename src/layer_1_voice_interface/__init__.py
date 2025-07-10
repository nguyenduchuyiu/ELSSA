"""
Voice Interface Layer
--------------------
This layer provides speech-to-text and text-to-speech capabilities,
as well as wake word detection for hands-free operation.
"""

from .audio_manager import AudioManager
from .speech_to_text import SpeechToText
from .text_to_speech import OpenVoiceTTS, CoquiTTS
from .wake_word_handler import WakeWordHandler

__all__ = [
    'AudioManager',
    'SpeechToText',
    'OpenVoiceTTS',
    'CoquiTTS',
    'WakeWordHandler',
]