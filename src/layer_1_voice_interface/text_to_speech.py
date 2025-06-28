import torch
import json
import numpy as np
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional
import threading

from openvoice import se_extractor  # type: ignore
from openvoice.api import BaseSpeakerTTS, ToneColorConverter  # type: ignore

from .audio_manager import AudioManager

class TextToSpeech:
    """
    Text-to-Speech engine using OpenVoice, supports text splitting, voice conversion,
    and direct audio playback. Loads resources lazily in separate background thread.
    """
    
    DEFAULT_BASE_DIR = 'openvoice/checkpoints/base_speakers/EN'
    DEFAULT_CONVERTER_DIR = 'openvoice/checkpoints/converter'
    DEFAULT_REFERENCE_PATH = 'openvoice/resources/ref_voice.mp3'
    DEFAULT_OUTPUT_DIR = 'openvoice/outputs'
    
    def __init__(
        self,
        ckpt_base_dir: str = DEFAULT_BASE_DIR,
        ckpt_converter_dir: str = DEFAULT_CONVERTER_DIR,
        reference_speaker_path: str = DEFAULT_REFERENCE_PATH,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        device: Optional[str] = None,
        max_len: int = 200,
        fade_ms: int = 10,
        sample_rate: int = 22050,  # OpenVoice default
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ckpt_base = Path(ckpt_base_dir)
        self.ckpt_conv = Path(ckpt_converter_dir)
        self.ref_path = Path(reference_speaker_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_len = max_len
        self.fade_ms = fade_ms
        self.sample_rate = sample_rate

        # Audio manager
        self.audio_manager = AudioManager(sample_rate=sample_rate)

        # Executor for synthesis tasks
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.ready = False

        # Attributes to initialize
        self.base_tts = None
        self.converter = None
        self.source_se = None
        self.target_se = None

        # Start dedicated thread for loading (non-daemon to ensure scheduling)
        self._load_thread = threading.Thread(target=self._lazy_init)
        self._load_thread.start()

    def _lazy_init(self):
        try:
            print("🔄 Loading TTS model in background...")
            # Load base TTS model config
            with open(self.ckpt_base / 'config.json', 'r') as f:
                cfg = json.load(f)
            self.sample_rate = cfg['data']['sampling_rate']

            # Instantiate and load checkpoints
            self.base_tts = BaseSpeakerTTS(str(self.ckpt_base / 'config.json'), device=self.device)
            self.base_tts.load_ckpt(str(self.ckpt_base / 'checkpoint.pth'))
            self.converter = ToneColorConverter(str(self.ckpt_conv / 'config.json'), device=self.device)
            self.converter.load_ckpt(str(self.ckpt_conv / 'checkpoint.pth'))

            # Speaker embeddings
            self.source_se = torch.load(self.ckpt_base / 'en_default_se.pth').to(self.device)
            self.target_se = self._extract_target_se()

            self.ready = True
            print("✅ TTS model loaded and ready.")
        except Exception as e:
            print(f"❌ Error initializing TTS: {e}")

    def _extract_target_se(self):
        se, _ = se_extractor.get_se(
            str(self.ref_path),
            self.converter,
            target_dir=str(self.output_dir / 'processed'),
            vad=True
        )
        return se

    def _split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks for TTS processing"""
        sentences = re.split(r'(?<=[.!?]) +|\n+', text)
        chunks, current = [], ''
        for sent in sentences:
            sent = sent.strip()
            if len(current) + len(sent) + 1 <= self.max_len:
                current += sent + ' '
            else:
                chunks.append(current.strip())
                current = sent + ' '
        if current:
            chunks.append(current.strip())
        return chunks

    def _generate_chunk(
        self,
        text_chunk: str,
        idx: int,
        speaker: str,
        language: str,
        speed: float,
        emotion: str
    ) -> np.ndarray:
        tmp = self.output_dir / f'chunk_{idx}.wav'
        if not self.ready:
            raise RuntimeError("TTS model not loaded yet.")
        self.base_tts.tts(text_chunk, str(tmp), speaker=speaker, language=language, speed=speed)
        audio = self.converter.convert(
            audio_src_path=str(tmp),
            src_se=self.source_se,
            tgt_se=self.target_se,
            message=emotion
        )
        return self.audio_manager.apply_fade(audio, self.fade_ms)

    def speak(
        self,
        text: str,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell',
        play_audio: bool = True
    ) -> np.ndarray:
        """
        Convert text to speech and optionally play it
        
        Args:
            text: Text to synthesize
            speaker: Speaker voice to use
            language: Language for synthesis
            speed: Speech rate multiplier
            emotion: Emotion style for voice
            play_audio: Whether to play audio immediately
            
        Returns:
            Audio data as numpy array
        """
        # Ensure load thread has opportunity to run early
        if not self.ready and self._load_thread.is_alive():
            print("🕐 Waiting briefly for TTS to load...")
            self._load_thread.join(timeout=0)
        if not self.ready:
            # If still not ready, block fully
            print("🕐 Blocking until TTS ready...")
            self._load_thread.join()
            
        chunks = self._split_text(text)
        if not chunks:
            return np.array([], dtype=np.float32)

        futures: List[Future] = []
        for i, chunk_text in enumerate(chunks):
            futures.append(self.executor.submit(
                self._generate_chunk, chunk_text, i, speaker, language, speed, emotion
            ))

        combined_audio = []
        for fut in futures:
            audio = fut.result()
            combined_audio.append(audio)
            if play_audio:
                self.audio_manager.play_audio(audio)

        return np.concatenate(combined_audio)

    def close(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        if self._load_thread.is_alive():
            self._load_thread.join()
        self.audio_manager.close()