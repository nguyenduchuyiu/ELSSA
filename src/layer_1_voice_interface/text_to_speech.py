# File: src/layer_1_voice_interface/text_to_speech.py
import torch
import json
import numpy as np
import sounddevice as sd
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional
from openvoice import se_extractor  # type: ignore
from openvoice.api import BaseSpeakerTTS, ToneColorConverter  # type: ignore


def _split_text(text: str, max_len: int = 200) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +|\n+', text)
    chunks, current = [], ''
    for sent in sentences:
        sent = sent.strip()
        if len(current) + len(sent) + 1 <= max_len:
            current += sent + ' '
        else:
            chunks.append(current.strip())
            current = sent + ' '
    if current:
        chunks.append(current.strip())
    return chunks


def _apply_fade(audio: np.ndarray, sr: int, fade_ms: int = 10) -> np.ndarray:
    fade_samples = int(sr * fade_ms / 1000)
    if fade_samples <= 0 or audio.shape[0] < 2 * fade_samples:
        return audio
    fade = np.linspace(0.0, 1.0, fade_samples)
    audio[:fade_samples] *= fade
    audio[-fade_samples:] *= fade[::-1]
    return audio


class TextToSpeech:
    def __init__(
        self,
        ckpt_base_dir: str = 'openvoice/checkpoints/base_speakers/EN',
        ckpt_converter_dir: str = 'openvoice/checkpoints/converter',
        reference_speaker_path: str = 'openvoice/resources/ref_voice.mp3',
        output_dir: str = 'openvoice/outputs',
        device: Optional[str] = None,
        max_len: int = 200,
        fade_ms: int = 10,
    ):
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.ckpt_base = Path(ckpt_base_dir)
        self.ckpt_conv = Path(ckpt_converter_dir)
        self.ref_path = Path(reference_speaker_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.ckpt_base / 'config.json', 'r') as f:
            cfg = json.load(f)
        self.sr = cfg['data']['sampling_rate']

        self.base_tts = BaseSpeakerTTS(str(self.ckpt_base / 'config.json'), device=self.device)
        self.base_tts.load_ckpt(str(self.ckpt_base / 'checkpoint.pth'))
        self.converter = ToneColorConverter(str(self.ckpt_conv / 'config.json'), device=self.device)
        self.converter.load_ckpt(str(self.ckpt_conv / 'checkpoint.pth'))

        self.source_se = torch.load(self.ckpt_base / 'en_default_se.pth').to(self.device)
        self.target_se = self._extract_target_se()

        self.max_len = max_len
        self.fade_ms = fade_ms
        self.executor = ThreadPoolExecutor(max_workers=3)

    def _extract_target_se(self):
        se, _ = se_extractor.get_se(
            str(self.ref_path),
            self.converter,
            target_dir=str(self.output_dir / 'processed'),
            vad=True
        )
        return se

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
        self.base_tts.tts(text_chunk, str(tmp), speaker=speaker, language=language, speed=speed)
        audio = self.converter.convert(
            audio_src_path=str(tmp),
            src_se=self.source_se,
            tgt_se=self.target_se,
            message=emotion
        )
        return _apply_fade(audio, self.sr, self.fade_ms)

    def speak(
        self,
        text: str,
        speaker: str = 'default',
        language: str = 'English',
        speed: float = 1.0,
        emotion: str = '@MyShell'
    ) -> np.ndarray:
        chunks = _split_text(text, max_len=self.max_len)
        if not chunks:
            return np.array([], dtype=np.float32)

        # Initialize pipelined futures
        futures: List[Future] = []
        # Submit first chunk
        futures.append(self.executor.submit(
            self._generate_chunk, chunks[0], 0, speaker, language, speed, emotion
        ))
        # Submit second chunk if exists
        if len(chunks) > 1:
            futures.append(self.executor.submit(
                self._generate_chunk, chunks[1], 1, speaker, language, speed, emotion
            ))

        # Prepare audio stream after first chunk ready
        # Get and play chunk 0
        audio0 = futures[0].result()
        channels = 1 if audio0.ndim == 1 else audio0.shape[1]
        stream = sd.OutputStream(samplerate=self.sr, channels=channels, dtype='float32', blocksize=0)
        stream.start()
        stream.write(audio0)
        audio_segments = [audio0]

        # Loop through remaining chunks
        for i in range(1, len(chunks)):
            # Prefetch next chunk ahead
            if i + 1 < len(chunks):
                futures.append(self.executor.submit(
                    self._generate_chunk, chunks[i+1], i+1, speaker, language, speed, emotion
                ))
            # Get current chunk and play
            aud = futures[i].result()
            stream.write(aud)
            audio_segments.append(aud)

        stream.stop()
        stream.close()

        return np.concatenate(audio_segments)

    def close(self):
        self.executor.shutdown()

