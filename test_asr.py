from whispercpp import Whisper
import soundfile as sf
import numpy as np

model = Whisper.from_pretrained(model_name="tiny.en", basedir="models")

# Read audio file
audio, sr = sf.read("assets/audio/ref_voice.mp3")
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # convert to mono if stereo

# Whisper expects float32 PCM, 16kHz
if sr != 16000:
    import librosa
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    sr = 16000
audio = audio.astype(np.float32)

chunk_size = int(16000 * 5)  
overlap = 16000 // 2

results = []
start = 0
segments = []
while start < len(audio):
    end = int(start + chunk_size)
    chunk = audio[int(start):int(end)]
    if len(chunk) == 0:
        break

    # Calculate the offset in ms for this chunk
    chunk_offset_ms = int(start * 1000 / sr)
    segment = model.transcribe(chunk)
    # Adjust start_ms and end_ms for each segment in this chunk
    for seg in segment:
        seg["text"] = seg["text"].removesuffix(".").strip()
        seg['start_ms'] += chunk_offset_ms
        seg['end_ms'] += chunk_offset_ms
        print(seg)
    segments.extend(segment)
    start += chunk_size - overlap

import typing as t

def _find_best_overlap(prev: str, nxt: str) -> str:
    prev_words = prev.strip().split()
    next_words = nxt.strip().split()
    for i in range(min(len(prev_words), len(next_words)), 0, -1):
        if prev_words[-i:] == next_words[:i]:
            return ' '.join(next_words[:i])
    return ""

def _merge_overlapping_segments(segments: t.List[t.Dict[str, t.Any]]) -> t.List[t.Dict[str, t.Any]]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg['start_ms'] < last['end_ms']:
            overlap = _find_best_overlap(last['text'], seg['text'])
            if overlap:
                non_overlap = seg['text'].strip()[len(overlap):].strip()
                last['text'] = (last['text'].strip() + ' ' + non_overlap).strip()
                last['end_ms'] = seg['end_ms']
            else:
                merged.append(seg.copy())
        else:
            merged.append(seg.copy())
    return merged

def _complete_sentence(segments: t.List[t.Dict[str, t.Any]]) -> str:
    merged_segments = _merge_overlapping_segments(segments)
    return ' '.join([seg['text'].strip() for seg in merged_segments])

text = _complete_sentence(segments)
print(text)
