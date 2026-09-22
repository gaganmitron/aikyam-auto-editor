"""faster-whisper TranscriptionProvider (section 5). Swap for Google STT v2 by registering another provider."""
from __future__ import annotations
import math, os
from typing import Optional
from .models import Transcript, TranscriptSegment, Word
from .providers import TranscriptionProvider, register


class FasterWhisperProvider(TranscriptionProvider):
    name = "faster-whisper"

    def __init__(self, model: Optional[str] = None, device: str = "auto", compute_type: str = "int8",
                 language: Optional[str] = None, translate: bool = False,
                 min_confidence: float = 0.5):
        self.model_name = model or os.environ.get("WHISPER_MODEL", "base")
        self.device, self.compute_type = device, compute_type
        self.language, self.translate = language, translate
        self.min_confidence = min_confidence
        self._model = None

    def transcribe(self, media_path: str) -> Transcript:
        from . import ffmpeg as _ff
        if not _ff.has_audio_stream(media_path):                    # picture-only clip (stock footage): nothing to transcribe, and the decoder would crash on the missing stream
            return Transcript(language=self.language or "en", language_probability=0.0, segments=[])
        if self._model is None:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        segs, info = self._model.transcribe(
            media_path, language=self.language, word_timestamps=True, vad_filter=True,
            task="translate" if self.translate else "transcribe")
        out = []
        for s in segs:
            words = [Word(start=w.start, end=w.end, text=w.word.strip(), confidence=float(w.probability))
                     for w in (s.words or [])]
            conf, text = float(math.exp(s.avg_logprob)), s.text.strip()
            toks = text.lower().split()
            # Whisper hallucinates on singing/music/noise: low confidence, "no speech", or the same few words repeated
            if conf < self.min_confidence or s.no_speech_prob > 0.6 or (len(toks) >= 3 and len(set(toks)) / len(toks) < 0.5):
                continue
            out.append(TranscriptSegment(start=s.start, end=s.end, text=text, confidence=conf, words=words))
        return Transcript(language="en" if self.translate else info.language,
                          language_probability=float(info.language_probability), segments=out)


register("transcription", "faster-whisper", FasterWhisperProvider)
