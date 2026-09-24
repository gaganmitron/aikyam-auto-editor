"""Voiceover (creative/narration.py): espeak-ng synthesis and ducking, against the real espeak-ng
binary (fast for a few words -- no need to mock it)."""
import numpy as np
from aikyam_video.creative import mixer, narration


def test_synthesize_produces_audio_for_real_text_and_silence_for_blank():
    assert len(narration._synthesize("Aarti begins at the temple.")) > 0
    assert len(narration._synthesize("   ")) == 0


def test_apply_voiceover_ducks_the_mix_and_adds_speech_without_blowing_the_ceiling():
    sr = narration.SR
    dur = 3.0
    n = int(dur * sr)
    live = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)
    pcm = np.stack([live, live], axis=1)
    mix = mixer.MixResult(pcm=pcm.copy(), live=pcm.copy(), music=np.zeros_like(pcm), report={"music_used": False})
    cues = [{"start": 0.5, "end": 1.5, "text": "Welcome to the temple."}]

    out = narration.apply_voiceover(mix, cues, dur)

    assert out.pcm.shape == mix.pcm.shape
    assert not np.allclose(out.pcm, mix.pcm)                         # something changed
    assert out.report["voiceover"] is True
    assert out.report["true_peak_db"] <= mixer.CEILING_DB + 0.2      # re-limited, not left hot


def test_apply_voiceover_is_a_noop_with_no_cues():
    sr = narration.SR
    pcm = np.zeros((sr, 2), dtype=np.float32)
    mix = mixer.MixResult(pcm=pcm, live=pcm, music=np.zeros_like(pcm), report={})
    out = narration.apply_voiceover(mix, [], 1.0)
    assert out is mix
