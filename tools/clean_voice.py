#!/usr/bin/env python
"""Denoise the voice of a clip locally (RNNoise). Speech only: not for chant/music.

    python tools/clean_voice.py IN.mp4 [-o OUT.mp4]      (default OUT = IN-clean.ext)"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.audio_clean import denoise

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("-o"); ap.add_argument("--no-preserve-loudness", action="store_true"); a = ap.parse_args()
b, e = os.path.splitext(a.src)
print(denoise(a.src, a.o or f"{b}-clean{e}", preserve_loudness=not a.no_preserve_loudness))
