#!/usr/bin/env python
"""Second ASR pass over a finished reel vs the words the plan intended (see aikyam_video/creative/verify.py). Needs speech in the source clips.

    python tools/verify_reel.py results/my_run"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video.creative.verify import verify_render
from aikyam_video.models import Transcript
from aikyam_video.transcribe import FasterWhisperProvider

d = sys.argv[1]; plan = json.load(open(f"{d}/edit-plan.json"))
tr = {a["id"]: Transcript.model_validate_json(open(f"{d}/assets/{a['id']}/transcript.json").read()) for a in plan.get("assets", []) if os.path.exists(f"{d}/assets/{a['id']}/transcript.json")}
reel = next(f for f in ("reel-9x16.mp4", "square-1x1.mp4") if os.path.exists(f"{d}/{f}"))
for c in verify_render(plan, tr, FasterWhisperProvider().transcribe(f"{d}/{reel}")): print(f"{c.status:5} {c.name:22} {c.value} {c.detail}")
