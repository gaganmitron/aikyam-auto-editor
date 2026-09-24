"""--no-captions (Options.captions=False): both the classic planner and the Creative Engine must
ship a plan with an empty caption overlay, not just an empty `enabled` flag with stale cues."""
import json
import pytest
from aikyam_video import pipeline, stages
from aikyam_video.options import Options


@pytest.mark.parametrize("engine", ["classic", "creative"])
def test_no_captions_produces_no_caption_cues(engine, sample, tmp_path):
    out = str(tmp_path / engine)
    o = Options(temple_id="temple_123", whisper_model="base", formats=["reel"], engine=engine, captions=False)
    files = pipeline.run(sample, out, "plan", o)
    plan = json.load(open(files["edit_plan"]))
    assert plan["captions"]["enabled"] is False
    assert plan["captions"]["cues"] == []
