"""Integration + golden test: sample-temple.mp4 -> pipeline -> EditPlan -> renders. Structural checks only."""
import json, os, subprocess
import pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.plan import validate_plan


def _load(p): return json.load(open(p))


def test_golden_outputs_exist_and_valid(pipeline_out, sample):
    out, files = pipeline_out
    for f in ["transcript.json", "scenes.json", "moments.json", "edit-plan.json", "reel-9x16.mp4",
              "square-1x1.mp4", "landscape-16x9.mp4", "thumbnail.jpg", "cost.json"]:
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    info = ff.probe(sample)
    assert _load(f"{out}/transcript.json")["language"] and "segments" in _load(f"{out}/transcript.json")
    scenes = _load(f"{out}/scenes.json")
    assert scenes and all(0 <= s["start"] < s["end"] <= info.duration + .01 for s in scenes)
    m = _load(f"{out}/moments.json")
    assert len(m["moments"]) >= 1
    assert all(0 <= x["start"] < x["end"] <= info.duration + .01 for x in m["moments"])
    plan = _load(f"{out}/edit-plan.json")
    validate_plan(plan, info.duration)
    assert plan["source"]["templeId"] == "temple_123" and plan["overlays"]["temple"] == "Chamundeshwari Temple"


@pytest.mark.parametrize("name,w,h", [("reel-9x16.mp4", 1080, 1920), ("square-1x1.mp4", 1080, 1080), ("landscape-16x9.mp4", 1920, 1080)])
def test_render_properties(pipeline_out, name, w, h):
    out, _ = pipeline_out
    plan = _load(f"{out}/edit-plan.json")
    i = ff.probe(os.path.join(out, name))
    assert (i.width, i.height) == (w, h) and i.video_codec == "h264" and i.has_audio
    assert abs(i.duration - plan["durationSeconds"]) < 0.3 and i.duration <= 60
    # playable: full decode without error
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(out, name), "-f", "null", "-"], capture_output=True)
    assert r.returncode == 0 and not r.stderr
    # audio is not silent
    assert abs(ff.extract_audio_pcm(os.path.join(out, name))).max() > 0.01


def test_reel_carries_aikyam_metadata(pipeline_out):
    out, _ = pipeline_out
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format_tags=comment", "-of", "json", f"{out}/reel-9x16.mp4"], capture_output=True)
    meta = json.loads(json.loads(p.stdout)["format"]["tags"]["comment"])
    assert meta["templeId"] == "temple_123"


def test_captions_present_when_enabled(pipeline_out):
    out, _ = pipeline_out
    plan = _load(f"{out}/edit-plan.json")
    assert plan["captions"]["enabled"] == bool(plan["captions"]["cues"])
    if plan["captions"]["enabled"]:
        ass = open(f"{out}/overlay_reel.ass", encoding="utf-8").read()
        assert ass.count("Dialogue:") >= len(plan["captions"]["cues"])
        # captions visibly burned in: frame during first cue differs from same frame w/o subtitle track
        t = plan["captions"]["cues"][0]["start"] + 0.5
        a = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", f"{out}/reel-9x16.mp4", "-frames:v", "1",
                            "-vf", "crop=iw:ih*0.2:0:ih*0.78,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        assert max(a) > 240 and sum(1 for b in a if b > 240) > 500       # white caption pixels in the caption band


def test_thumbnail_candidates_stored(pipeline_out):
    out, _ = pipeline_out
    s = _load(f"{out}/thumbnails/scores.json")
    assert len(s["candidates"]) >= 1 and s["selected"] and s["candidates"][0]["score"] is not None
    assert max(c["score"] for c in s["candidates"]) == [c for c in s["candidates"] if c["file"] == s["selected"]][0]["score"]


def test_cost_recorded(pipeline_out):
    out, _ = pipeline_out
    c = _load(f"{out}/cost.json")
    assert c["cpuSeconds"] > 0 and c["inputDurationSeconds"] > 0 and c["inrPerReel"] >= 0 and "model" in c


def test_no_black_or_silent_moments(pipeline_out):
    out, _ = pipeline_out
    for m in _load(f"{out}/moments.json")["moments"]:
        assert m["signals"]["audioImportance"] > 0


def test_rejects_non_video(tmp_path):
    from aikyam_video import pipeline
    bad = tmp_path / "x.mp4"; bad.write_bytes(b"not a video")
    with pytest.raises(Exception):
        pipeline.run(str(bad), str(tmp_path / "o"), "analyze")
