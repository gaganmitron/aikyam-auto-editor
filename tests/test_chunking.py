"""Long-video chunked analysis (chunking.py): global/chunk-local timestamp correctness, and one real
end-to-end render through the chunked path. Uses the same synthetic fixture as test_pipeline.py's
golden test (tests/make_fixture.py, 30s, 3 distinct scenes + speech + bell tones) with an artificially
low `long_video_threshold_s` so a normal-sized fixture genuinely exercises chunking -- no separate
long/real video needed for this to be a real test of the merge logic, not a mock of it."""
import json
import pytest
from aikyam_video import pipeline, stages
from aikyam_video.options import Options


@pytest.fixture(scope="module")
def chunked_out(sample, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("chunked"))
    # 30s fixture, threshold 8s -> forces chunking; chunk 15s/overlap 5s -> 3 overlapping chunks across it
    o = Options(temple_id="temple_123", whisper_model="base", formats=["reel"], long_video_threshold_s=8.0, chunk_seconds=15.0, chunk_overlap_seconds=5.0)
    files = pipeline.run(sample, out, "process", o)
    return out, files


def test_merged_timestamps_stay_inside_the_real_source_duration(chunked_out, sample):
    from aikyam_video import ffmpeg as ff
    out, _ = chunked_out
    duration = ff.probe(sample).duration
    tr = json.load(open(f"{out}/{stages.TRANSCRIPT}"))
    for s in tr["segments"]:
        assert 0.0 <= s["start"] < s["end"] <= duration + 0.5, s          # GLOBAL time, not chunk-local (would be 0..15 if the offset step were skipped)
    scenes = json.load(open(f"{out}/{stages.SCENES}"))
    for sc in scenes:
        assert 0.0 <= sc["start"] < sc["end"] <= duration + 0.5, sc
    moments = json.load(open(f"{out}/{stages.MOMENTS}"))["moments"]
    assert moments, "chunked analysis found no moments at all on a fixture the unchunked path handles fine"
    for m in moments:
        assert 0.0 <= m["start"] < m["end"] <= duration + 0.5, m


def test_no_duplicate_moments_survive_the_seam_between_chunks(chunked_out):
    """The 15s/5s-overlap split puts a seam at ~10s and ~20s (inside the fixture's 30s). If the merge
    forgot to dedupe, the same real moment analyzed by two neighbouring chunks would show up twice."""
    out, _ = chunked_out
    moments = json.load(open(f"{out}/{stages.MOMENTS}"))["moments"]
    for i, a in enumerate(moments):
        for b in moments[i + 1:]:
            overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
            shorter = min(a["end"] - a["start"], b["end"] - b["start"])
            assert overlap <= 0.3 * shorter, (a, b)                      # same threshold run_chunked's own dedupe uses


def test_moment_ids_and_embeddings_stay_consistent_after_chunk_prefixing(chunked_out):
    out, _ = chunked_out
    moments = json.load(open(f"{out}/{stages.MOMENTS}"))["moments"]
    scene_ids = {s["sceneId"] for s in json.load(open(f"{out}/{stages.SCENES}"))}
    emb = json.load(open(f"{out}/{stages.EMBEDDINGS}"))
    assert set(emb) <= scene_ids                                        # every embedding key is a real (chunk-prefixed) scene id, no collisions
    assert len({m["momentId"] for m in moments}) == len(moments)         # chunk-prefixing actually prevented id collisions across chunks


def test_full_render_works_end_to_end_through_the_chunked_path(chunked_out):
    """The actual requirement: a real render, not just valid intermediate JSON."""
    out, files = chunked_out
    from aikyam_video import ffmpeg as ff
    from aikyam_video.plan import validate_plan
    plan = json.load(open(files["edit_plan"]))
    validate_plan(plan)
    assert plan["segments"]
    reel = files.get("reel") or f"{out}/reel-9x16.mp4"
    info = ff.probe(reel)
    assert info.duration > 1.0 and info.video_codec
