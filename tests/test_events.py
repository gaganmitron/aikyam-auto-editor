"""creative/events.py: embedding-based event-boundary detection (EXP-004). Synthetic only --
the real-footage validation (95% cluster purity, boundary found at Diwali's actual day/night
transition) is in docs/research/experiment_matrix.md, not re-run here (no ML models in tests)."""
from aikyam_video.creative.events import event_boundaries, event_windows, scene_similarities
from aikyam_video.models import SceneVision, VisionResult


def sv(i, start, end, emb):
    return SceneVision(sceneId=f"s{i}", start=start, end=end, vision=VisionResult(embedding=emb))


def two_clusters():
    """5 scenes near [1,0,0,0] (an "A" event, 0-25s) then 5 near [0,1,0,0] (a "B" event, 25-50s),
    tiny per-scene jitter (asymmetric between clusters, so within-cluster similarities don't tie
    exactly at the percentile cutoff) so within-cluster similarity is high but not a suspicious 1.0."""
    a = [sv(i, i * 5, i * 5 + 5, [1.0, 0.02 * i, 0.0, 0.0]) for i in range(5)]
    b = [sv(i + 5, 25 + i * 5, 25 + i * 5 + 5, [0.013 * i, 1.0, 0.0, 0.0]) for i in range(5)]
    return a + b


def test_boundary_found_at_the_real_cluster_transition():
    # a percentile over few samples is a CANDIDATE generator, not a precise classifier -- with only
    # 9 pairs, np.percentile(15) mathematically ties in a near-lowest runner-up alongside the one
    # true outlier (same behavior seen on the real Diwali data: 15th pct flagged ~4-5 of 45 pairs,
    # not just the single true boundary). Check the real one is found and it isn't flagging most pairs.
    bounds = event_boundaries(two_clusters(), percentile=15.0)
    assert 25.0 in bounds
    assert len(bounds) <= 3


def test_windows_span_the_whole_source_and_split_at_the_boundary():
    windows = event_windows(two_clusters(), source_duration=50.0, percentile=15.0)
    assert windows[0][0] == 0.0 and windows[-1][1] == 50.0
    assert any(w == (25.0, 50.0) for w in windows) or any(w[0] == 25.0 for w in windows)


def test_too_few_scenes_reports_nothing_rather_than_guessing():
    assert event_boundaries(two_clusters()[:3], percentile=15.0) == []


def test_scenes_without_an_embedding_are_skipped_not_treated_as_zero_similarity():
    scenes = two_clusters()
    scenes[2] = sv(2, 10, 15, [])                                         # no embedding: must not fabricate a boundary here
    sims = scene_similarities(scenes)
    assert len(sims) == len(two_clusters()) - 2                           # scene 2's two adjacent pairs both drop out, not zeroed


# ---------------------------------------------------------------- integration (EXP-030): event windows reach Shot.event_index
def test_build_shots_tags_each_shot_with_its_event_window(tmp_path):
    """Two candidate moments either side of a real embedding-similarity drop get DIFFERENT event_index values; the source is the same integration point highlights.validate_clip already uses (ctx.event_bounds)."""
    import subprocess
    from aikyam_video.creative.shots import build_shots
    from aikyam_video.models import Moment
    f = str(tmp_path / "two_events.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=25:d=30", "-pix_fmt", "yuv420p", f], check=True)
    vis = two_clusters()                                                            # from this file: 5 scenes 0-25s ("A"), 5 scenes 25-50s ("B") -- reuse the fixture that already proves the boundary lands at 25.0
    moms = [Moment(momentId="mA", start=2.0, end=8.0, reason="x", score=0.5), Moment(momentId="mB", start=26.0, end=30.0, reason="x", score=0.4)]
    shots = build_shots(f, moms, vis, None, asset_id="a1", duration=30.0, edge_info=False)
    by_id = {s.moment_id: s for s in shots}
    assert by_id["mA"].event_index is not None and by_id["mB"].event_index is not None
    assert by_id["mA"].event_index != by_id["mB"].event_index                       # different real events, so different windows


def test_build_shots_leaves_event_index_none_without_enough_scenes_or_duration():
    from aikyam_video.creative.shots import build_shots
    from aikyam_video.models import Moment
    mom = [Moment(momentId="m0", start=0.0, end=4.0, reason="x", score=0.5)]
    vis_short = two_clusters()[:2]                                                  # too few scenes for a percentile (events.py's own floor: < 4 transitions)
    shots = build_shots("/dev/null", mom, vis_short, None, asset_id="a1", duration=None, edge_info=False)
    assert shots[0].event_index is None
