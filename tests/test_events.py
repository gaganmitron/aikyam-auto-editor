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
