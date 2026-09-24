"""Learned clip ranker (ranker.py): prior-regularised ridge, safe loading, opt-in scorer."""
import numpy as np
from types import SimpleNamespace
from aikyam_video import ranker, scoring
from aikyam_video.models import Scene, SceneVision, VisionResult


def _vr(aes, nt=0.0, dev_label=None, sharp=0.5, bright=0.5, faces=None):
    return VisionResult(labels={dev_label: 0.9} if dev_label else {}, faces=faces or [], brightness=bright, sharpness=sharp, aesthetic=aes, no_text=nt)


def test_fit_recovers_a_real_signal_and_the_prior_holds_when_data_is_noise():
    rng = np.random.default_rng(0); X = rng.standard_normal((200, 6)); y = 2.0 * X[:, 0] + 0.1 * rng.standard_normal(200)
    w = np.array(ranker.fit(X, y, lam=1.0).w); assert w[0] > 0.8 and np.abs(w[1:]).max() < 0.2          # finds the one feature that matters
    noise = rng.standard_normal(40); prior = [0.5, 0.25, 0.5, 0, 0, 0]
    w2 = np.array(ranker.fit(X[:40], noise, prior=prior, lam=1e6).w); assert np.allclose(w2, prior, atol=1e-3)   # huge lam: the prior wins


def test_featurize_needs_an_image_text_model():
    assert ranker.featurize(VisionResult()) is None
    x = ranker.featurize(_vr(3.0, 1.0, "deity", faces=[[0, 0, 0.1, 0.1]])); assert x.shape == (len(ranker.FEATURES),) and x[2] == scoring.DEVOTIONAL["deity"] * 0.9


def test_artifact_for_another_feature_set_is_refused(tmp_path):
    p = tmp_path / "r.json"; p.write_text('{"features": ["x"], "mean": [0], "std": [1], "w": [1]}'); assert ranker.load(p) is None
    assert ranker.load(tmp_path / "missing.json") is None


def test_learned_scorer_is_neutral_without_an_artifact_and_ranks_within_the_video_with_one(tmp_path, monkeypatch):
    sv = [SceneVision(sceneId=f"s{i}", start=i * 10.0, end=i * 10.0 + 10, vision=_vr(float(i))) for i in range(5)]
    ctx = SimpleNamespace(vision=sv)
    monkeypatch.setattr(ranker, "load", lambda path=None: None); assert scoring.SCORERS["learned"](ctx, 0, 10) == 0.5
    rk = ranker.Ranker([0.0] * 6, [1.0] * 6, [1.0, 0, 0, 0, 0, 0]); monkeypatch.setattr(ranker, "load", lambda path=None: rk)
    ctx = SimpleNamespace(vision=sv); lo, hi = scoring.SCORERS["learned"](ctx, 0, 10), scoring.SCORERS["learned"](ctx, 40, 50)
    assert hi > lo and 0.0 <= lo <= hi <= 1.0                                                                  # higher aesthetic -> higher percentile
    assert scoring.DEFAULT_WEIGHTS["learned"] == 0.0                                                            # opt-in


def test_ranker_fitted_on_another_model_is_ignored(monkeypatch):
    from aikyam_video import vision
    sv = [SceneVision(sceneId=f"s{i}", start=i * 10.0, end=i * 10.0 + 10, vision=_vr(float(i))) for i in range(5)]
    rk = ranker.Ranker([0.0] * 6, [1.0] * 6, [1.0, 0, 0, 0, 0, 0], meta={"vision_model": "hf-hub:timm/SOME-OTHER-MODEL"})
    monkeypatch.setattr(ranker, "load", lambda path=None: rk)
    assert scoring.SCORERS["learned"](SimpleNamespace(vision=sv), 0, 10) == 0.5                   # neutral: numbers from another model mean something else
    rk.meta["vision_model"] = vision.model_name()
    assert scoring.SCORERS["learned"](SimpleNamespace(vision=sv), 40, 50) > 0.5
