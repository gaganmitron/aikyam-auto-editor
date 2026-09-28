"""Shot size (EXP-025): a zero-shot distribution over wide / medium / close / detail, stored with each scene and shot (no selection term uses it yet)."""
import numpy as np
from aikyam_video import shotsize
from aikyam_video.analysis import _merge
from aikyam_video.models import VisionResult


def _fake_embed(dim=32):
    rng = np.random.RandomState(0); table = {}
    def embed(texts):
        out = []
        for t in texts:
            if t not in table:
                v = rng.randn(dim); table[t] = v / np.linalg.norm(v)
            out.append(table[t])
        return np.array(out)
    return embed


def test_score_is_a_distribution_and_follows_the_prompt_it_matches():
    T, owner = shotsize.text_embeddings(_fake_embed())
    assert T.shape[0] == len(owner) == sum(len(v) for v in shotsize.SIZES.values()) and set(owner) == set(range(len(shotsize.KEYS)))
    for i, k in enumerate(shotsize.KEYS):
        emb = T[np.where(owner == i)[0][0]]                                   # an image "identical" to the first phrasing of class k
        d = shotsize.score(emb, T, owner, scale=20.0)
        assert abs(sum(d.values()) - 1.0) < 1e-6 and max(d, key=d.get) == k


def test_tightness_orders_the_sizes_and_blends_distributions():
    t = lambda k: shotsize.tightness({x: float(x == k) for x in shotsize.KEYS})
    assert t("wide") < t("medium") < t("close") < t("detail") and t("wide") == 0.0
    assert shotsize.tightness({"wide": 0.5, "medium": 0.5, "close": 0.0, "detail": 0.0}) == 0.5


def test_scene_merge_averages_shot_size_and_tolerates_old_caches():
    a = VisionResult(shot_size={"wide": 1.0, "medium": 0.0, "close": 0.0, "detail": 0.0}); b = VisionResult(shot_size={"wide": 0.0, "medium": 1.0, "close": 0.0, "detail": 0.0})
    m = _merge([a, b]); assert m.shot_size["wide"] == 0.5 and m.shot_size["medium"] == 0.5
    assert _merge([a, VisionResult()]).shot_size is None                        # a cache from before this field existed: unknown, not a wrong average
