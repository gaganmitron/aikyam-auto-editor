"""EXP-020: the fireworks anchor of role_label_consistency must be an unambiguous scene. Real Tirumala case: the only 'fireworks' scenes were an arati scene
(aarti 0.92 > fireworks 0.6), so an arati clip was flagged as 'looks like fireworks' by comparison with itself."""
import json
from aikyam_video.creative import qc


def _run(tmp_path, scenes, seg_scene):
    emb = {f"s{i}": e for i, (e, _) in enumerate(scenes)}
    vis = [{"sceneId": f"s{i}", "start": 10.0 * i, "end": 10.0 * i + 10.0, "vision": {"labels": l}} for i, (_, l) in enumerate(scenes)]
    (tmp_path / "embeddings.json").write_text(json.dumps(emb)); (tmp_path / "vision.json").write_text(json.dumps(vis))
    plan = {"segments": [{"start": 10.0 * seg_scene + 1, "end": 10.0 * seg_scene + 5, "reason": "aarti", "kind": "video"}]}
    return qc.check_role_label_consistency(plan, str(tmp_path))[0]


def test_ambiguous_fireworks_scene_is_not_an_anchor_but_real_fireworks_still_is(tmp_path):
    e = lambda *v: (lambda a: [x / sum(y * y for y in a) ** .5 for x in a])(list(v))
    devo = ({"deity": 0.9}, e(1, 0, 0)); flame = ({"aarti": 0.92, "fireworks": 0.6}, e(0, 1, 0)); clip = ({"aarti": 0.9}, e(0, 0.95, 0.05))
    r = _run(tmp_path, [(devo[1], devo[0]), (flame[1], flame[0]), (clip[1], clip[0])], 2)
    assert r.status == "pass"                                       # no unambiguous fireworks example -> the check has nothing to compare against
    real = ({"fireworks": 0.9}, e(0, 1, 0))
    r2 = _run(tmp_path, [(devo[1], devo[0]), (real[1], real[0]), (clip[1], clip[0])], 2)
    assert r2.status == "warn"                                      # a genuinely fireworks-like source still triggers the original check
