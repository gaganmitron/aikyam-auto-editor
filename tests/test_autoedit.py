import json, os
from aikyam_video.creative import autoedit as A
from aikyam_video.options import Options


def fake_run(scores):
    """run_fn writing a fake reel + qc for each variant; `scores` maps opening/pacing -> hook_score (stands in for the pipeline)."""
    calls = []
    def run(o, d):
        calls.append((o.opening, o.pacing)); open(os.path.join(d, "reel-9x16.mp4"), "w").write(f"{o.opening}/{o.pacing}")
        json.dump({"status": "pass", "checks": []}, open(os.path.join(d, "qc.json"), "w")); open(os.path.join(d, "edit-plan.json"), "w").write("{}")
        if os.path.exists(os.path.join(d, "assets_marker")) is False and not calls[1:]: open(os.path.join(d, "assets_marker"), "w").write("analysis")
        return {}
    return run, calls


def ev_from(scores):
    return lambda d: {"hook_score": scores.get(open(os.path.join(d, "reel-9x16.mp4")).read(), 0.1), "cuts_on_beat_pct": 1.0}


def test_best_variant_wins_and_analysis_is_reused(tmp_path):
    run, calls = fake_run({}); out = str(tmp_path)
    win, rows = A.best_of(run, out, Options(), 3, evaluate=ev_from({"establish/None": 0.9}))
    assert win == "establish" and [r["variant"] for r in rows] == ["default", "establish", "hook"] and len(calls) == 3
    assert open(os.path.join(out, "reel-9x16.mp4")).read() == "establish/None"                     # winner copied to the run dir
    assert os.path.exists(os.path.join(out, "var_hook", "assets_marker")) and not os.path.exists(os.path.join(out, "var_hook", "reel-9x16.mp4.bak"))   # analysis carried over
    assert json.load(open(os.path.join(out, "variants.json")))["winner"] == "establish"


def test_ties_keep_default_and_failed_variant_does_not_lose_the_rest(tmp_path):
    def run(o, d):
        if o.opening == "establish": raise RuntimeError("boom")
        open(os.path.join(d, "reel-9x16.mp4"), "w").write("x"); json.dump({"checks": []}, open(os.path.join(d, "qc.json"), "w"))
    win, rows = A.best_of(run, str(tmp_path), Options(), 3, evaluate=lambda d: {"hook_score": 0.5})
    assert win == "default" and any("boom" in r.get("error", "") for r in rows)


def test_score_penalises_qc_problems():
    good = A.reel_score({"hook_score": 0.5}, {"checks": []}); bad = A.reel_score({"hook_score": 0.5}, {"checks": [{"status": "fail"}, {"status": "warn"}]})
    assert good - bad == A.W["fail"] + A.W["warn"]


def test_user_choice_is_not_a_variant(tmp_path):
    run, calls = fake_run({}); A.best_of(run, str(tmp_path / "a"), Options(opening="hook", pacing="festive") if os.makedirs(tmp_path / "a") is None else None, 5, evaluate=lambda d: {"hook_score": 0.5})
    assert calls == [("hook", "festive")]                                                              # opening and pacing both fixed by the person: only the default edit remains
    run, calls = fake_run({}); os.makedirs(tmp_path / "b"); A.best_of(run, str(tmp_path / "b"), Options(opening="hook"), 5, evaluate=lambda d: {"hook_score": 0.5})
    assert all(c[0] == "hook" for c in calls) and len(calls) == 3                                     # default + the two pacing variants, opening untouched
