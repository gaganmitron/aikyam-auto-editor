from aikyam_video.creative import pacing, story
from aikyam_video.creative.shots import Population, build_shots
from aikyam_video.models import Moment, SceneVision, VisionResult
from tests.test_creative_story import e, shot


def pool(n=10):
    return [shot(i, i * 25, 8, {"aarti": 0.6 + 0.03 * (i % 4), "lamps": 0.5}, e(*[1.0 if j == i % 5 else 0.05 * (i // 5 + 1) for j in range(5)]), 0.5 + 0.02 * i, motion=[2 + (i % 3)] * 4 + [5] * 12, rms=-30) for i in range(n)]


def total(tl): return sum(c.length for c in tl.clips)


def test_an_explicit_target_is_reached_by_repeating_middle_roles_beyond_six_clips():
    sh = pool(); pop = Population(sh); prof = pacing.profile("devotional")
    base = story.plan_story(sh, pop, prof, 300.0, 45.0)
    assert len(base.clips) <= 9 and total(base) >= 45 - 2 - 5 * 0.5                                    # 45 s asked: ~41 s+ of clips (blends eat a little), never the 6-role ceiling of ~30 s
    assert any(d["type"] == "fill" for d in base.decisions) or len(base.clips) >= 6
    no_target = story.plan_story(sh, pop, prof, 300.0)
    assert len(no_target.clips) <= 7 and all(r in story.ROLES for r in no_target.arc)
    roles = [c.role for c in base.clips]; assert roles == sorted(roles, key=story.ROLES.index)          # the arc order still holds


def test_window_labels_are_time_weighted_not_the_luckiest_frame(tmp_path):
    import subprocess
    f = str(tmp_path / "v.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=160x90:r=10:d=10", "-pix_fmt", "yuv420p", f], check=True)
    vis = [SceneVision(sceneId="a", start=0, end=1, vision=VisionResult(labels={"idol": 0.9}, embedding=[1.0, 0.0])), SceneVision(sceneId="b", start=1, end=9, vision=VisionResult(labels={"idol": 0.1, "crowd": 0.8}, embedding=[0.0, 1.0]))]
    sh = build_shots(f, [Moment(momentId="m0", start=0.0, end=9.0, reason="x", score=0.5)], vis, None, edge_info=False)[0]
    assert sh.labels["idol"] < 0.3 and sh.labels["crowd"] > 0.6                                       # 1 s of idol in a 9 s window is not "idol 0.9"
