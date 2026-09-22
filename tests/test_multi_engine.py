"""Creative engine on MIXED inputs (2 videos + 2 images), real FFmpeg, no ML models: distinct footage, hand-made embeddings/moments.
Checks the plan contract (v3 assets), that stills get subject-aware Ken Burns, that the render matches the plan, and that sources never clash."""
import subprocess
import cv2, numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.creative import engine as E
from aikyam_video.creative.shots import image_shot
from aikyam_video.models import Moment, SceneVision, Transcript, VisionResult
from aikyam_video.options import Options
from aikyam_video.plan import validate_plan


def vid(path, src, d=20):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{src}=s=1280x720:r=25", "-f", "lavfi", "-i", "sine=f=300:d=%d" % d, "-t", str(d),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path], check=True); return path


def img(path, seed):
    r = np.random.RandomState(seed); im = cv2.GaussianBlur(r.randint(0, 255, (900, 1500, 3), np.uint8), (0, 0), 6)
    cv2.circle(im, (1050, 380), 90, (30, 60, 240), -1); cv2.imwrite(path, im); return path


def emb(seed, n=32): v = np.random.RandomState(seed).randn(n); return (v / np.linalg.norm(v)).tolist()


def source(sid, path, seed, lab):
    info = ff.probe(path); vis = []; moms = []
    for k in range(4):
        a, b = k * 5.0, (k + 1) * 5.0
        vis.append(SceneVision(sceneId=f"s{k}", start=a, end=b, vision=VisionResult(labels={lab: 0.8}, embedding=emb(seed * 10 + k))))
        moms.append(Moment(momentId=f"{sid}m{k}", start=a, end=b, reason=lab, score=0.5 + 0.05 * k))
    return E.Source(sid, path, "video", info, moms, vis, None, Transcript(language="en"))


@pytest.fixture(scope="module")
def mixctx(tmp_path_factory):
    d = tmp_path_factory.mktemp("mx")
    v1, v2 = vid(str(d / "a.mp4"), "mandelbrot"), vid(str(d / "b.mp4"), "testsrc2")
    i1, i2 = img(str(d / "i1.png"), 1), img(str(d / "i2.png"), 2)
    srcs = [source("v1", v1, 1, "aarti"), source("v2", v2, 2, "devotees"), E.Source("i1", i1, "image", None, shot=image_shot(i1, "i1")), E.Source("i2", i2, "image", None, shot=image_shot(i2, "i2"))]
    for s in srcs[2:]: s.shot.embedding = emb(int(s.id[1:]) * 100); s.shot.labels = {"deity": 0.7}
    o = Options(music="off", qc=True, formats=["reel"])
    return E.Ctx(v1, srcs[0].info, srcs[0].moments, srcs[0].vision, None, srcs[0].transcript, {}, o, "mix", sources=srcs), d


@pytest.fixture(scope="module")
def built(mixctx):
    ctx, d = mixctx; v1, o = ctx.src, ctx.o
    plans = E.build_plans(ctx); out = str(d / "out")
    import os; os.makedirs(out)
    files, qc, final = E.render_reel(plans[0], v1, out, o, {}, ["reel"])
    return plans[0], final, files, qc


def test_plan_is_v3_with_assets_and_uses_more_than_one_asset(built):
    plan, _, _, _ = built
    validate_plan(plan); assert plan["schemaVersion"] == 3
    used = {s["assetId"] for s in plan["segments"]}; assert len(used) >= 3 and {a["id"] for a in plan["assets"]} == used


def test_stills_get_ken_burns_and_are_capped_and_not_adjacent(built):
    plan, _, _, _ = built; segs = plan["segments"]
    imgs = [i for i, s in enumerate(segs) if s["kind"] == "image"]
    assert imgs and len(imgs) <= max(1, -(-len(segs) // 3)) and all(b - a > 1 for a, b in zip(imgs, imgs[1:]))
    for i in imgs:
        m = segs[i]["motion"]; assert m["zoom"][0] != m["zoom"][1] and 2.0 <= segs[i]["end"] - segs[i]["start"] <= 5.0


def test_no_source_window_is_used_twice(built):
    plan, _, _, _ = built; segs = [s for s in plan["segments"] if s["kind"] == "video"]
    for i, a in enumerate(segs):
        for b in segs[i + 1:]:
            assert a["assetId"] != b["assetId"] or a["end"] <= b["start"] or b["end"] <= a["start"]


def test_render_matches_plan_duration_and_loudness(built):
    plan, final, files, qc = built
    assert abs(ff.probe(files["reel"]).duration - final["durationSeconds"]) < 0.25
    assert qc["stats"]["true_peak_db"] <= -1.0 and abs(qc["stats"]["lufs"] + 16) < 2.0
    assert qc["status"] in ("pass", "warn")


def test_ken_burns_respects_resolution_and_face_fit():
    from aikyam_video.creative.stills import motion_for, Z_DRIFT
    hi = motion_for([4000, 3000], [0.7, 0.4, 0.05, 0.06], 9 / 16, 0)                 # big photo, small face: zoom limited by the face rule (Z_MAX)
    assert hi["zoom"][0] == 1.0 and 1.15 <= hi["zoom"][1] <= 1.5 and hi["center"][1][0] > hi["center"][0][0] * 0.9
    lo = motion_for([1500, 900], [0.7, 0.4, 0.0, 0.0], 9 / 16, 0); assert lo["zoom"][1] == Z_DRIFT       # low-res, no face: drift only
    out = motion_for([4000, 3000], [0.7, 0.4, 0.05, 0.06], 9 / 16, 1); assert out["zoom"][0] > out["zoom"][1] == 1.0   # odd index reverses (pull-out)


class _LLM:
    name = "fake"; model = "fake-1"
    def __init__(self, out): self.out = out
    def propose(self, system, user, schema):
        if isinstance(self.out, Exception): raise self.out
        return self.out


def test_llm_director_drives_a_mixed_plan_and_outage_falls_back(mixctx):
    import copy
    ctx, _ = mixctx
    def go(out):
        c = copy.copy(ctx); c.o = copy.copy(ctx.o); c.o.planner = "llm"; c.o.llm = _LLM(out); return E.build_plans(c)[0]
    good = go({"clips": [{"shotId": "v1_v1m0", "role": "OPENING", "seconds": 4, "why": "wide"}, {"shotId": "i1", "role": "REVEAL", "seconds": 3, "why": "still"},
                         {"shotId": "v2_v2m1", "role": "CLIMAX", "seconds": 4, "why": "peak"}], "overlayTitle": "Ganga Aarti"})
    assert good["planner"] == {"mode": "llm-director", "model": "fake-1"} and [s["assetId"] for s in good["segments"]] == ["v1", "i1", "v2"]
    validate_plan(good); assert good["segments"][1]["kind"] == "image" and good["segments"][1]["motion"]
    bad = go(RuntimeError("API down")); assert bad["planner"]["mode"] == "deterministic-fallback" and bad["segments"]


def test_moment_ids_repeated_across_videos_do_not_collide(mixctx):
    """Every video numbers its moments moment_0, moment_1 ...: the pooled shots need unique ids or the story sees them as the same footage."""
    import copy
    ctx = copy.deepcopy(mixctx[0])                                            # never mutate the shared module fixture
    for x in ctx.sources[:2]:
        for m in x.moments: m.momentId = m.momentId.replace(x.id + "m", "moment_")
    ids = [s["shotId"] for s in E.build_plans(ctx)[0]["segments"] if s["kind"] == "video"]
    assert len(ids) == len(set(ids)) and len({i.split("_")[0] for i in ids}) >= 2


def test_a_rhythmic_track_gets_dp_alignment_recorded_in_the_plan(mixctx):
    """Regression: the sync report carries a text field (method) next to numbers; the plan must serialise it."""
    import copy
    ctx, _ = mixctx; c = copy.copy(ctx); c.o = copy.copy(ctx.o); c.o.music = "auto"; c.o.music_track = "aikyam_festive_rhythm"
    p = E.build_plans(c)[0]; m = next(d for d in p["creative"]["decisions"] if d["type"] == "music")
    assert m["track"] == "aikyam_festive_rhythm" and m["sync"]["method"] == "dp" and 0.0 <= m["sync"]["on_beat_after"] <= 1.0
    validate_plan(p)
