import json, pytest
from aikyam_video import scenes as sc
from aikyam_video.captions import build_cues, lang_cfg, caption_style
from aikyam_video.entities import GazetteerEntityExtractor
from aikyam_video.jobs import InvalidTransition, JobStore
from aikyam_video.events import Event, consume
from aikyam_video.models import Moment, Transcript, TranscriptSegment, Word
from aikyam_video.plan import PlanError, plan_edit, snap_to_moments, validate_plan
from aikyam_video.reframe import crop_box, subject_x
from aikyam_video.models import SceneVision, VisionResult
from aikyam_video.scoring import DEFAULT_WEIGHTS, ScoringConfig, SCORERS, scorer, score
from aikyam_video.storage import LocalStorage


# ---- scene grouping
def test_group_scenes_merges_short_shots():
    assert sc.group_scenes([(0, 2), (2, 3), (3, 9), (9, 10)], 4) == [(0, 10)]   # trailing 1 s joins previous scene
    assert sc.group_scenes([(0, 5), (5, 10)], 4) == [(0, 5), (5, 10)]


def test_group_scenes_contiguous_and_min_len():
    shots = [(a, a + 1.0) for a in range(30)]
    g = sc.group_scenes(shots, 4)
    assert g[0][0] == 0 and g[-1][1] == 30
    assert all(a[1] == b[0] for a, b in zip(g, g[1:]))
    assert all(b - a >= 4 for a, b in g)


# ---- timestamp / plan validation
def _plan(**kw):
    p = {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": 100},
         "outputFormat": "REEL", "durationSeconds": 10, "aspectRatio": "9:16",
         "segments": [{"start": 10, "end": 20, "reason": "aarti", "score": 0.9}],
         "captions": {"enabled": False, "language": "kn"}, "overlays": {}, "transitions": {"type": "cut", "durationSeconds": 0},
         "audio": {"preserveOriginal": True}, "thumbnail": {}}
    p.update(kw); return p


def test_valid_plan():
    validate_plan(_plan())


@pytest.mark.parametrize("segs,dur", [
    ([{"start": 90, "end": 105, "reason": "x", "score": .5}], 15),          # past source end
    ([{"start": -1, "end": 9, "reason": "x", "score": .5}], 10),            # negative
    ([{"start": 10, "end": 20, "reason": "x", "score": .5}, {"start": 15, "end": 25, "reason": "x", "score": .5}], 20),  # overlap
    ([{"start": 0, "end": 70, "reason": "x", "score": .5}], 70),            # > 60 s
    ([{"start": 10, "end": 20, "reason": "x", "score": .5}], 12),           # duration mismatch
])
def test_invalid_plans_rejected(segs, dur):
    with pytest.raises(PlanError):
        validate_plan(_plan(segments=segs, durationSeconds=dur))


def test_schema_rejects_unknown_field():
    with pytest.raises(PlanError):
        validate_plan(_plan(bogus=1))


def test_snap_to_moments_drops_hallucinated_timestamps():
    ms = [Moment(momentId="m0", start=10, end=30, reason="aarti", score=.9)]
    out = snap_to_moments([{"start": 12, "end": 20, "reason": "a"}, {"start": 500, "end": 510, "reason": "made up"},
                           {"start": 25, "end": 40, "reason": "overrun"}], ms)
    assert [(s["start"], s["end"]) for s in out] == [(12, 20), (25, 30)]


# ---- planner determinism and budget
def test_planner_deterministic_and_budgeted():
    ms = [Moment(momentId=f"m{i}", start=i * 30, end=i * 30 + 20, reason="aarti", score=1 - i / 10) for i in range(6)]
    tr = Transcript(language="en")
    a = plan_edit("v", "/x", 300, ms, tr, [], {}, target_s=45)
    b = plan_edit("v", "/x", 300, ms, tr, [], {}, target_s=45)
    assert a == b and a["durationSeconds"] <= 45 and a["captions"]["enabled"] is False


# ---- scoring is configurable/pluggable
def test_scoring_weights_configurable_and_pluggable(tmp_path):
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
    f = tmp_path / "w.json"; f.write_text(json.dumps({"visualImportance": 0.0, "custom": 1.0}))
    with pytest.raises(ValueError):
        ScoringConfig.load(str(f))       # unknown scorer
    @scorer("custom")
    def _c(ctx, a, b): return 0.8
    try:
        cfg = ScoringConfig.load(str(f))
        total, comps = score(None, 0, 1, ScoringConfig({"custom": 1.0}))
        assert total == 0.8 and comps == {"custom": 0.8}
        assert cfg.weights["visualImportance"] == 0.0
    finally:
        SCORERS.pop("custom")


# ---- crop calculation
def test_crop_916_from_169_follows_subject():
    x, y, w, h = crop_box(1920, 1080, 9 / 16, 0.5)
    assert (w, h, y) == (608, 1080, 0) and abs(x + w / 2 - 960) <= 2
    x, *_ = crop_box(1920, 1080, 9 / 16, 0.9)
    assert x > 960
    x, _, w, _ = crop_box(1920, 1080, 9 / 16, 1.0)
    assert x + w == 1920 or x + w == 1919                 # clamped to frame
    assert crop_box(1920, 1080, 9 / 16, 0.0)[0] == 0


def test_crop_even_and_wider_target():
    x, y, w, h = crop_box(1080, 1920, 16 / 9, 0.5)        # portrait source -> landscape: crop vertically
    assert (x, w) == (0, 1080) and h % 2 == 0 and y >= 0


def test_subject_x_weighted_median():
    v = lambda x: VisionResult(saliency_x=x)
    sv = [SceneVision(sceneId="a", start=0, end=2, vision=v(0.2)), SceneVision(sceneId="b", start=2, end=10, vision=v(0.8))]
    assert subject_x(sv, 0, 10) == 0.8 and subject_x(sv, 50, 60) == 0.5


# ---- caption timing
def _tr():
    w = lambda t, a, b: Word(start=a, end=b, text=t)
    return Transcript(language="en", segments=[
        TranscriptSegment(start=10, end=14, text="om namah shivaya om", words=[w("om", 10, 10.5), w("namah", 10.6, 11.2), w("shivaya", 11.3, 12.4), w("om", 13, 13.5)]),
        TranscriptSegment(start=40, end=44, text="jai ganesha", words=[w("jai", 40, 41), w("ganesha", 41.2, 43)])])


def test_caption_sentence_timing_shifted_to_output_timeline():
    segs = [{"start": 12, "end": 14}, {"start": 40, "end": 44}]
    cues = build_cues(_tr(), segs, "sentence")
    assert cues[0]["start"] == 0 and cues[0]["end"] == 2          # clipped to segment, shifted to 0
    assert cues[1]["start"] == 2 and cues[1]["end"] == 6          # second segment starts at offset 2
    assert all(0 <= c["start"] < c["end"] <= 6 for c in cues)


def test_caption_word_mode_groups_and_stays_in_bounds():
    cues = build_cues(_tr(), [{"start": 10, "end": 14}], "word")
    assert sum(len(c["words"]) for c in cues) == 4 and all(len(c["words"]) <= 4 for c in cues)
    assert cues[0]["words"][0]["start"] == 0


def test_language_config_extensible_without_code():
    assert lang_cfg("kn")["font"] == "Noto Sans Kannada"
    assert lang_cfg("xx")["font"]                                   # unknown language falls back to default
    assert "Noto Sans Devanagari" in caption_style("C", 1080, 1920, "hi", {})


# ---- entities
def test_entity_resolution_to_ids():
    ex = GazetteerEntityExtractor()
    r = {e.entityId: e for e in ex.extract("The evening aarti for Chamundeshwari begins")}
    assert r["deity_45"].entityType == "DEITY" and r["ritual_12"].confidence >= 0.9
    assert any(e.entityId == "deity_45" for e in ex.extract("ಚಾಮುಂಡೇಶ್ವರಿ ದೇವಿಗೆ ಆರತಿ"))   # Kannada script


# ---- storage abstraction
def test_local_storage_roundtrip_and_traversal(tmp_path):
    s = LocalStorage(str(tmp_path / "s"))
    src = tmp_path / "a.txt"; src.write_text("hi")
    s.upload(str(src), "t/a.txt"); assert s.exists("t/a.txt")
    s.download("t/a.txt", str(tmp_path / "b.txt")); assert (tmp_path / "b.txt").read_text() == "hi"
    s.delete("t/a.txt"); assert not s.exists("t/a.txt")
    with pytest.raises(ValueError):
        s.upload(str(src), "../evil.txt")


def test_s3_storage_uses_bucket_and_presigns():
    from aikyam_video.storage import S3Storage
    class C:
        calls = []
        def generate_presigned_url(self, op, Params, ExpiresIn): self.calls.append((op, Params, ExpiresIn)); return "https://signed"
        def upload_file(self, *a, **k): self.calls.append(("up", a, k))
        def delete_object(self, **k): self.calls.append(("del", k))
    c = C(); s = S3Storage("bkt", client=c)
    assert s.signed_url("k", 60) == "https://signed" and c.calls[0] == ("get_object", {"Bucket": "bkt", "Key": "k"}, 60)
    assert s.signed_url("k", upload=True) and c.calls[1][0] == "put_object"


# ---- job state machine + idempotency
def test_state_machine_and_idempotency(tmp_path):
    st = JobStore(f"sqlite:///{tmp_path}/t.db")
    st.create_media("m1", "ten", "k")
    j = st.create_job("m1", "ten", "key1"); assert st.create_job("m1", "ten", "key1")["job_id"] == j["job_id"]
    for s in ["PROCESSING", "ANALYZED", "HIGHLIGHTS_READY", "EDIT_PLAN_READY", "RENDERING", "READY", "PUBLISHED"]:
        st.transition(j["job_id"], s)
    with pytest.raises(InvalidTransition):
        st.transition(j["job_id"], "PROCESSING")
    j2 = st.create_job("m1", "ten", "key2")
    st.transition(j2["job_id"], "PROCESSING"); st.transition(j2["job_id"], "FAILED", "boom")
    st.transition(j2["job_id"], "RETRYING"); st.transition(j2["job_id"], "PROCESSING")
    assert st.get_job(j2["job_id"])["attempt"] == 2
    with pytest.raises(InvalidTransition):
        st.transition(j2["job_id"], "READY")


def test_event_consumed_once(tmp_path):
    st = JobStore(f"sqlite:///{tmp_path}/e.db"); seen = []
    ev = Event(type="RenderRequested", jobId="j", mediaId="m", tenantId="t")
    assert consume(st, ev, seen.append) and not consume(st, ev, seen.append) and len(seen) == 1
    assert consume(st, ev.model_copy(update={"attempt": 2}), seen.append)   # retry attempt is a distinct event


def test_scene_times_rescaled_when_container_fps_is_bogus(monkeypatch):
    """WebM reports 120 fps for ~29 fps footage: scenedetect times end early; detect_shots must rescale to duration."""
    class SM:
        def add_detector(self, d): pass
        def detect_scenes(self, v): pass
        def get_scene_list(self):
            f = lambda s: type("T", (), {"get_seconds": lambda self: s})()
            return [(f(0), f(4)), (f(4), f(18))]
    monkeypatch.setattr(sc, "SceneManager", SM)
    monkeypatch.setattr(sc, "open_video", lambda p: type("V", (), {"frame_rate": 120})())
    monkeypatch.setattr(sc.ff, "probe", lambda p: type("I", (), {"duration": 72.0})())
    shots = sc.detect_shots("x")
    assert shots[-1][1] == pytest.approx(72.0) and shots[0] == (0.0, pytest.approx(16.0))


def test_candidates_never_exceed_source_after_rounding():
    from types import SimpleNamespace as N
    from aikyam_video.highlights import generate_candidates
    from aikyam_video.audio import AudioProfile
    import numpy as np
    z = np.zeros(0)
    ctx = N(info=N(duration=20.017), scenes=[N(sceneId="s", start=0.0, end=20.017)], audio=AudioProfile(0.5, z, z.astype(bool), z.astype(bool), z),
            segment_entities=[])
    assert all(0 <= a < b <= 20.017 for a, b in generate_candidates(ctx))


def test_planner_segments_never_exceed_source_after_rounding():
    """Regression (found by live mode): a moment ending at 30.0163 must not become a segment ending at 30.02."""
    from aikyam_video.plan import plan_edit, validate_plan
    ms = [Moment(momentId="m0", start=20.0, end=30.016341, reason="devotees", score=0.4)]
    p = plan_edit("v", "/x", 30.016341, ms, Transcript(language="en"), [], {})
    validate_plan(p, 30.016341)
    assert p["segments"][0]["end"] <= 30.016341


def test_s3_signed_url_uses_public_host_and_never_leaks_secret():
    from aikyam_video.storage import S3Storage
    s = S3Storage("bkt", endpoint_url="http://minio:9000", region="us-east-1", access_key="AKIATEST", secret_key="SECRETSECRET",
                  public_endpoint_url="https://media.example.org")
    u = s.signed_url("t/v/reel.mp4", 600)
    assert u.startswith("https://media.example.org/bkt/t/v/reel.mp4?") and "X-Amz-Signature=" in u and "X-Amz-Expires=600" in u
    assert "minio:9000" not in u and "SECRETSECRET" not in u                  # credentials never appear in what clients get
    assert "AKIATEST" in u                                                     # the (public) access key id is part of SigV4, the secret is not


def test_transcribing_a_video_without_audio_returns_an_empty_transcript(tmp_path):
    import subprocess
    from aikyam_video.transcribe import FasterWhisperProvider
    f = str(tmp_path / "silent.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x90:r=10:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", f], check=True)
    t = FasterWhisperProvider().transcribe(f); assert t.segments == [] and t.language


def test_a_silent_video_goes_through_audio_and_highlight_analysis(tmp_path):
    import subprocess
    from aikyam_video import ffmpeg as ff
    from aikyam_video.audio import profile
    f = str(tmp_path / "silent.mp4"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=160x90:r=10:d=2", "-c:v", "libx264", "-pix_fmt", "yuv420p", f], check=True)
    assert len(ff.extract_audio_pcm(f)) == 0
    p = profile(f); assert len(p.rms_db) == 0 and p.silence_ratio(0, 2) >= 0


def test_audio_only_files_still_decode(tmp_path):
    import subprocess
    from aikyam_video import ffmpeg as ff
    f = str(tmp_path / "a.mp3"); subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=1", f], check=True)
    assert ff.has_audio_stream(f) and len(ff.extract_audio_pcm(f)) > 8000


def test_hdr_hlg_footage_is_tone_mapped_to_sdr_when_rendered(tmp_path):
    """A phone HLG (BT.2020) clip must not come out flat next to SDR clips: the render tone-maps it, so the same pixel data looks different (and not clipped) vs an SDR-tagged copy."""
    import subprocess
    from types import SimpleNamespace as N
    import numpy as np
    from aikyam_video import ffmpeg as ff
    from aikyam_video.render import render_format
    def mk(name, tags):
        f = str(tmp_path / name)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=2", "-pix_fmt", "yuv420p10le", "-c:v", "libx265", "-x265-params", "log-level=none:" + tags, f], check=True); return f
    hlg = mk("hlg.mp4", "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc"); sdr = mk("sdr.mp4", "colorprim=bt709:transfer=bt709:colormatrix=bt709")
    assert ff.probe(hlg).hdr and not ff.probe(sdr).hdr
    def render(src, tag):
        p = {"schemaVersion": 1, "source": {"videoId": "v", "path": src, "durationSeconds": 2}, "outputFormat": "REEL", "aspectRatio": "9:16", "durationSeconds": 2, "segments": [{"start": 0, "end": 2, "reason": "x", "score": .5}],
             "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"}, "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True, "music": {"enabled": False}}, "thumbnail": {}}
        (tmp_path / tag).mkdir(); out = render_format(p, src, str(tmp_path / f"{tag}.mp4"), "reel", str(tmp_path / tag), mix=N(pcm=np.zeros((2 * 48000, 2), np.float32)))
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", out, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        return np.frombuffer(raw, np.uint8).astype(float)
    a, b = render(hlg, "h"), render(sdr, "s")
    assert abs(a.mean() - b.mean()) > 8 and (a >= 254).mean() <= (b >= 254).mean() + 0.02          # a real conversion happened, and it added no clipping
