import json, os, subprocess, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace as N
import numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.audio import AudioProfile
from aikyam_video.jobs import JobStore
from aikyam_video.models import EntityRef, Moment, SceneVision, Transcript, TranscriptSegment, VisionResult
from aikyam_video.options import Profile


# ------------------------------------------------------------------ helpers
def moving_dot_video(path, w=640, h=360, d=6):
    """White square moving left->right on grey, plus a tone: ground truth for subject tracking."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x202020:s={w}x{h}:d={d}:r=25", "-f", "lavfi", "-i", "color=c=white:s=80x80:d=%d:r=25" % d,
                    "-f", "lavfi", "-i", f"sine=f=440:d={d}", "-filter_complex", f"[0][1]overlay=x='t*{(w - 80) / d}':y=140:eval=frame[v]", "-map", "[v]", "-map", "2:a",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path], check=True)


class Serve:
    """Tiny HTTP server for webhook/KG tests. handler(body_dict, path) -> (status, dict)."""
    def __init__(self, handler):
        outer = self; self.calls = []
        class H(BaseHTTPRequestHandler):
            def do_POST(s):
                body = json.loads(s.rfile.read(int(s.headers["Content-Length"])) or b"{}"); outer.calls.append((s.path, body, dict(s.headers)))
                st, out = handler(body, s.path); data = json.dumps(out).encode()
                s.send_response(st); s.send_header("Content-Length", str(len(data))); s.end_headers(); s.wfile.write(data)
            def log_message(*a): pass
        self.srv = HTTPServer(("127.0.0.1", 0), H); self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
    def close(self): self.srv.shutdown()


# ------------------------------------------------------------------ reframing: tracking
def test_tracking_follows_moving_subject(tmp_path):
    from aikyam_video.reframe import MAX_PAN, track_subject
    v = str(tmp_path / "dot.mp4"); moving_dot_video(v)
    path = track_subject(v, 0.0, 6.0)
    xs = [x for _, x in path]
    assert len(path) >= 10 and xs[-1] > xs[0] + 0.2               # camera pans right with the subject
    assert all(b >= a - 0.03 for a, b in zip(xs, xs[1:]))          # monotone: no jitter
    assert all(abs(b - a) <= MAX_PAN * 0.5 + 1e-6 for a, b in zip(xs, xs[1:]))   # pan speed limit


def test_tracked_crop_renders_and_moves(tmp_path):
    """The crop expression is valid ffmpeg and the output frame content shifts with the path."""
    from aikyam_video.reframe import path_expr, track_subject
    v = str(tmp_path / "dot.mp4"); moving_dot_video(v, w=1280, h=720)
    path = track_subject(v, 0.0, 6.0)
    ce = 404
    out = str(tmp_path / "o.mp4")
    ff.run(["-i", v, "-vf", f"crop={ce}:720:x={path_expr(path, 1280, ce)}:y=0", "-t", "6", out])
    assert ff.probe(out).width == ce
    # the white square stays inside the tracked window at both ends of the clip
    for t in (0.5, 5.5):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", out, "-frames:v", "1", "-vf", "format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        assert (np.frombuffer(raw, np.uint8) > 200).sum() > 500


# ------------------------------------------------------------------ audio profile + moderation gating
def test_audio_profile_roundtrip_and_top_event(tmp_path):
    n = 20
    ev = {"bell": np.zeros(n), "bhajan": np.r_[np.zeros(10), np.ones(10) * 0.9]}
    p = AudioProfile(0.5, np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), ev)
    p.save(str(tmp_path / "a.npz")); q = AudioProfile.load(str(tmp_path / "a.npz"))
    assert q.top_event(5.0, 9.5) == "bhajan" and q.top_event(0, 4) is None and q.event_score("bhajan", 5, 10) > 0.8


def _ctx(vision, dur=100):
    from aikyam_video.highlights import Ctx
    n = int(dur / 0.5); z = np.zeros(n)
    return Ctx("x", N(duration=dur, has_audio=True), [], vision, Transcript(language="en"),
               AudioProfile(0.5, z - 20, z.astype(bool), z.astype(bool), z + 0.5, {}))


def test_moderation_flagged_clip_is_rejected():
    from aikyam_video.highlights import validate_clip
    bad = SceneVision(sceneId="s0", start=0, end=30, vision=VisionResult(moderation={"graphic_violence": 0.9}))
    ok = SceneVision(sceneId="s1", start=30, end=60, vision=VisionResult(moderation={"graphic_violence": 0.2}))
    assert validate_clip(_ctx([bad, ok]), 5, 15, []) == "moderation_flagged"
    assert validate_clip(_ctx([bad, ok]), 35, 45, []) is None


def test_personalization_boosts_preferred_label():
    from aikyam_video.highlights import _personalize
    sv = SceneVision(sceneId="s", start=0, end=30, vision=VisionResult(labels={"aarti": 0.9}, deities={"deity_2": 0.8}))
    ctx = _ctx([sv]); base = 0.5
    assert _personalize(ctx, 0, 10, base) == base                              # no profile: unchanged
    ctx.profile = Profile(preferred_labels={"aarti": 2.0}); a = _personalize(ctx, 0, 10, base)
    ctx.profile = Profile(preferred_labels={"procession": 2.0}); b = _personalize(ctx, 0, 10, base)
    ctx.profile = Profile(preferred_labels={"aarti": 2.0}, preferred_deities=["deity_2"]); c = _personalize(ctx, 0, 10, base)
    assert c > a > base and b == base


def test_audio_score_uses_events_when_tagger_present():
    from aikyam_video.scoring import audio
    n = 40; z = np.zeros(n)
    quiet = _ctx([]); quiet.audio = AudioProfile(0.5, z - 30, z.astype(bool), z.astype(bool), z + 0.1, {"bhajan": z})
    loud = _ctx([]); loud.audio = AudioProfile(0.5, z - 30, z.astype(bool), z.astype(bool), z + 0.1, {"bhajan": z + 0.9})
    assert audio(loud, 0, 10) > audio(quiet, 0, 10) + 0.3


# ------------------------------------------------------------------ Knowledge Graph
def test_kg_extractor_uses_kg_then_falls_back():
    from aikyam_video.kg import KGClient, KGEntityExtractor
    srv = Serve(lambda b, p: (200, {"matches": [{"text": "chamundi", "entityId": "deity_45", "entityType": "DEITY", "name": "Chamundeshwari", "confidence": 0.93},
                                               {"text": "navaratri utsav", "entityId": "festival_777", "entityType": "FESTIVAL", "name": "Navaratri Utsav", "confidence": 0.8}]}))
    try:
        ex = KGEntityExtractor(KGClient(srv.url, "tok"))
        r = {e.entityId: e for e in ex.extract("Chamundi devi at the Navaratri Utsav", "en")}
        assert r["deity_45"].name == "Chamundeshwari" and r["festival_777"].entityType == "FESTIVAL"   # id unknown to the seed catalog
        assert srv.calls[0][0] == "/v1/entities/resolve" and srv.calls[0][2]["Authorization"] == "Bearer tok"
        assert "navaratri utsav" in srv.calls[0][1]["texts"]
    finally:
        srv.close()
    down = KGEntityExtractor(KGClient("http://127.0.0.1:9", timeout=0.5))               # nothing listening
    assert any(e.entityId == "ritual_12" for e in down.extract("the evening aarti", "en"))   # falls back to local catalog


# ------------------------------------------------------------------ semantic library
def _art(tmp, emb_a, dur=60.0):
    os.makedirs(tmp, exist_ok=True)
    scenes = [{"sceneId": f"scene_{i}", "start": i * dur / len(emb_a), "end": (i + 1) * dur / len(emb_a), "keyframe": ""} for i in range(len(emb_a))]
    vis = [{"sceneId": s["sceneId"], "start": s["start"], "end": s["end"], "vision": {"labels": {"aarti": 0.9}}} for s in scenes]
    json.dump(scenes, open(f"{tmp}/scenes.json", "w")); json.dump(vis, open(f"{tmp}/vision.json", "w"))
    json.dump({s["sceneId"]: list(map(float, e)) for s, e in zip(scenes, emb_a)}, open(f"{tmp}/embeddings.json", "w"))


def _unit(*v):
    v = np.array(v, float); return v / np.linalg.norm(v)


def test_library_search_duplicates_and_similar(tmp_path):
    from aikyam_video import library
    st = JobStore(f"sqlite:///{tmp_path}/l.db")
    for m in ("m1", "m2", "m3", "m4"): st.create_media(m, "t", "k")
    _art(str(tmp_path / "a1"), [_unit(1, 0, 0), _unit(0, 1, 0)]); _art(str(tmp_path / "a2"), [_unit(1, 0, 0), _unit(0, 1, 0)])       # a2 = re-upload of a1
    _art(str(tmp_path / "a3"), [_unit(0, 0, 1), _unit(0, 0.1, 1)]); _art(str(tmp_path / "a4"), [_unit(1, 0, 0.05), _unit(0, 1, 0.05)], dur=300)  # a4: similar but 5x longer
    r1 = library.index_media(st, "m1", "t", str(tmp_path / "a1")); assert r1["duplicateOf"] is None
    assert library.index_media(st, "m2", "t", str(tmp_path / "a2"))["duplicateOf"] == "m1"          # near-duplicate flagged
    assert library.index_media(st, "m3", "t", str(tmp_path / "a3"))["duplicateOf"] is None
    assert library.index_media(st, "m4", "t", str(tmp_path / "a4"))["duplicateOf"] is None           # different duration: not a duplicate
    emb = N(embed_text=lambda qs: np.array([_unit(0, 0, 1)]))                                        # "query" points at m3's space
    top = library.search(st, emb, "t", "anything", 2)
    assert top[0]["mediaId"] == "m3" and top[0]["score"] > 0.9
    assert library.search(st, emb, "other-tenant", "x") == []                                        # tenant isolation
    assert library.similar(st, "m1", "t")[0]["mediaId"] in ("m2", "m4") and len(library.video_vector(st, "m1")) == 3


# ------------------------------------------------------------------ LLM planner
def _moments():
    return [Moment(momentId="m0", start=10, end=30, reason="aarti", score=0.9), Moment(momentId="m1", start=60, end=90, reason="deity", score=0.8)]


def _plan_kw(**kw):
    return dict(video_id="v", path="/x", duration=120.0, moments=_moments(), transcript=Transcript(language="en"), vision=[],
                entities={"TEMPLE": [EntityRef(text="t", name="Chamundeshwari Temple", entityType="TEMPLE", entityId="temple_123", confidence=1)]}, target_s=45, **kw)


class FakeLLM:
    name = "fake"
    def __init__(self, out=None, exc=None): self.out, self.exc, self.seen = out, exc, None
    def propose(self, system, user, schema):
        self.seen = (system, user, schema)
        if self.exc: raise self.exc
        return self.out


def test_llm_plan_snaps_hallucinated_timestamps_and_records_mode():
    from aikyam_video.planner_llm import plan_with_llm
    from aikyam_video.plan import validate_plan
    llm = FakeLLM({"segments": [{"momentId": "m0", "start": 12, "end": 24, "reason": "peak of the aarti"},
                                {"momentId": "zz", "start": 500, "end": 520, "reason": "hallucinated: beyond the video"},
                                {"momentId": "m1", "start": 55, "end": 100, "reason": "overruns its moment"}],
                   "transition": "fade", "overlayTitle": "Evening Aarti"})
    p = plan_with_llm(**_plan_kw(llm=llm))
    validate_plan(p, 120.0)
    assert p["planner"]["mode"] == "llm" and p["transitions"]["type"] == "fade"
    segs = [(s["start"], s["end"]) for s in p["segments"]]
    assert segs[0] == (12, 24) and segs[1][0] >= 60 and segs[1][1] <= 90        # clamped into moment m1; the fake "zz" was dropped
    assert len(segs) == 2 and p["durationSeconds"] <= 45
    assert "m0" in llm.seen[1] and "Chamundeshwari" in llm.seen[1]


@pytest.mark.parametrize("llm", [FakeLLM(exc=RuntimeError("API down")), FakeLLM({"segments": [{"momentId": "nope", "start": 1, "end": 5, "reason": "x"}], "transition": "cut", "overlayTitle": ""})])
def test_llm_failure_falls_back_to_deterministic(llm):
    from aikyam_video.planner_llm import plan_with_llm
    p = plan_with_llm(**_plan_kw(llm=llm))
    assert p["planner"]["mode"] == "deterministic-fallback" and p["segments"]


def test_anthropic_llm_request_shape_and_refusal():
    from aikyam_video.planner_llm import AnthropicLLM, PROPOSAL_SCHEMA
    calls = []
    class Msgs:
        def __init__(self, r): self.r = r
        def create(self, **kw): calls.append(kw); return self.r
    mk = lambda r: N(beta=N(messages=Msgs(r)))
    ok = N(stop_reason="end_turn", content=[N(type="text", text='{"segments": [], "transition": "cut", "overlayTitle": ""}')])
    out = AnthropicLLM(client=mk(ok)).propose("sys", "user", PROPOSAL_SCHEMA)
    k = calls[0]
    assert out["transition"] == "cut" and k["model"] == "claude-opus-5" and k["thinking"] == {"type": "adaptive"}
    assert k["output_config"]["format"]["schema"] is PROPOSAL_SCHEMA and "server-side-fallback-2026-07-01" in k["betas"]
    assert k["extra_body"] == {"fallbacks": "default"}
    with pytest.raises(RuntimeError, match="refused"):
        AnthropicLLM(client=mk(N(stop_reason="refusal", stop_details=N(category="cyber"), content=[]))).propose("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(RuntimeError, match="truncated"):
        AnthropicLLM(client=mk(N(stop_reason="max_tokens", content=[]))).propose("s", "u", PROPOSAL_SCHEMA)


# ------------------------------------------------------------------ templates, captions in more languages, music
def test_festival_template_selected_from_festival_id():
    from aikyam_video.plan import plan_edit
    ents = {"FESTIVAL": [EntityRef(text="Dasara", name="Dasara", entityType="FESTIVAL", entityId="festival_9", confidence=1)],
            "TEMPLE": [EntityRef(text="t", name="Chamundeshwari Temple", entityType="TEMPLE", entityId="temple_123", confidence=1)]}
    p = plan_edit("v", "/x", 120, _moments(), Transcript(language="en"), [], ents)
    assert p["overlays"]["template"] == "dasara" and p["source"]["festivalId"] == "festival_9"
    from aikyam_video.render import build_ass
    assert "&H0030A0FF" in build_ass(p, 1080, 1920)                       # Dasara accent colour applied to the title line


@pytest.mark.parametrize("lang,text,font", [("kn", "ಶ್ರೀ ಚಾಮುಂಡೇಶ್ವರಿ ದೇವಿ ಆರತಿ", "Noto Sans Kannada"), ("hi", "श्री गणेश आरती शुरू", "Noto Sans Devanagari"),
                                            ("te", "శ్రీ వేంకటేశ్వర స్వామి", "Noto Sans Telugu"), ("ta", "ஸ்ரீ முருகன் ஆரத்தி", "Noto Sans Tamil")])
def test_indic_captions_render_visible_text(tmp_path, lang, text, font):
    if font.lower() not in subprocess.run(["fc-list"], capture_output=True, text=True).stdout.lower():
        pytest.skip(f"{font} not installed")
    from aikyam_video.render import build_ass
    plan = {"captions": {"enabled": True, "language": lang, "mode": "sentence", "style": {}, "cues": [{"start": 0, "end": 2, "text": text, "words": []}]},
            "overlays": {"template": "divine_moment"}, "durationSeconds": 2}
    ass = tmp_path / "c.ass"; ass.write_text(build_ass(plan, 1080, 1920, text_brand=False), encoding="utf-8")
    assert font in ass.read_text(encoding="utf-8")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=2", "-vf", f"ass={ass}", "-frames:v", "1", "-ss", "1",
                          "-vf", f"ass={ass},crop=iw:ih*0.2:0:ih*0.6,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    assert (np.frombuffer(raw, np.uint8) > 200).sum() > 800             # glyphs drawn (not tofu-free proof, but text pixels exist)


def _plan_with_music(track):
    return {"schemaVersion": 1, "source": {"videoId": "v", "path": "/x", "durationSeconds": 6}, "outputFormat": "REEL", "durationSeconds": 4, "aspectRatio": "9:16",
            "segments": [{"start": 1, "end": 5, "reason": "x", "score": 0.9}], "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"},
            "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True, "normalize": True, "music": {"enabled": True, "trackId": track}},
            "thumbnail": {}}


def test_music_is_mixed_and_ducked_under_original(tmp_path, monkeypatch):
    from aikyam_video.render import render_format
    v = str(tmp_path / "src.mp4"); moving_dot_video(v)                  # 440 Hz tone as the "original audio"
    lib = tmp_path / "music"; lib.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=1200:d=2", str(lib / "royalty_safe_01.wav")], check=True)   # short: must loop
    (lib / "library.json").write_text(json.dumps({"tracks": [{"id": "royalty_safe_01", "title": "t", "file": "royalty_safe_01.wav", "licence": "AIKYAM-OWNED",
                                                              "licenceVerified": True, "energy": 0.3, "tags": {}}]}))
    monkeypatch.setenv("MUSIC_LIBRARY_DIR", str(lib))
    out = str(tmp_path / "o.mp4"); render_format(_plan_with_music("royalty_safe_01"), v, out, "square", str(tmp_path))
    i = ff.probe(out); assert i.has_audio and abs(i.duration - 4) < 0.3
    pcm = ff.extract_audio_pcm(out, 16000); spec = np.abs(np.fft.rfft(pcm)); f = np.fft.rfftfreq(len(pcm), 1 / 16000)
    e440, e1200 = spec[(f > 430) & (f < 450)].sum(), spec[(f > 1190) & (f < 1210)].sum()
    assert e440 > 0 and e1200 > 0 and e1200 < e440                     # both present; music ducked below the original


def test_music_track_must_be_in_library(tmp_path, monkeypatch):
    from aikyam_video.render import music_path
    monkeypatch.setenv("MUSIC_LIBRARY_DIR", str(tmp_path))
    for bad in ("nope", "../etc/passwd", "a/b"):
        with pytest.raises(FileNotFoundError): music_path(bad)


# ------------------------------------------------------------------ publishing (unit)
def test_webhook_publisher_posts_bearer_and_raises_on_error():
    from aikyam_video.publish import WebhookPublisher
    srv = Serve(lambda b, p: (200, {"ok": True}))
    try:
        assert WebhookPublisher(srv.url, "tok").publish({"mediaId": "m"})["status"] == 200
        assert srv.calls[0][2]["Authorization"] == "Bearer tok" and srv.calls[0][1] == {"mediaId": "m"}
    finally: srv.close()
    bad = Serve(lambda b, p: (500, {}))
    try:
        with pytest.raises(Exception): WebhookPublisher(bad.url).publish({})
    finally: bad.close()


def test_auto_publish_policy():
    from aikyam_video.publish import auto_publish_ok
    ok = {"segments": [{"score": 0.6}], "durationSeconds": 20}
    assert auto_publish_ok(ok) and not auto_publish_ok({**ok, "segments": [{"score": 0.2}]}) and not auto_publish_ok({**ok, "durationSeconds": 2}) \
        and not auto_publish_ok({**ok, "segments": []})


def test_anthropic_llm_falls_back_to_a_plain_json_request_when_the_structured_shape_is_rejected():
    from aikyam_video.planner_llm import AnthropicLLM, DIRECTOR_SCHEMA
    class BadRequestError(Exception): pass
    seen = []
    class Beta:
        def create(self, **kw): seen.append("beta"); raise BadRequestError("unknown parameter: output_config")
    class Plain:
        def create(self, **kw): seen.append(("plain", "thinking" in kw, "betas" in kw)); return N(stop_reason="end_turn", content=[N(type="text", text='Sure:\n{"clips": [], "overlayTitle": "x"}')])
    llm = AnthropicLLM(client=N(beta=N(messages=Beta()), messages=Plain()))
    assert llm.propose("s", "u", DIRECTOR_SCHEMA) == {"clips": [], "overlayTitle": "x"} and seen == ["beta", ("plain", False, False)]
    class Other(Exception): pass
    class Beta2:
        def create(self, **kw): raise Other("network down")
    with pytest.raises(Other): AnthropicLLM(client=N(beta=N(messages=Beta2()), messages=Plain())).propose("s", "u", DIRECTOR_SCHEMA)         # only request-shape errors trigger the fallback
