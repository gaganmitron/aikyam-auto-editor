"""Platform safe zones (portrait), grapheme-aware text length, template/logo placement. Uses libass itself to measure where text really lands."""
import subprocess
from types import SimpleNamespace as N
import numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.captions import caption_style, retime_cues
from aikyam_video.creative import qc as QC
from aikyam_video.render import build_ass, render_format
from aikyam_video.safezone import glyphs, zones

KN = "ಕ್ಷೇತ್ರ ನಮಸ್ತೇ"


def test_glyphs_count_clusters_not_code_points():
    assert glyphs(KN) == 8 and len(KN.replace(" ", "")) == 13 and glyphs("Om Namah") == 7 and glyphs("") == 0


def test_zones_portrait_vs_other():
    assert zones(1080, 1920) == {"top": 0.12, "bottom": 0.22, "side": 0.08} and zones(1920, 1080)["bottom"] == 0.08 and zones(1080, 1080)["top"] == 0.05


def test_caption_style_margins_follow_the_zone():
    st = caption_style("Cap", 1080, 1920, "en", {}).split(",")
    assert int(st[-2]) == int(1920 * 0.22) and int(st[-4]) == int(1080 * 0.08)                   # MarginV, MarginL
    assert int(caption_style("Cap", 1920, 1080, "en", {}).split(",")[-2]) == int(1080 * 0.08)


def test_reading_speed_uses_clusters():
    cue = [{"start": 0.0, "end": 0.5, "text": KN + " " + KN, "words": []}]
    need = retime_cues(cue, max_cps=17.0, min_dur=0.0, end=10.0)[0]["end"]
    assert abs(need - 16 / 17.0) < 0.01                                                              # 16 clusters, not 26 code points


def plan(text="Om Namah Shivaya " * 4, lang="en"):
    cues = [{"start": 0.5 + 2 * i, "end": 2.3 + 2 * i, "text": text, "words": []} for i in range(3)]
    return {"durationSeconds": 7.0, "captions": {"enabled": True, "language": lang, "mode": "sentence", "cues": cues, "style": {}},
            "overlays": {"template": "ritual_highlight", "temple": "Kashi Vishwanath Temple", "deity": "Lord Shiva", "ritual": "Ganga Aarti"},
            "segments": [{"start": 0, "end": 7, "kind": "video"}], "transitions": {"type": "cut", "durationSeconds": 0}}


@pytest.mark.parametrize("text,lang", [("Om Namah Shivaya " * 4, "en"), ((KN + " ") * 2, "kn")])
def test_captions_and_titles_stay_out_of_platform_zones(tmp_path, text, lang):
    p = plan(text, lang); ass = str(tmp_path / "overlay_reel.ass"); open(ass, "w", encoding="utf-8").write(build_ass(p, 1080, 1920, text_brand=False))
    checks = {c.name: c for c in QC.check_captions(p, ass, 1080, 1920)}
    assert checks["caption_overflow"].status == "pass", checks["caption_overflow"]
    assert checks["overlay_safe_zone"].status == "pass" and checks["caption_overlay_collision"].status == "pass"
    tit = [b for _, b in QC.caption_ink(ass, 1080, 1920, 7.0, "Ovl") if b]; assert tit and min(b[1] for b in tit) >= 0.12 * 1920     # titles start below the top bar


def test_the_old_positions_would_have_failed():
    """The check has teeth: a title at y=7% (the old template) is flagged."""
    p = plan(); ass_txt = build_ass(p, 1080, 1920, text_brand=False).replace(f"\\pos(540,{int(1920 * 0.18)})", f"\\pos(540,{int(1920 * 0.07)})")
    import tempfile, os
    d = tempfile.mkdtemp(); ass = os.path.join(d, "overlay_reel.ass"); open(ass, "w", encoding="utf-8").write(ass_txt)
    assert {c.name: c.status for c in QC.check_captions(p, ass, 1080, 1920)}["overlay_safe_zone"] == "fail"


def test_logo_is_below_the_top_bar_on_a_rendered_reel(tmp_path):
    from aikyam_video.render import LOGO
    if not LOGO.is_file(): pytest.skip("no logo asset")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=0x606060:s=1080x1920:r=25:d=3", "-frames:v", "1", str(tmp_path / "g.png")], check=True)
    pl = {"schemaVersion": 3, "source": {"videoId": "v", "path": str(tmp_path / "g.png"), "durationSeconds": 3}, "outputFormat": "REEL", "aspectRatio": "9:16", "durationSeconds": 3,
          "assets": [{"id": "i1", "kind": "image", "path": str(tmp_path / "g.png")}], "segments": [{"assetId": "i1", "kind": "image", "start": 0, "end": 3, "reason": "x", "score": .5}],
          "captions": {"enabled": False, "language": "en"}, "overlays": {"template": "divine_moment"}, "transitions": {"type": "cut", "durationSeconds": 0},
          "audio": {"preserveOriginal": True, "music": {"enabled": False}}, "thumbnail": {}}
    out = str(tmp_path / "o.mp4"); render_format(pl, "u", out, "reel", str(tmp_path), mix=N(pcm=np.zeros((3 * 48000, 2), np.float32)))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "1", "-i", out, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(1920, 1080, 3).astype(int); ink = (np.abs(f - 0x60).max(axis=2) > 40); rows = np.where(ink.any(axis=1))[0]
    assert len(rows) and rows.min() >= 0.12 * 1920 and rows.max() < 0.30 * 1920                    # the logo sits just under the top bar, in the upper part
    cols = np.where(ink.any(axis=0))[0]; assert cols.max() <= 1080 * (1 - 0.08) + 2                  # and inside the right margin


# ---------------------------------------------------------------- opening title: safe zone, timing, automatic naming
def title_plan(title, subtitle=None, lang="en", template="divine_moment"):
    return {"durationSeconds": 20.0, "captions": {"enabled": False, "language": lang, "mode": "sentence", "cues": [], "style": {}}, "overlays": {"template": template, "title": title, "subtitle": subtitle},
            "segments": [{"start": 0, "end": 20, "kind": "video"}], "transitions": {"type": "cut", "durationSeconds": 0}}


@pytest.mark.parametrize("title,sub,lang", [("Ganga Aarti", "Haridwar", "en"), ("Evening Ganga Aarti at Har Ki Pauri with thousands of devotees", "Uttarakhand, India", "en"), (KN * 2, "ಕಾಶಿ", "kn")])
def test_opening_title_sits_below_the_top_bar_wraps_inside_the_side_margins_and_leaves_after_3_6_seconds(tmp_path, title, sub, lang):
    p = title_plan(title, sub, lang); ass = str(tmp_path / "overlay_reel.ass"); open(ass, "w", encoding="utf-8").write(build_ass(p, 1080, 1920, text_brand=False))
    ink = QC.caption_ink(ass, 1080, 1920, 6.0, "Ovl", fps=4)
    on = [b for t, b in ink if b and t < 3.4]; off = [b for t, b in ink if t > 4.0]
    assert on and all(b is None for b in off)                                                                # visible early, gone afterwards
    assert min(b[1] for b in on) >= 0.12 * 1920 and max(b[3] for b in on) <= 0.5 * 1920                       # below the platform top bar, in the upper part
    assert min(b[0] for b in on) >= 0.08 * 1080 - 2 and max(b[2] for b in on) <= 0.92 * 1080 + 2                # inside the side margins even when long
    assert {c.name: c.status for c in QC.check_captions(p, ass, 1080, 1920)}.get("overlay_safe_zone", "pass") == "pass"


def test_titles_are_named_from_what_is_known_and_nothing_is_invented():
    from aikyam_video.models import EntityRef, Transcript
    from aikyam_video.plan import assemble_plan
    seg = [{"start": 0.0, "end": 5.0, "reason": "x", "score": 0.5}]; tr = {"type": "cut", "durationSeconds": 0}
    known = {"TEMPLE": [EntityRef(text="t", name="Kashi Vishwanath Temple", entityType="TEMPLE", entityId="temple_1", confidence=1)],
             "DEITY": [EntityRef(text="s", name="Lord Shiva", entityType="DEITY", entityId="deity_1", confidence=1)]}
    p = assemble_plan("v", "/x", 10.0, seg, tr, Transcript(language="en"), known, location="Varanasi")
    assert p["overlays"]["title"] == "Kashi Vishwanath Temple" and p["overlays"]["subtitle"] == "Varanasi"      # metadata-only: the temple names the reel; the deity is never drawn
    assert p["overlays"]["deity"] is None and p["overlays"]["ritual"] is None
    p2 = assemble_plan("v", "/x", 10.0, seg, tr, Transcript(language="en"), {}, location=None); assert "title" not in p2["overlays"]           # unknown footage: no invented title
    p3 = assemble_plan("v", "/x", 10.0, seg, tr, Transcript(language="en"), {}, title="Ganga Aarti", subtitle="Haridwar"); assert p3["overlays"]["title"] == "Ganga Aarti"


def test_inferred_names_are_never_drawn_only_upload_metadata_is():
    """Real footage produced "Aarti" / "Hanuman" / "Chamundeshwari" from vision labels and transcript words. Anything below confidence 1 (not upload metadata) is data only."""
    from aikyam_video.models import EntityRef, Transcript
    from aikyam_video.plan import assemble_plan
    seg = [{"start": 0.0, "end": 5.0, "reason": "x", "score": 0.5}]; tr = {"type": "cut", "durationSeconds": 0}
    E = lambda typ, name, i, c: EntityRef(text=name, name=name, entityType=typ, entityId=i, confidence=c)
    inferred = {"RITUAL": [E("RITUAL", "Aarti", "ritual_12", 0.49)], "DEITY": [E("DEITY", "Hanuman", "deity_5", 0.75)],
                "TEMPLE": [E("TEMPLE", "Some Temple", "temple_9", 0.9)], "FESTIVAL": [E("FESTIVAL", "Dasara", "festival_9", 0.96)]}
    p = assemble_plan("v", "/x", 10.0, seg, tr, Transcript(language="en"), inferred)
    ov = p["overlays"]
    assert ov["template"] == "divine_moment" and "title" not in ov and not any(ov[k] for k in ("temple", "deity", "ritual", "festival"))
    assert p["source"]["ritualId"] == "ritual_12"                                                       # still recorded as data
    given = {**inferred, "FESTIVAL": [E("FESTIVAL", "Dasara", "festival_9", 1.0)], "TEMPLE": [E("TEMPLE", "Chamundeshwari Temple", "temple_123", 1.0)]}
    g = assemble_plan("v", "/x", 10.0, seg, tr, Transcript(language="en"), given)["overlays"]
    assert g["template"] == "dasara" and g["festival"] == "Dasara" and g["temple"] == "Chamundeshwari Temple" and g["ritual"] is None and g["deity"] is None
