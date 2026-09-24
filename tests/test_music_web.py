import json, os
import numpy as np
from aikyam_video import music as M, music_web as W


def r(id, lic, ver, sec=200, url="http://x/a.mp3", prov="jamendo"):
    return {"id": id, "title": f"T{id}", "creator": "Cee", "url": url, "foreign_landing_url": f"http://l/{id}", "license": lic, "license_version": ver, "license_url": "http://lic", "duration": sec * 1000, "provider": prov}


def test_only_usable_licences_and_lengths_survive():
    res = [r("1", "cc0", "1.0"), r("2", "by", "4.0"), r("3", "by-nc", "4.0"), r("4", "by-sa", "3.0"), r("5", "by-nd", "4.0"), r("6", "by", "2.0"), r("7", "pdm", "1.0"), r("8", "by", "4.0", sec=20), r("9", "by", "3.0", sec=900)]
    out = W.search("x", 50, fetch=lambda u: json.dumps({"results": res}).encode())
    assert [c["id"] for c in out] == ["1", "2", "7"] and [c["licence"] for c in out] == ["CC0", "CC-BY-4.0", "PUBLIC-DOMAIN"]


def test_queries_follow_the_content_and_profile():
    q = W.queries({"aarti": 0.9, "lamps": 0.7, "crowd": 0.1}, "contemplative"); assert q[:3] == ["mantra", "indian", "meditation"] and len(q) <= 6 and len(set(q)) == len(q) and all(" " not in x for x in q)
    assert "drums" in W.queries({"procession": 0.8}, "festive") and "sitar" in W.queries({}, "devotional")
    assert "aarti" in W.description({"aarti": 0.9}, "devotional")


class FakeEmb:
    """text -> unit vectors; audio embedding chosen by the file name: 'good' aligns with the description, 'speech' with the negative prompt."""
    def text(self, texts):
        v = np.eye(len(texts)); return v
    def audio(self, path):
        return np.array([1.0, 0, 0, 0, 0]) if "good" in path else np.array([0.2, 0.9, 0, 0, 0]) if "speech" in path else np.array([0.5, 0, 0.5, 0, 0])


def test_rank_prefers_music_like_the_description_over_speech_and_noise():
    out = W.rank({"a": "/x/good.mp3", "b": "/x/speech.mp3", "c": "/x/other.mp3"}, "desc", "devotional", FakeEmb(), kind_of=lambda p: "melodic")
    assert [x[1] for x in out] == ["a", "c", "b"]


def test_registered_track_passes_the_licence_gate_with_attribution(tmp_path):
    lib = tmp_path / "music"; (lib / "web").mkdir(parents=True); f = lib / "web" / "t.mp3"; f.write_bytes(b"\0" * 10)
    os.environ["MUSIC_LIBRARY_DIR"] = str(lib)
    try:
        c = {"id": "abc-123", "title": "Om", "creator": "Sage", "landing": "http://l/1", "licence": "CC-BY-4.0", "seconds": 120.0, "licenceUrl": "u"}
        t = W.register(c, str(f), "melodic", ["aarti"]); assert t.licence == "CC-BY-4.0" and "Sage" in t.attribution and t.kind == "melodic"
        tracks, bad = M.load_library(); assert [x.id for x in tracks] == [t.id] and not bad
    finally: os.environ.pop("MUSIC_LIBRARY_DIR")


def test_find_never_raises_when_offline(tmp_path):
    def boom(url): raise OSError("no network")
    t, why = W.find({"aarti": 0.9}, "devotional", 45, fetch=boom, lib=str(tmp_path)); assert t is None and "unavailable" in why


def test_a_beat_bonus_prefers_a_track_with_a_detectable_tempo_but_content_still_counts():
    import types
    class An:                                                                                     # what analyze_samples returns: kind + a tempo (or None)
        def __init__(self, bpm): self.kind, self.bpm = "melodic", bpm
    tempos = {"/x/good.mp3": None, "/x/speech.mp3": 100.0, "/x/other.mp3": 100.0}                # the best-matching track has no beat
    import aikyam_video.creative.musicdna as md, aikyam_video.ffmpeg as ff
    orig_an, orig_pcm = md.analyze_samples, ff.extract_audio_pcm
    md.analyze_samples = lambda pcm: An(tempos[pcm]); ff.extract_audio_pcm = lambda p, sr: p
    try:
        paths = {"a": "/x/good.mp3", "b": "/x/speech.mp3", "c": "/x/other.mp3"}
        base = W.rank(paths, "desc", "devotional", FakeEmb())
        bonus = W.rank(paths, "desc", "devotional", FakeEmb(), beat_bonus=1.0)
        small = W.rank(paths, "desc", "devotional", FakeEmb(), beat_bonus=0.20)
    finally:
        md.analyze_samples, ff.extract_audio_pcm = orig_an, orig_pcm
    assert base[0][1] == "a" and bonus[0][2]["beat"] is True and bonus[0][1] != "a"               # content wins by default; a big enough bonus lets a track with a tempo overtake a beatless one
    assert small[0][1] == "a"                                                                     # the small bonus we ship never overrides a clearly better content match


def test_extra_search_words_are_added_not_cut_off():
    labels = {"aarti": 0.9, "procession": 0.8}
    assert "kirtan" not in W.queries(labels, "contemplative")                                        # the usual list is capped at 6
    q = W.queries(labels, "contemplative", ["bhajan", "kirtan", "tabla"], n=9); assert {"bhajan", "kirtan", "tabla"} <= set(q)
