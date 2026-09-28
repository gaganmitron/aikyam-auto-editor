"""EXP-026 / E4: do transition costs make a better sequence? Builds BLIND A/B pairs of reels from the SAME footage and the SAME planner; only the transition terms differ (--sequence-terms).
  1. every configuration = (subset of the 4 videos) x (target length) x (pacing profile); the planner runs twice (terms off / on) up to the story stage; only configurations whose ORDERED clip list differs are kept
  2. up to N diverse pairs are fully planned and rendered (silent: the judgement is the order and flow of the pictures), A/B assignment randomised, key kept in key.json
  3. results/e4/index.html: watch, pick A / B / no difference, copy the result text
usage: python tools/e4_pairs.py [--pairs 20]        (one heavy job: ~1 h on the CPU laptop; resumable, finished renders are skipped)
  E4b (EXP-029): python tools/e4_pairs.py --out results/e4b --treat spread_sources --min-assets 2 --targets 30,40,50 --per-subset 2 --require-fewer-same-source --pairs 12"""
import argparse, copy, dataclasses, itertools, json, os, random, shutil, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aikyam_video import multi, shotsize, stages
from aikyam_video.creative import engine as E, story
from aikyam_video.options import Options

INPUTS = ["inputs/tirumala_darshan_2021.mp4", "inputs/tirumala_drone_2025.mp4", "inputs/tirumala_tourist_2024_trimmed.mp4", "inputs/tirupati_padmavathi_abhishekam_2020.mp4"]
NAMES = {"v1": "darshan", "v2": "drone", "v3": "tourist", "v4": "padmavathi"}
OUT = "results/e4"; ART = f"{OUT}/art"; SEED = 26; TREAT_W = None            # TREAT_W: transition weights of the treated arm (set in main from --treat)


class _Stop(Exception):
    pass


def setup():
    os.makedirs(ART, exist_ok=True)
    if not os.path.isdir(f"{ART}/assets") and os.path.isdir("results/tirumala_multi2/assets"):
        shutil.copytree("results/tirumala_multi2/assets", f"{ART}/assets")          # the analysis of these exact files (path+size+mtime are checked by ingest)
    o = Options(formats=["reel"], allow_silent=True, source_audio="bed", transcript=False, captions=False, qc=False)
    srcs, _ = multi.ingest(INPUTS, ART, o)
    from aikyam_video.vision import ClipVision                                       # backfill shot size for the analysis cached before the field existed
    cv = ClipVision(); Tm, own = shotsize.text_embeddings(cv.embed_text); scale = float(cv.model.logit_scale.exp())
    for x in srcs:
        for sv in x.vision:
            assert sv.vision.embedding, "scene embedding missing: cannot backfill shot size"
            sv.vision.shot_size = shotsize.score(np.asarray(sv.vision.embedding), Tm, own, scale)
    del cv
    orig, cache = E.build_shots, {}

    def memo(*a, **kw):                                                              # shot building decodes video: do it once per (file, pool size)
        key = (a[0], kw.get("top_k"), kw.get("asset_id"), kw.get("diverse"))
        if key not in cache: cache[key] = orig(*a, **kw)
        return copy.deepcopy(cache[key])
    E.build_shots = memo
    return o, srcs


def ctx_for(subset, o):
    ents = multi._merge_entities([stages._load_entities(os.path.join(ART, "assets", s.id)) for s in subset])
    if len(subset) == 1:                                                              # a single video is planned like the normal single-video run (chronology rules, larger pool)
        s = subset[0]; return E.Ctx(s.path, s.info, s.moments, s.vision, s.audio, s.transcript, ents, o, NAMES[s.id])
    s = subset[0]; return E.Ctx(s.path, s.info, s.moments, s.vision, s.audio, s.transcript, ents, o, "mix", sources=subset)


def screen_one(subset, o):
    """(baseline timeline, treated timeline) for one configuration, stopping right after the story stage."""
    orig = story.plan_stories

    def spy(*a, **kw):
        t0 = orig(copy.deepcopy(a[0]), *a[1:], **dict(kw, transition=None)); t1 = orig(copy.deepcopy(a[0]), *a[1:], **dict(kw, transition=TREAT_W))
        raise _Stop((t0[0], t1[0]))
    story.plan_stories = spy
    try:
        E.build_plans(ctx_for(subset, o))
    except _Stop as s:
        return s.args[0]
    finally:
        story.plan_stories = orig


def same_src(tl): return sum(1 for x, y in zip(tl.clips, tl.clips[1:]) if x.shot.asset_id == y.shot.asset_id)
def seq(tl): return [(c.shot.id, round(c.start, 1)) for c in tl.clips]
def dur(tl): return sum(c.length for c in tl.clips)


def audio_for(subset):
    return subset[0].audio if len(subset) == 1 else {s.id: s.audio for s in subset}


def render(plan, subset, out_dir, o):
    os.makedirs(out_dir, exist_ok=True); path = os.path.join(out_dir, "reel-9x16.mp4")
    if not os.path.exists(path): E.render_reel(plan, subset[0].path, out_dir, o, audio_for(subset), ["reel"], "reel", None)
    return path


def silent_small(src, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-an", "-vf", "scale=540:-2", "-c:v", "libx264", "-crf", "24", "-preset", "veryfast", "-movflags", "+faststart", dst], check=True)


PAGE = """<!doctype html><meta charset=utf-8><title>__TITLE__: which reel flows better?</title>
<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;background:#111;color:#eee}
.pair{border:1px solid #333;border-radius:10px;padding:14px;margin:18px 0}.vids{display:flex;gap:16px;justify-content:center}
video{width:46%;max-height:70vh;background:#000;border-radius:6px}label{margin-right:16px;cursor:pointer}button{font-size:16px;padding:8px 14px}
textarea{width:100%;height:80px;background:#222;color:#eee}small{color:#aaa}</style>
<h2>Which reel flows better as an edit?</h2>
<p>Each pair is made from the same footage. The reels are <b>silent on purpose</b>: judge the <b>order and flow of the pictures</b> (which clip comes after which, does it feel like a well-edited story?), not the sound and not small length differences.
Pick A, B, or "no difference" if you honestly cannot tell. Your picks are saved in this browser as you go.</p>
<div id=root></div><h3>Result</h3><button onclick="out()">Show / copy my result</button><textarea id=res readonly></textarea>
<script>
const N=__N__;const root=document.getElementById('root');const S=JSON.parse(localStorage.getItem('__KEY__')||'{}');
for(let i=1;i<=N;i++){const id=String(i).padStart(2,'0');const d=document.createElement('div');d.className='pair';
d.innerHTML=`<b>Pair ${i}</b> <small>(${'__DUR__'.split(',')[i-1]})</small><div class=vids><video src="pairs/pair_${id}/A.mp4" controls loop muted playsinline></video><video src="pairs/pair_${id}/B.mp4" controls loop muted playsinline></video></div>
<p style="text-align:center"><small>left = A, right = B</small><br>
<label><input type=radio name=p${i} value=A> A is better</label><label><input type=radio name=p${i} value=B> B is better</label><label><input type=radio name=p${i} value="="> no difference</label></p>`;
root.appendChild(d);d.querySelectorAll('input').forEach(r=>{if(S[i]===r.value)r.checked=true;r.onchange=()=>{S[i]=r.value;localStorage.setItem('__KEY__',JSON.stringify(S))}})}
function out(){const t=Array.from({length:N},(_,k)=>(k+1)+':'+(S[k+1]||'?')).join(' ');const a=document.getElementById('res');a.value=t;a.select();try{navigator.clipboard.writeText(t)}catch(e){}}
</script>"""


def write_page(pairs):
    labels = ",".join(f"{'+'.join(NAMES[x] for x in p['assets'])} {p['target']:.0f}s" for p in pairs)
    open(f"{OUT}/index.html", "w").write(PAGE.replace("__KEY__", os.path.basename(OUT.rstrip("/"))).replace("__TITLE__", os.path.basename(OUT.rstrip("/")).upper()).replace("__N__", str(len(pairs))).replace("__DUR__", labels.replace("'", "")))
    print(f"page written for {len(pairs)} pairs: {os.path.abspath(OUT)}/index.html", flush=True)


def main():
    global OUT, ART, TREAT_W
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=int, default=20); ap.add_argument("--page-only", action="store_true", help="rebuild index.html from the pairs finished so far (safe while a run is going)")
    ap.add_argument("--out", default="results/e4"); ap.add_argument("--treat", default="sequence_terms", choices=["sequence_terms", "spread_sources"], help="which opt-in option is the treated arm")
    ap.add_argument("--min-assets", type=int, default=1); ap.add_argument("--targets", default="30,40"); ap.add_argument("--per-subset", type=int, default=3)
    ap.add_argument("--require-fewer-same-source", action="store_true", help="keep only configurations where the treated reel has fewer back-to-back clips from the same video"); a = ap.parse_args()
    OUT = a.out.rstrip("/"); ART = f"{OUT}/art"; TREAT_W = E.SPREAD_TERMS if a.treat == "spread_sources" else E.SEQUENCE_TERMS
    if a.page_only: return write_page(json.load(open(f"{OUT}/pairs.json")))
    o, srcs = setup(); rng = random.Random(SEED); t0 = time.time()
    configs = []
    for r in range(a.min_assets, 5):
        for sub in itertools.combinations(srcs, r):
            for target in [float(x) for x in a.targets.split(",")]:
                for pace in ("devotional", "contemplative"):
                    configs.append((sub, target, pace))
    rng.shuffle(configs); os.makedirs(OUT, exist_ok=True); screen = []
    print(f"{len(configs)} configurations; screening (terms off vs on) ...", flush=True)
    for k, (sub, target, pace) in enumerate(configs):
        oc = dataclasses.replace(o, target_seconds=target, pacing=pace)
        try: t0s, t1s = screen_one(list(sub), oc)
        except Exception as ex:
            print("  skip", [s.id for s in sub], target, pace, type(ex).__name__, str(ex)[:80], flush=True); continue
        differs = seq(t0s) != seq(t1s); ok = differs and abs(dur(t0s) - dur(t1s)) <= 4.0 and (not a.require_fewer_same_source or same_src(t1s) < same_src(t0s))
        screen.append({"assets": [s.id for s in sub], "target": target, "pacing": pace, "differs": differs, "duration_ok": ok, "base": seq(t0s), "treat": seq(t1s), "dur": [round(dur(t0s), 1), round(dur(t1s), 1)], "same_source": [same_src(t0s), same_src(t1s)]})
        print(f"  [{k + 1}/{len(configs)}] {'+'.join(NAMES[s.id] for s in sub):40s} {target:.0f}s {pace:13s} {'DIFFERENT' if differs else 'same'}{'' if ok or not differs else ' (duration gap)'}", flush=True)
    json.dump(screen, open(f"{OUT}/screen.json", "w"), indent=1)
    usable = [c for c in screen if c["duration_ok"]]; print(f"\n{len(usable)} of {len(screen)} configurations give a different sequence with a comparable length", flush=True)
    chosen, per = [], {}
    for c in usable:                                                                  # diverse: at most 3 pairs from the same set of videos
        key = tuple(c["assets"])
        if per.get(key, 0) < a.per_subset and len(chosen) < a.pairs: chosen.append(c); per[key] = per.get(key, 0) + 1
    by = {s.id: s for s in srcs}; pairs, key = [], {}
    for i, c in enumerate(chosen, 1):
        sub = [by[x] for x in c["assets"]]; oc = dataclasses.replace(o, target_seconds=c["target"], pacing=c["pacing"]); d = f"{OUT}/pairs/pair_{i:02d}"
        info = {"n": i, "assets": c["assets"], "target": c["target"], "pacing": c["pacing"]}
        for tag, opt in (("base", oc), ("treat", dataclasses.replace(oc, **{a.treat: True}))):
            plan = E.build_plans(ctx_for(sub, opt))[0]
            p = render(plan, sub, f"{d}/{tag}", opt)
            info[tag] = {"file": p, "duration": plan["durationSeconds"], "clips": [(NAMES.get(s.get("assetId"), "single"), round(s["start"], 1), round(s["end"], 1), s["role"], s.get("beat")) for s in plan["segments"]],
                         "transition_cost": next((x for x in plan["creative"]["decisions"] if x["type"] == "transition_cost"), None)}
        flip = rng.random() < 0.5; ab = ("treat", "base") if flip else ("base", "treat")           # A/B randomised: the key says which is which
        for lab, tag in zip("AB", ab): silent_small(info[tag]["file"], f"{d}/{lab}.mp4")
        key[f"{i:02d}"] = {"A": ab[0], "B": ab[1]}; pairs.append(info)
        json.dump(pairs, open(f"{OUT}/pairs.json", "w"), indent=1, default=str); json.dump(key, open(f"{OUT}/key.json", "w"), indent=1)
        print(f"pair {i}/{len(chosen)} done ({time.time() - t0:.0f}s elapsed)", flush=True)
    write_page(pairs)
    print(f"\nDONE: {len(pairs)} pairs. Open {os.path.abspath(OUT)}/index.html in a browser.", flush=True)


if __name__ == "__main__":
    main()
