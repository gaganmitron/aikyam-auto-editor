"""EXP-013: can a small vision-language model UNDERSTAND a frame better than SigLIP zero-shot, at a speed this CPU can afford?
Task: closed-set scene type on the 35 bench frames. Scene-type truth is derived from tools/bench_labels_truth.json (see TYPE below).
usage: python tools/bench_vlm.py smolvlm256 [n_frames]      (SigLIP baseline always runs; needs results/bench_labels/frames.json)"""
import json, os, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPES = {  # letter -> (name, prompt phrase for SigLIP, option text for the VLM)
    "A": ("aarti", "a priest waving a burning aarti lamp at night", "a priest waving a burning fire lamp (aarti)"),
    "B": ("deity", "a decorated deity idol in a shrine", "a decorated deity idol or statue in a shrine"),
    "C": ("exterior", "a temple building seen from outside, across a lake or at night", "the outside of a temple building"),
    "D": ("devotees", "devotees praying with folded hands in a crowd", "people praying or standing in a crowd"),
    "E": ("procession", "people carrying a decorated palanquin or canopy in a procession", "people carrying a decorated palanquin or canopy"),
    "F": ("rites", "attendants performing a ritual inside a temple sanctum", "attendants or priests performing a ritual inside a temple"),
    "G": ("food", "people sitting in rows being served food", "people sitting and eating a meal"),
    "H": ("other", "a title card, text or an unrelated close-up", "a text page, title card or unrelated close-up"),
}


def scene_type(name, labels):
    L = set(labels)
    if not L: return "H"
    if "aarti" in L or ("lamps" in L and "priest" in L and "deity" not in L): return "A"
    if "deity" in L or "idol" in L: return "B"
    if "food_service" in L: return "G"
    if "procession" in L: return "E"
    if L & {"temple_architecture", "temple_tank", "temple_night"}: return "C"
    if "priest" in L: return "F"
    if "devotees" in L or "crowd" in L: return "D"
    return "H"


frames = json.load(open("results/bench_labels/frames.json"))
names = [os.path.basename(f)[:-4] for f in frames]
truth = json.load(open("tools/bench_labels_truth.json"))
y = [scene_type(n, truth[n]) for n in names]
n_use = int(sys.argv[2]) if len(sys.argv) > 2 else len(frames)
idx = np.linspace(0, len(frames) - 1, n_use).round().astype(int) if n_use < len(frames) else np.arange(len(frames))
print("scene-type distribution of the truth:", {k: y.count(k) for k in sorted(set(y))})


def report(tag, pred, secs):
    ok = [pred[i] == y[i] for i in idx]
    print(f"{tag:22s} accuracy {np.mean(ok):.2f} on {len(idx)} frames   {np.mean(secs):.1f}s/frame" if secs else f"{tag:22s} accuracy {np.mean(ok):.2f} on {len(idx)} frames")
    wrong = [(names[i], y[i], pred[i]) for i in idx if pred[i] != y[i]]
    print("   wrong (frame, truth, predicted):", wrong[:14])


# ---- baseline: SigLIP zero-shot on the same closed set (embeddings cached by bench_clips.py)
E = np.load("results/bench_labels/emb.npz")["E"]
from aikyam_video.vision import ClipVision
cv = ClipVision(); T = cv.embed_text([v[1] for v in TYPES.values()]); keys = list(TYPES)
sig_pred = [keys[int(np.argmax(E[i] @ T.T))] for i in range(len(frames))]
report("SigLIP zero-shot", sig_pred, None)
del cv

which = sys.argv[1] if len(sys.argv) > 1 else "smolvlm256"
if which.startswith("smolvlm"):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    torch.set_num_threads(int(os.environ.get("BENCH_THREADS", "6")))
    repo = {"smolvlm256": "HuggingFaceTB/SmolVLM-256M-Instruct", "smolvlm500": "HuggingFaceTB/SmolVLM-500M-Instruct"}[which]
    proc = AutoProcessor.from_pretrained(repo, size={"longest_edge": 512}, do_image_splitting=False); model = AutoModelForVision2Seq.from_pretrained(repo, torch_dtype=torch.bfloat16 if os.environ.get("BENCH_BF16") else torch.float32).eval()
    opts = "\n".join(f"{k}. {v[2]}" for k, v in TYPES.items())
    q = f"What is shown in this image? Choose one:\n{opts}\nAnswer with the letter only."
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    pred, secs, raw = {}, [], {}
    for i in idx:
        t0 = time.time()
        inp = proc(text=prompt, images=[Image.open(frames[i]).convert("RGB")], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=4, do_sample=False)
        ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        raw[names[i]] = ans; pred[i] = next((c for c in ans.upper().replace("ANSWER", "") if c in TYPES), "?"); secs.append(time.time() - t0)
        print(f"  {names[i]:14s} truth {y[i]} vlm {pred[i]} ({ans!r}) {secs[-1]:.1f}s", flush=True)
    report(which, pred, secs)
