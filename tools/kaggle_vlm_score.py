"""Run on a free GPU (Kaggle/Colab T4) -- no aikyam code needed. Scores video frames with SmolVLM2-2.2B.
Per frame: reel-worthiness 1-5 (expected value from the next-token probabilities, one forward pass) + scene type A-H.
usage:  python kaggle_vlm_score.py VIDEO_OR_FRAMES_DIR [--step 2] [--ratings bench_clips_ratings.json] [-o vlm_scores.json]
  VIDEO       -> one frame every --step seconds (ffmpeg)      FRAMES_DIR -> every *.jpg/png in it
  --ratings   -> also prints Spearman vs the hand ratings (keys = frame file stems); use this on the 35 bench frames FIRST.
Output: [{"name": "...", "t": seconds|null, "score": 1..5, "type": "A".."H", "type_p": 0..1}]
Kaggle setup cell:  !pip -q install -U transformers accelerate num2words   (ffmpeg is preinstalled)"""
import argparse, glob, json, os, subprocess, tempfile
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
TYPES = {"A": "a priest waving a burning fire lamp (aarti)", "B": "a decorated deity idol or statue in a shrine",
         "C": "the outside of a temple building", "D": "people praying or standing in a crowd",
         "E": "people carrying a decorated palanquin or canopy", "F": "attendants or priests performing a ritual inside a temple",
         "G": "people sitting and eating a meal", "H": "a text page, title card or unrelated close-up"}
Q_SCORE = ("You judge frames for a short devotional temple Reel. Rate this frame 1-5: 5 = clear subject, good composition, "
           "devotionally meaningful, sharp; 1 = blurred, cluttered, text overlay/credits or meaningless. Answer with one digit.")
Q_TYPE = "What is shown in this image? Choose one:\n" + "\n".join(f"{k}. {v}" for k, v in TYPES.items()) + "\nAnswer with the letter only."


def frames_of(src, step):
    if os.path.isdir(src):
        fs = sorted(p for p in glob.glob(os.path.join(src, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
        return [(os.path.splitext(os.path.basename(p))[0], None, p) for p in fs]
    td = tempfile.mkdtemp()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-i", src, "-vf", f"fps=1/{step},scale=-2:768", "-q:v", "3", f"{td}/%06d.jpg"], check=True)
    return [(f"t{i * step + step / 2:.1f}", i * step + step / 2, p) for i, p in enumerate(sorted(glob.glob(f"{td}/*.jpg")))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--ratings"); ap.add_argument("-o", default="vlm_scores.json"); a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(REPO, size={"longest_edge": 768}, do_image_splitting=False)
    model = AutoModelForImageTextToText.from_pretrained(REPO, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    tok = proc.tokenizer

    def ids(s):  # the answer token may or may not carry a leading space
        return sorted({tok.encode(v, add_special_tokens=False)[0] for v in (s, " " + s)})
    # digits: " 3" may split into [space, "3"] -> then every option shares the space token (all scores 3.0). End the prompt with the space and use the digit token itself.
    split = len(tok.encode(" 3", add_special_tokens=False)) == 2
    digit_ids = {d: [tok.encode(d, add_special_tokens=False)[-1]] for d in "12345"} if split else {d: ids(d) for d in "12345"}
    letter_ids = {k: ids(k) for k in TYPES}

    def next_probs(img, q, choices):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True) + (" " if split and choices is digit_ids else "")
        inp = proc(text=text, images=[img], return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        s = torch.stack([torch.logsumexp(lg[v], 0) for v in choices.values()])
        return dict(zip(choices, torch.softmax(s, 0).tolist()))

    out = []
    for name, t, path in frames_of(a.src, a.step):
        img = Image.open(path).convert("RGB")
        ps, pt = next_probs(img, Q_SCORE, digit_ids), next_probs(img, Q_TYPE, letter_ids)
        ty = max(pt, key=pt.get)
        out.append({"name": name, "t": t, "score": round(sum(int(d) * p for d, p in ps.items()), 3), "type": ty, "type_p": round(pt[ty], 3)})
        print(out[-1], flush=True)
    json.dump(out, open(a.o, "w"), indent=1)
    if len({o["score"] for o in out}) == 1:
        print("WARNING: every score is identical -> token ids are wrong, send this output to Claude")
    if a.ratings:
        from scipy.stats import spearmanr
        r = json.load(open(a.ratings)); ok = [o for o in out if o["name"] in r]
        print(f"Spearman vs hand ratings: {spearmanr([o['score'] for o in ok], [r[o['name']] for o in ok])[0]:.2f} on {len(ok)} frames")


if __name__ == "__main__":
    main()
