"""EXP-032 (weak sources), GPU-laptop half: is Real-ESRGAN restoration WORTH it, not just whether it runs? On this CPU it measured 106 s for ONE 640x422 frame
(infeasible for video: 3.7-8.8 h per 5-10 s clip) -- that number says nothing about whether the RESULT looks better, only that it is too slow here to find out.
This tool restores a handful of real soft clips (Padmavathi -- our softest source) frame by frame on a GPU, re-encodes them, and scores BEFORE vs AFTER
with the signals this project already trusts (DOVER fused, EXP-028/034; sharpness) plus a visual contact sheet, so the answer is measured, not assumed.

Setup (BSD-3-Clause, real pretrained weights, already verified to import cleanly against this repo's pinned torch once the known break is patched, EXP-032):
    pip install realesrgan basicsr
    # one-line community shim for a torchvision internal that newer torchvision removed (see CLAUDE.md gotchas):
    python -c "import torchvision,os; open(os.path.dirname(torchvision.__file__)+'/transforms/functional_tensor.py','w').write('from torchvision.transforms.functional import rgb_to_grayscale\\n')"
    curl -L -o results/esrgan_weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
DOVER (for scoring) is set up separately per tools/e1_dover.py's docstring.

usage: python tools/e_restore.py [--n 3] [--outscale 2]        -> results/e_restore/<clip>_{before,after}.mp4 + a printed before/after table
Deliberately NOT wired into the render pipeline: this is a measurement, run once and decide, matching how DOVER (EXP-028) and every other Tier-3 tool here was adopted."""
import argparse, json, os, subprocess, sys, tempfile
import cv2
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = "results/e_restore"
CANDIDATES = [("padmavathi", "inputs/tirupati_padmavathi_abhishekam_2020.mp4", 12.5, 4.0), ("padmavathi2", "inputs/tirupati_padmavathi_abhishekam_2020.mp4", 51.6, 4.0)]  # (name, path, start, len) -- our softest source, EXP-025's rated close-ups


def restorer(outscale, device):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return RealESRGANer(scale=4, model_path=f"{ROOT}/results/esrgan_weights/RealESRGAN_x4plus.pth", model=model, tile=400, half=(device == "cuda"), device=device), outscale


def restore_clip(src, start, length, dst, up, outscale, fps=25):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        cap = cv2.VideoCapture(src); cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        writer = None; n = int(length * fps)
        for i in range(n):
            ok, frame = cap.read()
            if not ok: break
            out, _ = up.enhance(frame, outscale=outscale)
            if writer is None:
                h, w = out.shape[:2]; writer = cv2.VideoWriter(f"{td}/v.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(out)
            print(f"  frame {i + 1}/{n}", flush=True)
        cap.release()
        if writer: writer.release()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{td}/v.mp4", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", dst], check=True)


def plain_clip(src, start, length, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", src, "-an", "-c:v", "libx264", "-crf", "18", dst], check=True)


def sharpness(path):
    cap = cv2.VideoCapture(path); vals = []
    while True:
        ok, f = cap.read()
        if not ok: break
        vals.append(cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    return float(np.median(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=len(CANDIDATES)); ap.add_argument("--outscale", type=float, default=2.0)
    ap.add_argument("--device", default=None, help="cuda | cpu; default: cuda if available"); a = ap.parse_args()
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("WARNING: no CUDA device found -- this WILL be extremely slow (measured 106 s/frame on a CPU laptop, EXP-032). Pass --device cpu to proceed anyway.", flush=True)
    up, outscale = restorer(a.outscale, device)
    rows = []
    for name, path, start, length in CANDIDATES[: a.n]:
        before, after = f"{OUT}/{name}_before.mp4", f"{OUT}/{name}_after.mp4"
        plain_clip(path, start, length, before)
        print(f"restoring {name} ({path} @ {start}s, {length}s, outscale {outscale})...", flush=True)
        restore_clip(path, start, length, after, up, outscale)
        rows.append({"name": name, "sharpness_before": sharpness(before), "sharpness_after": sharpness(after)})
        print(f"  sharpness: {rows[-1]['sharpness_before']:.0f} -> {rows[-1]['sharpness_after']:.0f}", flush=True)
    json.dump(rows, open(f"{OUT}/sharpness.json", "w"), indent=1)
    print("\nNow run DOVER (tools/e1_dover.py's environment) on the *_before.mp4 / *_after.mp4 pairs in", OUT, "for the real quality-model verdict --")
    print("sharpness alone can go up from restoration artefacts that do NOT look better; DOVER is the signal this project trusts (EXP-028/034).")


if __name__ == "__main__":
    main()
