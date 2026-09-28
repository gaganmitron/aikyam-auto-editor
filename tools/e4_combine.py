"""Put every finished E4 pair into ONE video to watch straight through: for each pair, a 1.5 s title card, reel A, a title card, reel B. Labels never say which reel used the transition terms.
usage: python tools/e4_combine.py     -> results/e4/all_pairs.mp4 (silent, 540x960)"""
import json, os, subprocess, sys
OUT = "results/e4"; W, H = 540, 960; FONT = "/usr/share/fonts/truetype/lato/Lato-Medium.ttf"; TMP = f"{OUT}/_combine"; os.makedirs(TMP, exist_ok=True)
pairs = json.load(open(f"{OUT}/pairs.json")); parts = []


def run(*a): subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


def card(text, sub, path):
    run("-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=25:d=1.5", "-vf",
        f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2-30,drawtext=fontfile={FONT}:text='{sub}':fontcolor=0xbbbbbb:fontsize=30:x=(w-text_w)/2:y=(h)/2+40",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "veryfast", path)


def clip(src, label, path):
    run("-i", src, "-an", "-vf", f"scale={W}:{H},fps=25,drawtext=fontfile={FONT}:text='{label}':fontcolor=white:fontsize=34:box=1:boxcolor=black@0.55:boxborderw=10:x=20:y=20,format=yuv420p",
        "-c:v", "libx264", "-crf", "24", "-preset", "veryfast", path)


for p in pairs:
    n = p["n"]; d = f"{OUT}/pairs/pair_{n:02d}"
    for lab in "AB":
        c, k = f"{TMP}/c_{n:02d}{lab}.mp4", f"{TMP}/k_{n:02d}{lab}.mp4"
        if not os.path.exists(k): clip(f"{d}/{lab}.mp4", f"Pair {n} - {lab}", k)
        if not os.path.exists(c): card(f"Pair {n}", f"reel {lab}", c)
        parts += [c, k]
open(f"{TMP}/list.txt", "w").write("".join(f"file '{os.path.abspath(x)}'\n" for x in parts))
run("-f", "concat", "-safe", "0", "-i", f"{TMP}/list.txt", "-c", "copy", f"{OUT}/all_pairs.mp4")
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f"{OUT}/all_pairs.mp4"], capture_output=True, text=True).stdout)
print(f"{len(pairs)} pairs -> {OUT}/all_pairs.mp4 ({dur / 60:.1f} min)")
