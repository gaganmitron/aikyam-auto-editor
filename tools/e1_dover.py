"""EXP-028 / E1: does DOVER (aesthetic + technical video quality, arXiv 2211.04894; S-Lab License 1.0 = NON-COMMERCIAL, so this is an analysis, not a product component) predict our clip ratings?
Scores every 4 s benchmark window of tools/bench_gate (windows whose source video is still on disk) and writes results/e1/dover_<model>.json.
Run with the separate environment: results/dovervenv/bin/python tools/e1_dover.py [--model dover|mobile]   (needs results/DOVER cloned and its weights in results/DOVER/pretrained_weights)"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); DV = f"{ROOT}/results/DOVER"
sys.path.insert(0, DV); os.chdir(DV)
import torch, yaml
from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
from dover.models import DOVER
MEAN, STD = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])


def fuse(tech, aes):                                                                       # the repo's own score-level fusion (evaluate_one_video.py)
    x = (tech - 0.1107) / 0.07355 * 0.6104 + (aes + 0.08285) / 0.03774 * 0.3896
    return float(1 / (1 + np.exp(-x)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="dover", choices=["dover", "mobile"]); a = ap.parse_args()
    opt = yaml.safe_load(open(f"{DV}/{'dover.yml' if a.model == 'dover' else 'dover-mobile.yml'}"))
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    ev = DOVER(**opt["model"]["args"]); ev.load_state_dict(torch.load(f"{DV}/{opt['test_load_path'][2:]}", map_location="cpu")); ev.eval()
    dopt = opt["data"]["val-l1080p"]["args"]; samplers = {}
    for st, so in dopt["sample_types"].items():
        samplers[st] = UnifiedFrameSampler(so["clip_len"], so["num_clips"], so["frame_interval"]) if "t_frag" not in so else UnifiedFrameSampler(so["clip_len"] // so["t_frag"], so["t_frag"], so["frame_interval"], so["num_clips"])
    all_wins = [w for w in json.load(open(f"{ROOT}/results/bench_gate/windows.json")) if os.path.exists(f"{ROOT}/{w['path']}")]
    out_path = f"{ROOT}/results/e1/dover_{a.model}.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}                    # resumable: skip windows already scored (EXP-034 extends EXP-028's sample)
    wins = [w for w in all_wins if w["id"] not in out]
    print(f"{len(out)} already scored, {len(wins)} new", flush=True)
    for i, w in enumerate(wins):
        with tempfile.TemporaryDirectory(dir=f"{ROOT}/results/tmp_pip") as td:
            clip = f"{td}/w.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{w['start']}", "-t", "4", "-i", f"{ROOT}/{w['path']}", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "ultrafast", clip], check=True)
            views, _ = spatial_temporal_view_decomposition(clip, dopt["sample_types"], samplers)
            for k, v in views.items():
                nc = dopt["sample_types"][k].get("num_clips", 1)
                views[k] = ((v.permute(1, 2, 3, 0) - MEAN) / STD).permute(3, 0, 1, 2).reshape(v.shape[0], nc, -1, *v.shape[2:]).transpose(0, 1)
            with torch.no_grad(): r = [x.mean().item() for x in ev(views)]
        out[w["id"]] = {"technical": r[0], "aesthetic": r[1], "fused": fuse(r[0], r[1])}
        print(f"[{i + 1}/{len(wins)}] {w['id']}: technical {r[0]:.3f} aesthetic {r[1]:.3f} fused {out[w['id']]['fused']:.3f}", flush=True)
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
