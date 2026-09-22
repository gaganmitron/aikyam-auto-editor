"""Runs the neural boundary models in the Python 3.11 env (.venv-beat): TransNetV2 shot-boundary detection (MIT, soCzech/TransNetV2, PyTorch port transnetv2-pytorch) and
Silero VAD speech segmentation (MIT, snakers4/silero-vad). Called by aikyam_video.creative.mlx.
    python ml_worker.py cuts   VIDEO   ->  {"cuts":   [seconds...]}
    python ml_worker.py speech MEDIA   ->  {"speech": [[start, end]...]}"""
import json, subprocess, sys
import numpy as np


def _ff(args):
    return subprocess.run(["ffmpeg", "-v", "error", *args], capture_output=True).stdout


def cuts(path, thr=0.5, fps=25):
    import torch
    from transnetv2_pytorch import TransNetV2
    raw = _ff(["-i", path, "-an", "-vf", f"fps={fps},scale=48:27:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
    v = np.frombuffer(raw, np.uint8).reshape(-1, 27, 48, 3)
    if len(v) < 3: return []
    m = TransNetV2(); m.eval()
    p = m.predict_frames(torch.from_numpy(v.copy()), quiet=True)[0].numpy()
    pk = [i for i in range(len(p)) if p[i] > thr and p[i] >= p[max(i - 1, 0)] and p[i] >= p[min(i + 1, len(p) - 1)]]
    out = []
    for i in pk:
        if not out or (i + 1) / fps - out[-1] > 0.4: out.append(round((i + 1) / fps, 3))      # the first frame of the new shot
    return out


def speech(path):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    raw = _ff(["-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
    a = np.frombuffer(raw, np.float32)
    if len(a) < 16000: return []
    ts = get_speech_timestamps(torch.from_numpy(a.copy()), load_silero_vad(), sampling_rate=16000, return_seconds=True, min_silence_duration_ms=300)
    return [[round(t["start"], 3), round(t["end"], 3)] for t in ts]


if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    print(json.dumps({"cuts": cuts(path)} if cmd == "cuts" else {"speech": speech(path)}))
