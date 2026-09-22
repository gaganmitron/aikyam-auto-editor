"""Runs Beat This! (MIT, https://github.com/CPJKU/beat_this) in its OWN interpreter: it needs Python >= 3.10 syntax and torch, the main environment is 3.9.
Usage (called by aikyam_video.creative.beats):  python beat_this_worker.py AUDIO.wav  ->  JSON {"beats": [...], "downbeats": [...]} on stdout."""
import json, sys

if __name__ == "__main__":
    from beat_this.inference import File2Beats
    b, d = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)(sys.argv[1])
    print(json.dumps({"beats": [round(float(x), 4) for x in b], "downbeats": [round(float(x), 4) for x in d]}))
