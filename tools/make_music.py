"""Generates the STARTER devotional music library (music/*.mp3 + music/library.json) by numpy synthesis.
These are Aikyam-owned (we generated every sample) => no licence risk, but they are SIMPLE synthesised placeholders.
Replace/extend with commissioned recordings by adding entries to music/library.json (see README, section Music)."""
import json, subprocess, sys
import numpy as np

SR, DUR = 44100, 75            # 75 s > the 60 s reel cap: a track never needs to loop
rng = np.random.default_rng(7)
T = np.arange(SR * DUR) / SR
SA = 130.81                    # C3 as "Sa"; Pa = 1.5*Sa, upper Sa = 2*Sa


def add(buf, sig, start):
    i = int(start * SR); n = min(len(sig), len(buf) - i)
    if n > 0: buf[i:i + n] += sig[:n]


def pluck(freq, dur=6.0, decay=3.2, bright=0.6):
    """Karplus-Strong string with a slow decay + a touch of 'jivari' buzz (extra harmonics)."""
    n = int(SR * dur); N = int(SR / freq); buf = rng.uniform(-1, 1, N) * np.linspace(1, bright, N)
    out = np.zeros(n)
    for i in range(n):
        out[i] = buf[i % N]; buf[i % N] = 0.4985 * (buf[i % N] + buf[(i + 1) % N])
    return out * np.exp(-np.arange(n) / SR / decay * 0.6)


def tanpura(vol=1.0, dur=DUR):
    """4-string cycle Pa - Sa' - Sa' - Sa (low), repeating every 6 s."""
    out = np.zeros(SR * dur); cache = {f: pluck(f) for f in (SA * 1.5, SA * 2, SA)}
    for k in range(int(dur // 6) + 1):
        for j, f in enumerate((SA * 1.5, SA * 2, SA * 2, SA)):
            add(out, cache[f] * (0.9 if j < 3 else 1.1), k * 6 + j * 1.35)
    return out / (np.abs(out).max() + 1e-9) * vol


def bell(freq, dur=7.0):
    """Inharmonic partials of a struck bell, each with its own decay."""
    t = np.arange(int(SR * dur)) / SR; out = np.zeros_like(t)
    for r, a, d in ((1, 1.0, 3.5), (2.0, 0.6, 2.6), (2.76, 0.5, 2.0), (4.07, 0.3, 1.4), (5.4, 0.25, 1.0), (8.93, 0.12, 0.6)):
        out += a * np.sin(2 * np.pi * freq * r * t + rng.uniform(0, 6.28)) * np.exp(-t / d)
    return out * np.minimum(1, t / 0.004)


def dhol(dur=DUR, bpm=104):
    """Low 'dhagga' thump + high 'tilli' slap on an 8-beat teental-like cycle, plus off-beat shaker."""
    out = np.zeros(SR * dur); beat = 60 / bpm
    def thump():
        t = np.arange(int(SR * 0.35)) / SR
        return np.sin(2 * np.pi * (60 + 90 * np.exp(-t * 28)) * t) * np.exp(-t * 9)
    def slap():
        t = np.arange(int(SR * 0.12)) / SR
        n = rng.normal(0, 1, len(t)); n = n - np.convolve(n, np.ones(6) / 6, mode="same")   # crude high-pass
        return (n * 0.5 + np.sin(2 * np.pi * 420 * t) * 0.5) * np.exp(-t * 30)
    pat = [(0, "T"), (1, "S"), (1.5, "S"), (2, "T"), (3, "S"), (3.5, "T"), (4, "T"), (5, "S"), (6, "T"), (6.5, "S"), (7, "S")]
    cyc = 8 * beat; th, sl = thump(), slap()
    for k in range(int(dur / cyc) + 1):
        for b, kind in pat: add(out, (th * 1.0 if kind == "T" else sl * 0.55), k * cyc + b * beat)
        for b in range(8): add(out, slap() * 0.18, k * cyc + (b + 0.5) * beat)
    return out / (np.abs(out).max() + 1e-9)


def flute(dur=DUR):
    """Slow Bhoopali phrase (Sa Re Ga Pa Dha Sa') with vibrato, breath noise and a soft attack."""
    scale = {"S": 1, "R": 9 / 8, "G": 5 / 4, "P": 3 / 2, "D": 5 / 3, "S'": 2, "d": 5 / 6, "p": 3 / 4}
    phrases = ["G P D P G R S", "S R G P G R S", "G P D S' D P G", "P D S' D P G R S", "G R S d S R G", "R G P D P G R S"]
    out = np.zeros(SR * dur); t0 = 1.0; base = SA * 4
    for k in range(40):
        for name in phrases[k % len(phrases)].split():
            f = base * scale[name]; ln = rng.uniform(1.4, 2.6); t = np.arange(int(SR * ln)) / SR
            vib = 1 + 0.006 * np.sin(2 * np.pi * 5.2 * t) * np.minimum(1, t / 0.5)
            ph = 2 * np.pi * np.cumsum(f * vib) / SR
            tone = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)
            tone += rng.normal(0, 0.05, len(t))
            env = np.minimum(1, t / 0.15) * np.minimum(1, (ln - t) / 0.35)
            add(out, tone * env * 0.6, t0); t0 += ln + rng.uniform(0.1, 0.4)
            if t0 > dur - 3: break
        t0 += 2.5
        if t0 > dur - 3: break
    return out / (np.abs(out).max() + 1e-9)


def reverb(x, sec=1.6, wet=0.28):
    n = int(SR * sec); ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / (sec / 4)); ir /= np.abs(ir).sum()
    m = 1 << (len(x) + n).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, m) * np.fft.rfft(ir, m), m)[:len(x)]
    return (1 - wet) * x + wet * y * (np.abs(x).max() / (np.abs(y).max() + 1e-9))


def finish(x, fade_in=2.0, fade_out=3.0, peak=0.9, target_rms_db=-22.0):
    """Equal loudness across tracks (RMS over the audible part), then cap the peak."""
    active = x[np.abs(x) > 0.02 * np.abs(x).max()]
    x = x * (10 ** (target_rms_db / 20) / (np.sqrt((active ** 2).mean()) + 1e-9))
    x = x / max(1.0, np.abs(x).max() / peak)
    n = len(x); x[:int(fade_in * SR)] *= np.linspace(0, 1, int(fade_in * SR)); x[-int(fade_out * SR):] *= np.linspace(1, 0, int(fade_out * SR))
    return x


def stereo(x, width=0.012):
    d = int(SR * width); return np.stack([x, np.concatenate([np.zeros(d), x[:-d]])], axis=1)      # tiny Haas delay = width


def write_mp3(name, x):
    pcm = (np.clip(stereo(x), -1, 1) * 32767).astype("<i2").tobytes()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "2", "-i", "-", "-c:a", "libmp3lame", "-q:a", "3", f"music/{name}.mp3"], input=pcm, check=True)


def main():
    tp = tanpura()
    bells = np.zeros(SR * DUR)
    for k, t in enumerate(np.arange(3, DUR - 8, 4.5)): add(bells, bell(SA * (4 if k % 3 else 6)) * (0.7 + 0.3 * rng.random()), t)
    tracks = {
        "aikyam_tanpura_calm": finish(reverb(tp)),
        "aikyam_temple_bells": finish(reverb(0.7 * tanpura(1.0) + 0.55 * bells, 2.2, 0.35)),
        "aikyam_festive_rhythm": finish(reverb(0.9 * dhol() + 0.35 * tanpura(1.0), 0.9, 0.18)),
        "aikyam_flute_meditative": finish(reverb(0.85 * flute() + 0.5 * tanpura(1.0), 2.0, 0.32)),
    }
    for n, x in tracks.items():
        write_mp3(n, x)
    own = dict(licence="AIKYAM-OWNED", licenceVerified=True, source="generated by tools/make_music.py (numpy synthesis); no third-party samples", attribution=None)
    lib = {"_comment": "Only tracks whose licence is in AIKYAM_ALLOWED_MUSIC_LICENCES and licenceVerified=true are ever used. energy: 0 calm .. 1 driving.", "tracks": [
        {**own, "id": "aikyam_tanpura_calm", "title": "Tanpura Drone (calm)", "file": "aikyam_tanpura_calm.mp3", "kind": "drone", "moods": ["calm", "devotional", "meditative"], "energy": 0.15,
         "tags": {"labels": ["deity", "idol", "devotees", "abhishekam", "lamps", "temple_architecture", "priest"], "deities": [], "rituals": ["ritual_13", "ritual_14"], "festivals": []}},
        {**own, "id": "aikyam_temple_bells", "title": "Temple Bells over Tanpura", "file": "aikyam_temple_bells.mp3", "kind": "drone", "moods": ["sacred", "aarti"], "energy": 0.3,
         "tags": {"labels": ["aarti", "lamps", "deity", "idol", "priest", "abhishekam"], "deities": [], "rituals": ["ritual_12", "ritual_14"], "festivals": []}},
        {**own, "id": "aikyam_festive_rhythm", "title": "Festive Dhol Rhythm", "file": "aikyam_festive_rhythm.mp3", "kind": "rhythmic", "moods": ["festive", "energetic"], "energy": 0.85,
         "tags": {"labels": ["procession", "ritual_dance", "crowd", "decorations", "fireworks"], "deities": [], "rituals": ["ritual_16", "ritual_17"],
                  "festivals": ["festival_9", "festival_10", "festival_11", "festival_12", "festival_13", "festival_14"]}},
        {**own, "id": "aikyam_flute_meditative", "title": "Bhoopali Flute (meditative)", "file": "aikyam_flute_meditative.mp3", "kind": "melodic", "moods": ["calm", "devotional", "darshan"], "energy": 0.25,
         "tags": {"labels": ["deity", "idol", "priest", "devotees", "temple_architecture", "flowers", "decorations"], "deities": [], "rituals": [], "festivals": []}},
    ]}
    for t in lib["tracks"]: t["durationSeconds"] = DUR
    json.dump(lib, open("music/library.json", "w"), indent=1)
    print("wrote", len(tracks), "tracks + music/library.json")


if __name__ == "__main__":
    main()
