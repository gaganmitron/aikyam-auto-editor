"""Shot analysis on synthetic footage with KNOWN defects: measures must rank them, and the in/out search must avoid them."""
import subprocess
import numpy as np, pytest
from aikyam_video.creative.model import Shot
from aikyam_video.creative.shots import Population, analyze_window, best_window

TS = "testsrc2=s=640x360:r=25"


def make(path, parts):
    """parts: list of (seconds, filter chain on testsrc2); concatenated into one 25 fps video."""
    inputs, chains = [], []
    for i, (d, vf) in enumerate(parts):
        inputs += ["-f", "lavfi", "-i", f"{TS}:d={d}"]; chains.append(f"[{i}:v]{vf},scale=640:360,setsar=1,format=yuv420p[p{i}]")
    fg = ";".join(chains) + ";" + "".join(f"[p{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=1:a=0[v]"
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fg, "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True)


SHARP = "null"
BLUR = "gblur=sigma=9"
SHAKY = "crop=w=560:h=300:x='40+38*sin(t*47)':y='30+28*cos(t*61)'"           # violent, high-frequency wobble
DARK = "eq=brightness=-0.75:contrast=0.4"


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("s") / "v.mp4")
    make(p, [(4, SHARP), (4, BLUR), (4, SHAKY), (4, DARK)])
    return p


def shot(clip, i):
    a = i * 4.0
    return Shot(f"s{i}", a, a + 4.0, f"m{i}", 0.5, {}, [], None, analyze_window(clip, a, a + 4.0, None))


def test_quality_ranks_defects_below_clean_footage(clip):
    shots = [shot(clip, i) for i in range(4)]; pop = Population(shots)
    q = [float(pop.slot_quality(s.slots).mean()) for s in shots]
    assert q[0] > q[1] + 0.15                 # sharp beats blurred
    assert q[0] > q[2] + 0.15                 # steady beats shaky
    assert q[0] > q[3] + 0.10                 # detailed beats crushed/dark-flat
    assert shots[2].slots.jitter and np.mean(shots[2].slots.jitter) > 3 * np.mean(shots[0].slots.jitter)


def test_motion_and_concentration_are_measured(clip):
    s0, s2 = shot(clip, 0), shot(clip, 2)
    assert np.mean(s2.slots.motion) > np.mean(s0.slots.motion)                # the shaking clip changes more frame to frame
    assert all(0 <= c <= 1 for c in s0.slots.concentration) and all(0 <= l <= 1 for l in s0.slots.luma)


def test_best_window_avoids_the_blurry_lead_in(tmp_path):
    p = str(tmp_path / "lead.mp4"); make(p, [(3, BLUR), (7, SHARP)])            # 3 s of blur then 7 s sharp: the in-point must not be in the blur
    sh = Shot("s", 0.0, 10.0, "m", 0.5, {}, [], None, analyze_window(p, 0.0, 10.0, None))
    a, b, q = best_window(sh, Population([sh]), 5.0)
    assert a >= 2.5 and b - a == pytest.approx(5.0, abs=0.51) and b <= 10.0     # starts in (or right at the start of) the sharp part
    a2, b2, _ = best_window(sh, Population([sh]), 20.0)                          # asking for more than exists returns the whole shot
    assert (a2, b2) == (0.0, 10.0)


def test_best_window_out_point_can_snap_to_quiet_but_never_leaves_the_shot(clip):
    sh = shot(clip, 0); pop = Population([sh])
    a, b, _ = best_window(sh, pop, 2.0, snap=lambda t: t + 5.0)                  # a hostile snap that tries to jump far past the end
    assert sh.start <= a < b <= sh.end


def test_population_is_adaptive_not_absolute(clip):
    """The same slot ranks differently in a different population: quality is relative to the footage, not a pixel threshold."""
    blur_only = Population([shot(clip, 1)]); mixed = Population([shot(clip, 0), shot(clip, 1)])
    v = np.array([np.median(shot(clip, 1).slots.sharpness)])
    assert blur_only.rank("sharpness", v)[0] > mixed.rank("sharpness", v)[0] + 0.2   # "sharp for THIS footage" vs among sharper footage
