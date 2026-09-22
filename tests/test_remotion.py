"""Remotion renders the SAME EditPlan. Skipped when Node/npm install/Chromium are unavailable."""
import subprocess
import numpy as np, pytest
from aikyam_video import ffmpeg as ff
from aikyam_video.render_remotion import available, render_format

pytestmark = pytest.mark.skipif(not available(), reason="needs `npm install` in renderer/remotion, node and chromium")


def test_remotion_renders_plan_with_overlays_and_audio(tmp_path):
    from tests.test_features import moving_dot_video
    src = str(tmp_path / "s.mp4"); moving_dot_video(src, w=1280, h=720, d=6)
    plan = {"schemaVersion": 1, "source": {"videoId": "v", "path": src, "durationSeconds": 6, "templeId": "temple_123"}, "outputFormat": "REEL", "durationSeconds": 3, "aspectRatio": "9:16",
            "segments": [{"start": 0, "end": 3, "reason": "x", "score": .9, "subjectPath": [[0, 0.06], [3, 0.5]]}],
            "captions": {"enabled": True, "language": "en", "mode": "sentence", "style": {}, "cues": [{"start": 0.2, "end": 2.8, "text": "Om Namah Shivaya", "words": []}]},
            "overlays": {"template": "ritual_highlight", "temple": "Chamundeshwari Temple", "deity": "Durga", "ritual": "Evening Aarti"},
            "transitions": {"type": "cut", "durationSeconds": 0}, "audio": {"preserveOriginal": True, "normalize": True}, "thumbnail": {}}
    out = str(tmp_path / "o.mp4"); render_format(plan, src, out, "reel", str(tmp_path))
    i = ff.probe(out)
    assert (i.width, i.height, i.video_codec) == (1080, 1920, "h264") and i.has_audio and abs(i.duration - 3) < 0.3
    def frame(t, vf):
        return np.frombuffer(subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", out, "-frames:v", "1", "-vf", vf + ",format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout, np.uint8)
    assert (frame(1.5, "crop=iw:ih*0.06:0:ih*0.06") > 200).sum() > 1000            # temple title drawn near the top
    assert (frame(1.5, "crop=iw:ih*0.12:0:ih*0.80") > 200).sum() > 500              # caption/ritual text drawn near the bottom
    # the tracked path pans: the white square is on the left early in the clip, further right later
    def square_x(t):
        f = frame(t, "crop=iw:ih*0.14:0:ih*0.18,scale=108:27").reshape(27, 108)   # band with the square, free of overlay text
        ys, xs = np.where(f > 200); return xs.mean() if len(xs) else None
    a, b = square_x(0.3), square_x(2.7)
    assert a is not None and b is not None and b > a + 10             # the square drifts right within the window: camera lags/pans as planned
