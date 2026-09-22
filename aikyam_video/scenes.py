"""Shot detection, scene grouping, keyframes (section 6)."""
from __future__ import annotations
import os
from typing import List, Tuple
import cv2
from scenedetect import ContentDetector, SceneManager, open_video
from .models import Scene
from . import ffmpeg as ff


def detect_shots(path: str, threshold: float = 27.0, min_len_s: float = 1.0) -> List[Tuple[float, float]]:
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(1, int(min_len_s * video.frame_rate))))
    sm.detect_scenes(video)
    shots = [(a.get_seconds(), b.get_seconds()) for a, b in sm.get_scene_list()]
    dur = ff.probe(path).duration
    if not shots:  # no cuts: whole video is one shot
        return [(0.0, dur)]
    # Some containers (e.g. WebM/VP9) advertise a bogus frame rate (120 fps for ~29 fps footage), which makes frame
    # numbers -> seconds wrong. The last shot always ends at the last frame, so rescale to the real duration.
    k = dur / shots[-1][1] if shots[-1][1] > 0 else 1.0
    return [(a * k, b * k) for a, b in shots] if abs(k - 1) > 0.02 else shots


def group_scenes(shots: List[Tuple[float, float]], min_scene_s: float = 4.0) -> List[Tuple[float, float]]:
    """Merge consecutive shots until each scene is >= min_scene_s. Trailing short scene joins the previous."""
    scenes: List[Tuple[float, float]] = []
    cur = None
    for a, b in shots:
        cur = (a, b) if cur is None else (cur[0], b)
        if cur[1] - cur[0] >= min_scene_s:
            scenes.append(cur); cur = None
    if cur is not None:
        scenes[-1:] = [(scenes[-1][0], cur[1])] if scenes else [cur]
    return scenes


def detect_scenes(path: str, out_dir: str, threshold: float = 27.0, min_scene_s: float = 4.0) -> List[Scene]:
    os.makedirs(os.path.join(out_dir, "keyframes"), exist_ok=True)
    groups = group_scenes(detect_shots(path, threshold), min_scene_s)
    scenes = []
    for i, (a, b) in enumerate(groups):
        kf = os.path.join(out_dir, "keyframes", f"scene_{i}.jpg")
        ff.frame_at(path, a + (b - a) / 2, kf)  # middle frame: avoids transition frames
        scenes.append(Scene(sceneId=f"scene_{i}", start=round(a, 3), end=round(b, 3), keyframe=kf))
    return scenes
