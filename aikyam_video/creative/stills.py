"""Ken Burns for stills: subject-aware 2D camera move, expressed as plan data (`motion`), rendered by FFmpeg.

Rules (research: documentary practice + AutoFlip's 'keep the subject, never crop into softness'):
  * the move ENDS tight on the subject (face box if found, else the detail centroid) and STARTS wider; every other still reverses (pull-out) so a run of stills breathes;
  * zoom is bounded by resolution (but never below the 8% drift: a low-res still is already upscaled at zoom 1, 8% more is invisible): the window must keep >= MIN_PX_FRAC of the output width in real source pixels (no upscaling blur);
  * with a face, the tightest window still holds the face at <= FACE_FILL of its width; without a subject the move is a slow 8% drift;
  * eased (smoothstep), never linear: motion starts and stops softly.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

MIN_PX_FRAC = 0.75       # source pixels across the window >= 75% of the output width
FACE_FILL = 0.55         # a face may fill at most 55% of the window width
Z_MAX = 1.5              # beyond this a still stops reading as a photograph and starts reading as a crop
Z_DRIFT = 1.08


def motion_for(size, focus, aspect: float, index: int = 0, out_w: int = 1080) -> Dict:
    """{'zoom': [z0, z1], 'center': [[x, y], [x, y]], 'easing'} for an image of `size`=[w, h] with subject `focus`=[cx, cy, fw, fh] (normalised)."""
    w, h = size; cx, cy, fw, fh = focus
    w0 = h * aspect if w / h >= aspect else float(w)                    # zoom-1 window width in source px
    zres = w0 / (MIN_PX_FRAC * out_w)                                   # resolution bound
    if fw > 0:
        zface = w0 / max(fw * w / FACE_FILL, 1.0)                        # zoom at which the face fills FACE_FILL of the window width
        zmax = float(np.clip(min(Z_MAX, zres, max(zface, 1.15)), Z_DRIFT, Z_MAX))
    else:
        zmax = Z_DRIFT
    # keep the tight window inside the image: half-window in normalised units
    hw, hh = w0 / zmax / 2 / w, (w0 / aspect) / zmax / 2 / h
    fx, fy = float(np.clip(cx, hw, 1 - hw)), float(np.clip(cy, hh, 1 - hh))
    wide = [float(np.clip(0.5 + 0.35 * (cx - 0.5), 0, 1)), float(np.clip(0.5 + 0.35 * (cy - 0.5), 0, 1))]     # the wide start leans toward the subject, it does not sit on it
    if index % 2 == 0:
        return {"zoom": [1.0, round(zmax, 3)], "center": [[round(wide[0], 3), round(wide[1], 3)], [round(fx, 3), round(fy, 3)]], "easing": "smooth"}
    return {"zoom": [round(zmax, 3), 1.0], "center": [[round(fx, 3), round(fy, 3)], [round(wide[0], 3), round(wide[1], 3)]], "easing": "smooth"}
