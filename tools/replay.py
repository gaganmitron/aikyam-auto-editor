"""Re-run plan+render on an already-analysed directory (analysis artifacts on disk): usage  python tools/replay.py <dir> <source video> [--music off] [--reels N]"""
import sys
from aikyam_video import stages
from aikyam_video.options import Options
d, src = sys.argv[1], sys.argv[2]
o = Options(formats=["reel"], music="off" if "--music" in sys.argv and sys.argv[sys.argv.index("--music") + 1] == "off" else "auto",
            reels=int(sys.argv[sys.argv.index("--reels") + 1]) if "--reels" in sys.argv else 1)
stages.plan(src, d, o); print(stages.render(src, d, o))
