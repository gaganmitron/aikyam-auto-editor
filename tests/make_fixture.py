"""Builds tests/fixtures/sample-temple.mp4: SYNTHETIC (generated colour scenes + espeak speech + bell tones).
It exercises the pipeline; it is not real temple footage. Supply a real clip for quality evaluation."""
import subprocess, sys, tempfile, os

def build(out):
    td = tempfile.mkdtemp()
    speech = os.path.join(td, "s.wav")
    subprocess.run(["espeak-ng", "-s", "130", "-w", speech,
        "Welcome to Chamundeshwari temple. Now the evening aarti begins. Devotees offer flowers and lamps to the goddess Durga."], check=True)
    # 3 visually distinct 10 s scenes (warm lamp-like gradients with detail), 30 s total, 1280x720
    v = ["-f", "lavfi", "-i", "testsrc2=s=1280x720:r=25:d=10,eq=saturation=1.4:brightness=0.05,hue=h=20",
         "-f", "lavfi", "-i", "mandelbrot=s=1280x720:r=25,trim=duration=10,hue=h=40",
         "-f", "lavfi", "-i", "life=s=320x180:mold=10:r=25:ratio=0.1:death_color=#301000:life_color=#ffa020,scale=1280:720:flags=neighbor,trim=duration=10,format=yuv420p"]
    a = ["-i", speech, "-f", "lavfi", "-i", "sine=f=880:d=30,volume=0.3,afade=t=out:st=2:d=1,aresample=16000",
         "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:a=0.02"]
    g = ("[0:v][1:v][2:v]concat=n=3:v=1:a=0[v];[3:a]apad=whole_dur=30[sp];"
         "[4:a]volume='if(lt(mod(t\,7)\,0.4)\,1\,0)':eval=frame[bell];[sp][bell][5:a]amix=inputs=3:duration=first:normalize=0[a]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *v, *a, "-filter_complex", g, "-map", "[v]", "-map", "[a]",
                    "-t", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", out], check=True)

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample-temple.mp4")
