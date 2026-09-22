import os, subprocess, sys
from pathlib import Path
import pytest

FIX = Path(__file__).parent / "fixtures" / "sample-temple.mp4"


@pytest.fixture(scope="session")
def sample() -> str:
    """tests/fixtures/sample-temple.mp4 (synthetic; generated on demand). Override with AIKYAM_SAMPLE=/path/real.mp4."""
    if os.environ.get("AIKYAM_SAMPLE"):
        return os.environ["AIKYAM_SAMPLE"]
    if not FIX.exists():
        FIX.parent.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_fixture.py"), str(FIX)], check=True)
    return str(FIX)


@pytest.fixture(scope="session")
def pipeline_out(sample, tmp_path_factory):
    from aikyam_video import pipeline
    out = str(tmp_path_factory.mktemp("out"))
    files = pipeline.run(sample, out, "process", pipeline.Options(temple_id="temple_123", whisper_model="base"))
    return out, files
