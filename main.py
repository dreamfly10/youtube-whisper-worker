import os
import glob
import tempfile
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()


class Req(BaseModel):
    videoUrl: str


@app.get("/health")
def health():
    return {"ok": True}


def run_yt_dlp(video_url: str, workdir: str) -> str:
    """
    Downloads best audio to workdir using yt-dlp.
    Returns the path to the downloaded audio file.
    """
    # IMPORTANT: don't hardcode audio.m4a; YouTube often returns webm/opus.
    out_template = os.path.join(workdir, "audio.%(ext)s")

    # Prefer m4a if available; otherwise fallback to any bestaudio
    fmt = "bestaudio[ext=m4a]/bestaudio"

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--no-warnings",
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "--extractor-args",
        "youtube:player_client=android",
        "-f",
        fmt,
        "-o",
        out_template,
        video_url,
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="yt-dlp timed out")

    if p.returncode != 0:
        # Return last part of stderr to keep it readable
        tail = (p.stderr or p.stdout or "")[-1500:]
        raise HTTPException(status_code=502, detail=f"yt-dlp failed:\n{tail}")

    # Find whatever file yt-dlp actually produced
    candidates = glob.glob(os.path.join(workdir, "audio.*"))
    candidates = [c for c in candidates if os.path.isfile(c)]

    if not candidates:
        raise HTTPException(status_code=502, detail="yt-dlp reported success but no audio file was created")

    # Pick the largest file (most likely the real audio)
    candidates.sort(key=lambda pth: os.path.getsize(pth), reverse=True)
    audio_path = candidates[0]

    # Guard against empty files
    if os.path.getsize(audio_path) < 1024:
        raise HTTPException(
            status_code=502,
            detail=f"yt-dlp produced an empty/too-small file: {os.path.basename(audio_path)}",
        )

    return audio_path


@app.post("/transcribe")
def transcribe(req: Req):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    video_url = (req.videoUrl or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="videoUrl is required")

    # Create client lazily so the app can boot even if env vars are missing
    client = OpenAI(api_key=api_key)

    with tempfile.TemporaryDirectory() as td:
        audio_path = run_yt_dlp(video_url, td)

        try:
            with open(audio_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"OpenAI Whisper failed: {str(e)}")

        text = (getattr(resp, "text", None) or "").strip()
        if len(text) < 10:
            raise HTTPException(status_code=502, detail="empty transcript")

        return {"transcript": text}
