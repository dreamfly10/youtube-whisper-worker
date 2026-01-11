import os
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()


# -----------------------------
# Models
# -----------------------------
class TranscribeReq(BaseModel):
    # Accept BOTH shapes to avoid mismatch issues:
    # - {"videoUrl": "..."} (your current)
    # - {"url": "..."}      (common for APIs / your Next.js might send this)
    videoUrl: Optional[str] = Field(default=None)
    url: Optional[str] = Field(default=None)


# -----------------------------
# Helpers
# -----------------------------
def get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # IMPORTANT: do NOT crash at import time
        # Return a clean error when /transcribe is called
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def pick_video_url(req: TranscribeReq) -> str:
    video_url = (req.videoUrl or req.url or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="videoUrl (or url) is required")
    return video_url


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/transcribe")
def transcribe(req: TranscribeReq):
    # Will only error when endpoint is called, not at server startup
    client = get_openai_client()

    video_url = pick_video_url(req)

    # Optional: yt-dlp sometimes needs a HOME set in containers
    # (Railway usually has one, but this avoids edge cases)
    env = os.environ.copy()
    env.setdefault("HOME", "/tmp")

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "audio.m4a")

        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--no-playlist",
            "-o", out_path,
            video_url,
        ]

        try:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="yt-dlp timed out")

        if p.returncode != 0:
            # Return the last chunk of stderr for debugging
            err = (p.stderr or "").strip()
            tail = err[-1200:] if err else "unknown error"
            raise HTTPException(status_code=502, detail=f"yt-dlp failed: {tail}")

        if not os.path.exists(out_path):
            raise HTTPException(status_code=502, detail="audio file not created")

        try:
            with open(out_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"whisper failed: {str(e)}")

        text = (getattr(resp, "text", "") or "").strip()
        if len(text) < 10:
            raise HTTPException(status_code=502, detail="empty transcript")

        return {"transcript": text}
