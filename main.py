import os
import tempfile
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Req(BaseModel):
    videoUrl: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
def transcribe(req: Req):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    video_url = req.videoUrl.strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="videoUrl is required")

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "audio.m4a")

        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--no-playlist",
            "-o", out_path,
            video_url
        ]

        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="yt-dlp timed out")

        if p.returncode != 0:
            raise HTTPException(
                status_code=502,
                detail=f"yt-dlp failed: {p.stderr[-800:]}"
            )

        if not os.path.exists(out_path):
            raise HTTPException(status_code=502, detail="audio file not created")

        try:
            with open(out_path, "rb") as f:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        text = (resp.text or "").strip()
        if len(text) < 10:
            raise HTTPException(status_code=502, detail="empty transcript")

        return {"transcript": text}
