#!/usr/bin/env python3
"""Koshur Tools — FastAPI backend.

Endpoints:
  GET  /                        the single-page app
  POST /api/transcribe          audio upload -> Kashmiri text
  POST /api/translate           {"text"} -> {"translation"}
  POST /api/video-subtitles     video upload -> {srt, video_url}
  GET  /api/download/<name>     generated files (subtitled videos)
  GET  /api/inpage-families     InPage converter families (needs data file)
  POST /api/inpage              {"text", "family"} -> {"converted"}

Run:  uvicorn app:app --port 8000
Models load lazily on first use and stay loaded.
"""
import json, subprocess, tempfile, uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import koshur_models as km

BASE = Path(__file__).parent
OUT_DIR = BASE / "generated"
OUT_DIR.mkdir(exist_ok=True)
INPAGE_FILE = BASE / "data" / "observations.json"

app = FastAPI(title="Koshur Tools")


def to_wav16k(src: Path, dst: Path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)


def fmt_ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / ("in_" + (file.filename or "audio"))
        raw.write_bytes(await file.read())
        wav = Path(td) / "a.wav"
        try:
            to_wav16k(raw, wav)
        except subprocess.CalledProcessError:
            raise HTTPException(400, "Could not read that audio file. Try WAV, MP3, M4A or WebM.")
        pcm, sr = km.read_wav(wav)
        if sr != 16000:
            raise HTTPException(400, "Audio conversion failed.")
        text = " ".join(e["text"] for e in km.get_asr().transcribe_pcm(pcm, sr) if e["text"])
    return {"text": text}


class TranslateIn(BaseModel):
    text: str


@app.post("/api/translate")
async def translate(body: TranslateIn):
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Type something in English first.")
    return {"translation": km.get_translator().translate(text)}


@app.post("/api/video-subtitles")
async def video_subtitles(file: UploadFile = File(...)):
    vid_id = uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / ("in_" + (file.filename or "video.mp4"))
        raw.write_bytes(await file.read())
        wav = Path(td) / "a.wav"
        try:
            to_wav16k(raw, wav)
        except subprocess.CalledProcessError:
            raise HTTPException(400, "Could not read that video file.")
        pcm, sr = km.read_wav(wav)
        entries = [e for e in km.get_asr().transcribe_pcm(pcm, sr) if e["text"]]
        if not entries:
            raise HTTPException(422, "No Kashmiri speech was recognised in this video.")
        srt = ""
        for i, e in enumerate(entries, 1):
            srt += f"{i}\n{fmt_ts(e['start'])} --> {fmt_ts(e['end'])}\n{e['text']}\n\n"
        srt_path = Path(td) / "subs.srt"
        srt_path.write_text(srt, encoding="utf-8")
        out_name = f"subtitled_{vid_id}.mp4"
        out_path = OUT_DIR / out_name
        mux = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw),
                              "-i", str(srt_path), "-c", "copy", "-c:s", "mov_text",
                              str(out_path)], capture_output=True)
        video_url = f"/api/download/{out_name}" if mux.returncode == 0 and out_path.exists() else None
    return {"srt": srt, "video_url": video_url}


@app.get("/api/download/{name}")
async def download(name: str):
    p = OUT_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(p, filename=p.name)


_inpage_cache = None


def _inpage():
    global _inpage_cache
    if _inpage_cache is None:
        if not INPAGE_FILE.exists():
            return None
        d = json.loads(INPAGE_FILE.read_text(encoding="utf-8"))
        mapping = d.get("mapping", {})
        fams = {}
        for key, val in mapping.items():
            fam, _, code = key.partition(":")
            fams.setdefault(fam, {})[code] = val.get("text", "")
        _inpage_cache = fams
    return _inpage_cache


@app.get("/api/inpage-families")
async def inpage_families():
    fams = _inpage()
    if fams is None:
        return {"available": False, "families": []}
    return {"available": True, "families": sorted(fams.keys())}


class InPageIn(BaseModel):
    text: str
    family: str


@app.post("/api/inpage")
async def inpage(body: InPageIn):
    fams = _inpage()
    if fams is None:
        raise HTTPException(501, "InPage data file (data/observations.json) is not installed on this server.")
    table = fams.get(body.family)
    if table is None:
        raise HTTPException(400, f"Unknown font family: {body.family}")
    out = []
    for ch in body.text:
        o = ord(ch)
        if 0xF000 <= o <= 0xF0FF:
            out.append(table.get(f"F0{o - 0xF000:02X}", ch))
        else:
            out.append(ch)
    return {"converted": " ".join("".join(out).split())}


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
