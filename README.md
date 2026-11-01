# Koshur Tools — by Aazim Ameen

A single-page web app for Kashmiri: editor + fonts + on-screen keyboard,
speech-to-text, English→Kashmiri translation, video auto-subtitles, and an
InPage→Unicode converter. Design: ivory / white / deep forest green / sage /
terracotta, per Aazim's own design spec.

## Run

```bash
pip install -r requirements.txt
uvicorn app:app --port 8000
# open http://localhost:8000
```

Needs `ffmpeg` on the PATH (audio/video conversion). Models download
automatically on first use from Hugging Face into `models/`
(Aazu34/kashmiri-asr-ctc + Aazu34/english-kashmiri-translation), or point
`KOSHUR_ASR_DIR` / `KOSHUR_TR_DIR` at local model folders.

## What's inside (mapped to the Koshur Editor app's features)

| App feature | Here |
|---|---|
| Text editor + Kashmiri keyboard | Editor tab, on-screen keyboard |
| Fonts + import your own .ttf/.otf | Editor tab (4 bundled fonts + Import) |
| Translation | Translate tab (English → Kashmiri, beam search) |
| Speech-to-text / dictation | Speech to Text tab (upload + microphone) |
| Video auto-subtitles | Video Subtitles tab (SRT + subtitled MP4) |
| Roman → Kashmiri | Editor tab button (basic transliteration) |
| Text-to-speech | Editor "Listen" (browser voice, Urdu/Arabic fallback — no Kashmiri voice exists in browsers) |
| Poster/design export | Editor "Download poster PNG" (1080×1350) |
| InPage → Unicode | InPage Converter tab (needs the data file, below) |
| Saved projects | Editor Save/Load (browser localStorage) |

Not included: OCR / handwriting scan (no OCR engine bundled), Koshur Studio
publishing + accounts/cloud sync (needs a hosted account service), thermal
printing, and the app's full FFmpeg video editor (only subtitle generation
is here).

## InPage data file

The converter needs `data/observations.json` — the font-encoding table from
the Koshur Editor app's assets (`assets/flutter_assets/assets/inpage_pdf/`).
It is present in this working copy for local use. It comes from inside the
app, so **do not ship it in a public deployment** — remove `data/` before
putting this site online; the tab then shows a clean "not installed" state.

## Files

- `app.py` — FastAPI backend (all endpoints)
- `koshur_models.py` — ASR + translation engines (verified inference code)
- `static/index.html` — the whole frontend (design + all tabs)
- `static/fonts/` — Jameel Nastaleeq, Narqalam, Noto Naskh, Diwany Thuluth
