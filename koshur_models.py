#!/usr/bin/env python3
"""Koshur Tools — model engines (self-contained).

ASR: Kashmiri CTC model (int8 ONNX). Frontend built exactly from the
bundle's asr_config.json; greedy CTC decode (measured best for this model —
beam search and LM rescoring gave no gain, 2026-11-01).

Translation EN->KS: encoder-decoder ONNX, IndicTrans2 convention
("eng_Latn kas_Arab" prefix, decoder starts at </s> id 2), beam search
width 4 + length normalisation + trigram-repeat blocking (measured better
than greedy, which catastrophically loops on some inputs).

Model files resolve in this order:
  1. $KOSHUR_ASR_DIR / $KOSHUR_TR_DIR
  2. ./models/asr and ./models/translation next to this file
  3. downloaded from Hugging Face (Aazu34/kashmiri-asr-ctc,
     Aazu34/english-kashmiri-translation) into ./models/
"""
import json, os, unicodedata, wave
from pathlib import Path
import numpy as np

BASE = Path(__file__).parent
EOS = 2


def _resolve(env_name, sub, repo):
    env = os.environ.get(env_name)
    if env and Path(env).exists():
        return Path(env)
    local = BASE / "models" / sub
    need = {"asr": "kashmiri_asr_ctc.int8.onnx", "translation": "encoder_model.onnx"}[sub]
    if (local / need).exists():
        return local
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, local_dir=str(local))
    return local


# ---------------- ASR ----------------
class KashmiriASR:
    def __init__(self, model_dir):
        import onnxruntime as ort
        self.cfg = json.loads((model_dir / "asr_config.json").read_text())
        self.hann = np.fromfile(model_dir / "hann_window.f32", dtype=np.float32)
        fb = np.fromfile(model_dir / "mel_filterbank.f32", dtype=np.float32)
        self.mel_fb = fb.reshape(self.cfg["mel_features"], self.cfg["fft_bins"])
        self.pieces = json.loads((model_dir / "kashmiri_tokenizer_pieces.json").read_text(encoding="utf-8"))
        self.sess = ort.InferenceSession(str(model_dir / "kashmiri_asr_ctc.int8.onnx"),
                                         providers=["CPUExecutionProvider"])

    def features(self, x):
        cfg = self.cfg
        y = np.empty_like(x); y[0] = x[0]; y[1:] = x[1:] - cfg["preemphasis"] * x[:-1]
        n_fft, win, hop = cfg["n_fft"], cfg["window_size"], cfg["window_stride"]
        yp = np.pad(y, n_fft // 2, mode="reflect")
        n_frames = 1 + (len(yp) - n_fft) // hop
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        window = np.concatenate([self.hann, np.zeros(n_fft - win, dtype=np.float32)])
        frames = yp[idx] * window
        spec = np.fft.rfft(frames, n=n_fft, axis=1)
        power = (np.abs(spec) ** cfg["mag_power"]).T
        mel = self.mel_fb @ power
        mel = np.log(np.maximum(mel, cfg["log_zero_guard_value"]))
        mean = mel.mean(axis=1, keepdims=True)
        std = mel.std(axis=1, keepdims=True, ddof=1 if cfg["std_unbiased"] else 0)
        return ((mel - mean) / (std + cfg["normalization_epsilon"])).astype(np.float32)

    def transcribe(self, pcm):
        if len(pcm) < 1600:
            return ""
        mel = self.features(np.asarray(pcm, dtype=np.float32))
        logits, enc_len = self.sess.run(None, {
            "mel_features": mel[None],
            "feature_length": np.array([mel.shape[1]], dtype=np.int64)})
        ids = logits[0].argmax(axis=-1)[: int(enc_len[0])]
        out, prev = [], -1
        for i in ids:
            if i != prev and i != self.cfg["blank_id"]:
                out.append(int(i))
            prev = i
        text = "".join(self.pieces[i] for i in out).replace("▁", " ").strip()
        return unicodedata.normalize("NFC", text)

    def transcribe_pcm(self, pcm, sr, chunk=10.0):
        """Returns [{start, end, text}] — used for subtitles too."""
        entries, t, total = [], 0.0, len(pcm) / sr
        while t < total:
            end = min(t + chunk, total)
            entries.append({"start": t, "end": end,
                            "text": self.transcribe(pcm[int(t * sr):int(end * sr)])})
            t = end
        return entries


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sw}")
    if ch > 1:
        pcm = pcm.reshape(-1, ch)[:, 0]
    return pcm, sr


# ---------------- Translation ----------------
def _log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


class EnglishToKashmiri:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.src_tok = Tokenizer.from_file(str(model_dir / "tokenizer_src.json"))
        self.tgt_tok = Tokenizer.from_file(str(model_dir / "tokenizer_tgt.json"))
        self.enc = ort.InferenceSession(str(model_dir / "encoder_model.onnx"),
                                        providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(model_dir / "decoder_model.onnx"),
                                        providers=["CPUExecutionProvider"])

    def translate(self, text, beam_width=4, max_len=80, alpha=0.7):
        enc_ids = self.src_tok.encode(f"eng_Latn kas_Arab {text}").ids
        input_ids = np.array([enc_ids], dtype=np.int64)
        attn = np.ones_like(input_ids)
        hidden = self.enc.run(None, {"input_ids": input_ids, "attention_mask": attn})[0]
        B = beam_width
        live = [([EOS], 0.0)]
        finished = []
        for _ in range(max_len):
            if not live:
                break
            seqs = [b[0] for b in live]
            maxlen = max(len(s) for s in seqs)
            padded = np.full((len(seqs), maxlen), EOS, dtype=np.int64)
            for i, s in enumerate(seqs):
                padded[i, :len(s)] = s
            hb = np.repeat(hidden, len(seqs), axis=0)
            ab = np.repeat(attn, len(seqs), axis=0)
            logits = self.dec.run(["logits"], {
                "input_ids": padded,
                "encoder_attention_mask": ab,
                "encoder_hidden_states": hb})[0]
            cands = []
            for i, (seq, score) in enumerate(live):
                lp = _log_softmax(logits[i, len(seq) - 1].astype(np.float64))
                top = np.argpartition(lp, -B)[-B:]
                seen_tri = {tuple(seq[j:j + 3]) for j in range(len(seq) - 2)}
                for c in top:
                    c = int(c)
                    ns = score + float(lp[c])
                    if c == EOS:
                        finished.append((seq + [c], ns))
                    elif len(seq) >= 2 and tuple(seq[-2:] + [c]) in seen_tri:
                        continue
                    else:
                        cands.append((seq + [c], ns))
            cands.sort(key=lambda x: x[1] / (len(x[0]) ** alpha), reverse=True)
            live = cands[:B]
        pool = finished if finished else live
        best = max(pool, key=lambda x: x[1] / (len(x[0]) ** alpha))
        return self.tgt_tok.decode(best[0], skip_special_tokens=True)


# ---------------- lazy singletons ----------------
_asr = None
_tr = None


def get_asr():
    global _asr
    if _asr is None:
        _asr = KashmiriASR(_resolve("KOSHUR_ASR_DIR", "asr", "Aazu34/kashmiri-asr-ctc"))
    return _asr


def get_translator():
    global _tr
    if _tr is None:
        _tr = EnglishToKashmiri(_resolve("KOSHUR_TR_DIR", "translation",
                                         "Aazu34/english-kashmiri-translation"))
    return _tr
