"""
STEP 2 — Macinamento (gira in LOCALE sulla 5090).

Per ogni file audio in data/raw/ :
    1. decodifica a mono 24kHz via ffmpeg (qualsiasi formato in ingresso)
    2. lo spezza in finestre da WINDOW_SEC (default 10 min) -> ogni finestra = 1 campione
    3. diarization (pyannote 3.1, forzata a 2 speaker) sulla finestra
    4. costruisce lo stereo "finto": speaker dominante -> canale SX, secondo -> canale DX
       (è l'approccio J-CHAT/j-moshi per trasformare dialogo mono in formato Moshi)
    5. filtro qualità: scarta finestre che NON sono vero dialogo a 2 voci
    6. salva il wav stereo 24k 16-bit in data/stereo/
    7. trascrive ogni canale con whisper-timestamped (word-level, lingua=it)
    8. salva il .json gemello {"alignments": [[parola,[start,end],"A"/"B"], ...]}

IDEMPOTENTE: se il .json di una finestra esiste già, la salta. Puoi interrompere e
rilanciare quando vuoi. I file problematici lasciano un .err e non bloccano il batch.

Uso:
    python 02_process.py

Richiede HF_TOKEN nell'ambiente e l'accettazione (una volta sola, sul sito HF) dei
gate dei modelli:
    https://huggingface.co/pyannote/speaker-diarization-3.1
    https://huggingface.co/pyannote/segmentation-3.0
"""
import os
import sys
import json
import warnings
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF
import soundfile as sf

import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Decodifica audio robusta via ffmpeg (mono, 24k, float32) -> numpy
# Evita problemi di backend codec di torchaudio su Windows.
# ---------------------------------------------------------------------------
def decode_mono(path: Path, sr: int = config.TARGET_SR) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fallito: {proc.stderr.decode(errors='ignore')[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def make_windows(total_sec: float):
    """Lista di (start_sec, end_sec). Scarta una coda finale troppo corta."""
    wins = []
    s = 0.0
    while s < total_sec:
        e = min(s + config.WINDOW_SEC, total_sec)
        if e - s >= config.MIN_DURATION_SEC:
            wins.append((s, e))
        s = e
    return wins


def load_models():
    from pyannote.audio import Pipeline
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    # token: da HF_TOKEN se impostato, altrimenti usa quello cachato da `hf auth login`
    tok = config.HF_TOKEN or None
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("Carico pyannote/speaker-diarization-3.1 ...")
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=tok)
    if diar is None:
        sys.exit("[X] Pipeline pyannote non caricata: token mancante o gate non accettati.\n"
                 "    1) hf auth login  (o $env:HF_TOKEN='hf_xxx')\n"
                 "    2) accetta i gate: huggingface.co/pyannote/speaker-diarization-3.1 "
                 "e .../segmentation-3.0")
    if dev == "cuda":
        diar.to(torch.device("cuda"))
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    [!] CUDA non disponibile: girerà su CPU (lentissimo).")

    # faster-whisper (CTranslate2): ~23x realtime sulla 5090, batched per spingere oltre.
    print(f"Carico faster-whisper '{config.WHISPER_MODEL}' ...")
    base = WhisperModel(config.WHISPER_MODEL, device=dev,
                        compute_type="float16" if dev == "cuda" else "int8")
    wmodel = BatchedInferencePipeline(model=base)
    return diar, wmodel


def transcribe_channel(wmodel, audio_24k: np.ndarray) -> list:
    """Ritorna lista di word-level [text, [start, end]] per un canale mono (faster-whisper)."""
    wav16 = AF.resample(torch.from_numpy(audio_24k), config.TARGET_SR, config.WHISPER_SR).numpy()
    # vad_filter salta il silenzio (i canali mascherati sono ~metà silenzio) -> più veloce.
    segments, _ = wmodel.transcribe(
        wav16, language=config.LANGUAGE, beam_size=1,
        word_timestamps=True, vad_filter=True, batch_size=16,
    )
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append([w.word.strip(), [float(w.start), float(w.end)]])
    return words


def process_window(seg_audio: np.ndarray, diar, wmodel, out_wav: Path, out_json: Path):
    sr = config.TARGET_SR
    wav_t = torch.from_numpy(seg_audio)[None, :]  # [1, N]

    # 1) diarization (forzata a 2 speaker per dialogo pulito)
    diar_out = diar({"waveform": wav_t, "sample_rate": sr},
                    min_speakers=config.DIAR_MIN_SPEAKERS,
                    max_speakers=config.DIAR_MAX_SPEAKERS)
    # pyannote 4.x ritorna DiarizeOutput; 3.x ritorna direttamente l'Annotation
    diarization = getattr(diar_out, "speaker_diarization", diar_out)

    segs = defaultdict(list)
    durs = defaultdict(float)
    for turn, _, spk in diarization.itertracks(yield_label=True):
        segs[spk].append((turn.start, turn.end))
        durs[spk] += (turn.end - turn.start)

    top = sorted(durs.items(), key=lambda x: x[1], reverse=True)[:2]
    if len(top) < 2:
        return "skip:1speaker"
    # filtro qualità: il secondo speaker deve parlare abbastanza -> vero dialogo
    ratio = top[1][1] / max(top[0][1], 1e-6)
    if ratio < config.MIN_SECONDARY_SPEAKER_RATIO:
        return f"skip:monologue({ratio:.2f})"

    spk_L, spk_R = top[0][0], top[1][0]
    N = seg_audio.shape[0]
    chL = np.zeros(N, dtype=np.float32)
    chR = np.zeros(N, dtype=np.float32)

    def paint(ch, spk):
        for s, e in segs[spk]:
            a = int(s * sr); b = min(int(e * sr), N)
            if a < N:
                ch[a:b] = seg_audio[a:b]

    paint(chL, spk_L)
    paint(chR, spk_R)

    # 2) salva stereo 24k 16-bit (SX = moshi/voce principale, DX = interlocutore)
    stereo = np.stack([chL, chR], axis=1)  # [N, 2] per soundfile
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), stereo, sr, subtype="PCM_16")

    # 3) trascrizione word-level dei due canali
    words_A = transcribe_channel(wmodel, chL)
    words_R = transcribe_channel(wmodel, chR)

    # "SPEAKER_MAIN" = etichetta che moshi-finetune si aspetta per il canale sinistro (la
    # voce di Moshi): train.py usa keep_main_only=True e SCARTA ogni altra etichetta.
    # Con "A"/"B" il flusso testo del training risultava VUOTO (bug scoperto col Run 1).
    alignments = [[t, ts, "SPEAKER_MAIN"] for t, ts in words_A] + [[t, ts, "B"] for t, ts in words_R]
    alignments.sort(key=lambda x: x[1][0])
    if not alignments:
        out_wav.unlink(missing_ok=True)
        return "skip:notext"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"alignments": alignments}, f, ensure_ascii=False)
    return "ok"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="processa solo N file (campione distribuito tra i canali) per test")
    args = ap.parse_args()

    config.STEREO_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in config.RAW_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in config.AUDIO_EXTS)
    if not files:
        sys.exit(f"[X] Nessun audio in {config.RAW_DIR}. Scarica con 01_download.py o "
                 f"copia lì le tue 300h già pronte.")

    if args.limit and len(files) > args.limit:
        # round-robin tra i canali (cartelle), così il campione tocca tutti
        from collections import defaultdict, OrderedDict
        by_ch = OrderedDict()
        for p in files:
            by_ch.setdefault(p.parent, []).append(p)
        picked, idx = [], 0
        while len(picked) < args.limit and any(idx < len(v) for v in by_ch.values()):
            for v in by_ch.values():
                if idx < len(v) and len(picked) < args.limit:
                    picked.append(v[idx])
            idx += 1
        files = picked
        print(f"[CAMPIONE] processo {len(files)} file (round-robin tra {len(by_ch)} canali)\n")
    print(f"Trovati {len(files)} file da processare.\n")

    diar, wmodel = load_models()

    n_ok = n_skip = n_err = 0
    for fi, path in enumerate(files, 1):
        stem = path.stem
        # nome univoco includendo la sottocartella (evita collisioni tra canali)
        rel = path.relative_to(config.RAW_DIR).with_suffix("")
        flat = "__".join(rel.parts)
        flat = "".join(c if c.isalnum() or c in "-_" else "-" for c in flat)[:120]

        print(f"\n--- [{fi}/{len(files)}] {path.name} ---")
        try:
            audio = decode_mono(path)
            total = len(audio) / config.TARGET_SR
            wins = make_windows(total)
            print(f"    durata {total/60:.1f} min -> {len(wins)} finestre")

            for wi, (ws, we) in enumerate(wins):
                base = config.STEREO_DIR / f"{flat}_{wi:03d}"
                out_wav, out_json = base.with_suffix(".wav"), base.with_suffix(".json")
                err_file = base.with_suffix(".err")
                if out_json.exists():
                    continue  # già fatto
                if err_file.exists():
                    continue
                seg = audio[int(ws * config.TARGET_SR):int(we * config.TARGET_SR)]
                try:
                    status = process_window(seg, diar, wmodel, out_wav, out_json)
                    if status == "ok":
                        n_ok += 1; print(f"    [ok] finestra {wi}")
                    else:
                        n_skip += 1; print(f"    [skip] finestra {wi}: {status}")
                except Exception as e:
                    if "cuda" in repr(e).lower() and "out of memory" in repr(e).lower():
                        torch.cuda.empty_cache()
                    n_err += 1
                    err_file.write_text(repr(e)[:500], encoding="utf-8")
                    print(f"    [ERR] finestra {wi}: {e}")
        except Exception as e:
            n_err += 1
            print(f"    [ERR] file {path.name}: {e}")

    print(f"\n=== FATTO === ok={n_ok}  skip={n_skip}  err={n_err}")
    print("Ora costruisci il dataset con: python 03_build_dataset.py")


if __name__ == "__main__":
    main()
