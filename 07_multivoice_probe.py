"""
STEP 7 (diagnostico, una tantum) — PROBE 3-VOCI.

Problema (vedi health.md §1): 02_process.py forza la diarization a 2 speaker, quindi i
contenuti a 3+ voci NON vengono scartati: le voci extra vengono spalmate sui 2 canali e
nessuno stadio a valle se ne accorge. Inoltre gonfiano la metrica turni/min.

Questo script misura la contaminazione REALE: campiona N finestre "clean" per canale
(dal CSV dell'health check), ridecodifica il segmento ORIGINALE da data/raw (non lo
stereo mascherato) e rifà la diarization SENZA forzare 2 speaker (max_speakers=4).
Se un 3° speaker ha una quota tempo significativa, la finestra è contaminata.

Uso:
    python 07_multivoice_probe.py --csv data/health_20260717.csv --per-channel 25

Modalità alternativa --from-jsonl (per mie-ore, dove il raw non esiste più in locale):
pesca i campioni da un jsonl clean (path relativi a data/) e diarizza il DOWNMIX L+R
dello stereo mascherato (leggermente meno preciso del raw, ma i canali sono
complementari: la somma ricostruisce ~tutto il parlato originale).
    python 07_multivoice_probe.py --from-jsonl data/train_clean.jsonl --prefix mie-ore/ --per-channel 50

Output: data/multivoice_probe.csv + report a video per canale.
Solo lettura sul dataset: non modifica/elimina niente.
"""
import csv
import json
import argparse
import random
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import config

# Canali da sondare di default: VUOTO — passali con --channels "Canale-1,Canale-2".
# Consiglio: includi sempre anche un canale di CONTROLLO che sai essere 1-a-1 pulito
# (se il probe è sano, lì deve trovare ~2 voci e ~0% contaminazione).
CHANNELS: list[str] = []

# un "3° speaker" conta se ha almeno questa quota del tempo di parlato totale
THIRD_VOICE_MIN_SHARE = 0.10
# uno speaker è "presente" se ha almeno questa quota (per contare n_speakers)
PRESENT_MIN_SHARE = 0.05


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)


def build_uploader_map():
    """sanitized dir name -> vera cartella in data/raw/youtube"""
    base = config.RAW_DIR / "youtube"
    return {sanitize(d.name): d for d in base.iterdir() if d.is_dir()}


def decode_window(raw_file: Path, start: float, dur: float, sr: int = config.TARGET_SR):
    cmd = ["ffmpeg", "-v", "error", "-nostdin",
           "-ss", str(start), "-t", str(dur), "-i", str(raw_file),
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore")[:200])
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/health.csv",
                    help="CSV per-campione prodotto da 04_health_check.py")
    ap.add_argument("--per-channel", type=int, default=25,
                    help="finestre clean per canale (0 = TUTTE, per la scansione completa)")
    ap.add_argument("--channels", default="",
                    help="lista canali separati da virgola (default: i 4 a rischio + controllo)")
    ap.add_argument("--from-jsonl", default="",
                    help="pesca i campioni da questo jsonl (path relativi a data/) e "
                         "diarizza il downmix L+R dello stereo invece del raw")
    ap.add_argument("--prefix", default="",
                    help="con --from-jsonl: tieni solo i path che iniziano così")
    ap.add_argument("--out", default="data/multivoice_probe.csv")
    args = ap.parse_args()

    random.seed(0)
    if args.from_jsonl:
        pool = []
        with open(config.ROOT / args.from_jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["path"].startswith(args.prefix):
                    pool.append({"file": r["path"], "channel": args.prefix.rstrip("/") or "jsonl",
                                 "dur": str(r["duration"])})
        n = len(pool) if args.per_channel <= 0 else min(args.per_channel, len(pool))
        picked = random.sample(pool, n)
        channels = [pool[0]["channel"]] if pool else []
        print(f"Campioni da sondare (da {args.from_jsonl}, prefix '{args.prefix}'): {len(picked)}/{len(pool)}")
    else:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()] or CHANNELS
        if not channels:
            raise SystemExit('[X] Specifica i canali da sondare: --channels "Canale-1,Canale-2" '
                             "(nomi come nella colonna 'channel' del CSV health)")
        rows = list(csv.DictReader(open(config.ROOT / args.csv, encoding="utf-8")))
        by_ch = defaultdict(list)
        for r in rows:
            if r.get("clean") == "True" and r.get("channel") in channels:
                by_ch[r["channel"]].append(r)
        picked = []
        for ch in channels:
            pool = by_ch.get(ch, [])
            n = len(pool) if args.per_channel <= 0 else min(args.per_channel, len(pool))
            picked += random.sample(pool, n)
        print(f"Campioni da sondare: {len(picked)} "
              f"({', '.join(f'{ch}:{len(pool) if args.per_channel <= 0 else min(args.per_channel, len(pool))}' for ch, pool in ((c, by_ch.get(c, [])) for c in channels))})")

    up_map = build_uploader_map()

    from pyannote.audio import Pipeline
    print("Carico pyannote/speaker-diarization-3.1 ...")
    diar = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    token=config.HF_TOKEN or None)
    if torch.cuda.is_available():
        diar.to(torch.device("cuda"))
        print(f"    GPU: {torch.cuda.get_device_name(0)}")

    out_rows = []
    for i, r in enumerate(picked, 1):
        stem = Path(r["file"]).stem
        try:
            if args.from_jsonl:
                # downmix L+R dello stereo mascherato (i canali sono complementari)
                import soundfile as sf
                data, sr = sf.read(str(config.DATA_DIR / r["file"]), dtype="float32",
                                   always_2d=True)
                if sr != config.TARGET_SR:
                    raise RuntimeError(f"sr={sr}")
                audio = np.clip(data[:, 0] + data[:, 1], -1.0, 1.0)
            else:
                parts = stem.split("__")                  # youtube__<upl>__<id>_<www>
                idwin = parts[-1]
                vid, wi = idwin[:-4], int(idwin[-3:])
                upl_dir = up_map.get(parts[1])
                cands = list(upl_dir.glob(vid + ".*")) if upl_dir else []
                if not cands:
                    print(f"[{i}/{len(picked)}] {stem}: RAW NON TROVATO, salto")
                    continue
                audio = decode_window(cands[0], wi * config.WINDOW_SEC, float(r["dur"]))
            wav_t = torch.from_numpy(audio)[None, :]
            diar_out = diar({"waveform": wav_t, "sample_rate": config.TARGET_SR},
                            max_speakers=4)
            ann = getattr(diar_out, "speaker_diarization", diar_out)
            durs = defaultdict(float)
            for turn, _, spk in ann.itertracks(yield_label=True):
                durs[spk] += turn.end - turn.start
            shares = sorted(durs.values(), reverse=True)
            tot = sum(shares) or 1e-9
            shares = [d / tot for d in shares]
            n_present = sum(1 for s in shares if s >= PRESENT_MIN_SHARE)
            share3 = shares[2] if len(shares) > 2 else 0.0
            out_rows.append({
                "file": r["file"], "channel": r["channel"],
                "n_speakers": n_present, "share3": round(share3, 3),
                "shares": json.dumps([round(s, 3) for s in shares]),
                "contaminated": share3 >= THIRD_VOICE_MIN_SHARE,
            })
            print(f"[{i}/{len(picked)}] {r['channel'][:20]:20s} spk={n_present} "
                  f"share3={share3:.2f} {'<-- 3+ VOCI' if share3 >= THIRD_VOICE_MIN_SHARE else ''}")
        except Exception as e:
            print(f"[{i}/{len(picked)}] {stem}: ERR {e}")

    out_p = config.ROOT / args.out
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "channel", "n_speakers",
                                          "share3", "shares", "contaminated"])
        w.writeheader()
        w.writerows(out_rows)

    print("\n" + "=" * 64)
    print(f"  PROBE 3-VOCI — 3° speaker con quota >= {int(THIRD_VOICE_MIN_SHARE*100)}% del parlato")
    print("=" * 64)
    by = defaultdict(list)
    for r in out_rows:
        by[r["channel"]].append(r)
    for ch in channels:
        rs = by.get(ch, [])
        if not rs:
            continue
        bad = [r for r in rs if r["contaminated"]]
        med3 = float(np.median([r["share3"] for r in rs]))
        print(f"  {ch[:42]:<42} contaminati {len(bad)}/{len(rs)} "
              f"({100*len(bad)/len(rs):.0f}%)  share3 mediano {med3:.2f}")
    print(f"\n  Dettaglio per-finestra: {out_p}")


if __name__ == "__main__":
    main()
