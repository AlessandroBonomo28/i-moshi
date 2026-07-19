"""
STEP 9 — PREFLIGHT: gate finale prima di upload su Spaces e training.

Differenza rispetto a 04_health_check: 04 analizza la CARTELLA (qualita' dei campioni),
questo valida i MANIFEST REALI (train_clean.jsonl / eval_clean.jsonl / clean_files.txt)
contro il contratto verificato nel sorgente di moshi-finetune (17/07/2026):
  - il loader spezza ogni entry in chunk da duration_sec (100s) e padda l'ultimo con
    zero_padding (interleaver.py) -> campioni corti in TRAIN sono gestiti;
  - la loss e' somma(loss*pesi)/somma(pesi) (loss.py): UN chunk con maschera vuota
    fa nan, e l'eval ACCUMULA su tutti i batch (eval.py) -> un solo chunk cattivo
    avvelena l'intero eval, in modo deterministico (nessuno shuffle in eval);
  - l'eval valuta SOLO i primi 40 batch in ordine di file (eval.py:37);
  - il testo tiene SOLO le parole "SPEAKER_MAIN" (interleaver keep_main_only);
  - i path dei jsonl sono relativi alla posizione del jsonl (data/).

Uso:
    python 09_preflight.py            # gate standard (campiona il train)
    python 09_preflight.py --full     # scansione etichette su TUTTO il train (lenta)

Exit code 0 = PASS (si puo' caricare/trainare), 1 = FAIL con la lista dei problemi.
Read-only: non modifica niente.
"""
import re
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import soundfile as sf

import config

EVAL_MIN_DUR = 590.0
CROP_SEC = 100.0          # duration_sec del training: se cambi il config, cambia qui
LABEL_SAMPLE = 400        # json del train campionati per il check etichette (senza --full)
WAV_SPOT = 30             # wav campionati per verifica header/durata vs manifest


def ep_key(path: str) -> str:
    return re.sub(r"_\d{3}$", "", Path(path).stem)


def load_jsonl(p: Path):
    rows, problems = [], []
    raw = p.read_bytes()
    if b"\r\n" in raw:
        problems.append(f"{p.name}: contiene CRLF (deve essere LF per rsync/linux)")
    for i, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            assert isinstance(r["path"], str) and float(r["duration"]) > 0
            rows.append(r)
        except Exception as e:
            problems.append(f"{p.name}:{i}: riga malformata ({e})")
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="check etichette su TUTTI i json del train (default: campione)")
    args = ap.parse_args()

    fails: list[str] = []
    warns: list[str] = []

    def fail(msg):
        fails.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(msg):
        warns.append(msg)
        print(f"  [warn] {msg}")

    def ok(msg):
        print(f"  [ok]   {msg}")

    print("=" * 64)
    print("  PREFLIGHT — gate finale manifest -> upload/training")
    print("=" * 64)

    # 1) jsonl ben formati, niente duplicati, path esistenti in coppia
    print("\n[1] Manifest ben formati e file presenti")
    train, p1 = load_jsonl(config.DATA_DIR / "train_clean.jsonl")
    evalr, p2 = load_jsonl(config.DATA_DIR / "eval_clean.jsonl")
    for m in p1 + p2:
        fail(m)
    for name, rows in (("train", train), ("eval", evalr)):
        dups = len(rows) - len({r["path"] for r in rows})
        if dups:
            fail(f"{name}: {dups} path duplicati")
    missing = 0
    for r in train + evalr:
        wav = config.DATA_DIR / r["path"]
        if not wav.exists() or not wav.with_suffix(".json").exists():
            missing += 1
    if missing:
        fail(f"{missing} coppie wav+json del manifest mancanti su disco")
    else:
        ok(f"train {len(train)} + eval {len(evalr)} campioni, tutte le coppie esistono")

    # clean_files.txt copre tutto
    listed = set((config.DATA_DIR / "clean_files.txt").read_text(encoding="utf-8").split())
    not_listed = [r["path"] for r in train + evalr if r["path"] not in listed]
    if not_listed:
        fail(f"{len(not_listed)} path dei jsonl NON in clean_files.txt (upload li perderebbe)")
    else:
        ok("ogni path dei jsonl e' in clean_files.txt")

    # 2) contratto etichette: il testo del training = solo SPEAKER_MAIN
    print(f"\n[2] Etichette (SPEAKER_MAIN obbligatorio) — eval TUTTO, train "
          f"{'TUTTO' if args.full else f'campione {LABEL_SAMPLE}'}")
    random.seed(0)
    to_check = [r["path"] for r in evalr] + (
        [r["path"] for r in train] if args.full
        else [r["path"] for r in random.sample(train, min(LABEL_SAMPLE, len(train)))])
    bad_lab = []
    for p in to_check:
        try:
            al = json.loads((config.DATA_DIR / p).with_suffix(".json")
                            .read_text(encoding="utf-8"))["alignments"]
        except Exception as e:
            bad_lab.append(f"{p}: json illeggibile ({e})")
            continue
        labs = {a[2] for a in al}
        if "SPEAKER_MAIN" not in labs:
            bad_lab.append(f"{p}: labels={labs} (testo VUOTO in training!)")
    if bad_lab:
        for m in bad_lab[:10]:
            fail(m)
        if len(bad_lab) > 10:
            fail(f"... e altri {len(bad_lab)-10}")
    else:
        ok(f"{len(to_check)} json verificati: SPEAKER_MAIN presente ovunque")

    # 3) regole EVAL (anti-nan, anti-leakage, eval veloce e rappresentativo)
    print("\n[3] Regole eval")
    non_stereo = [r for r in evalr if not r["path"].startswith("stereo/")]
    if non_stereo:
        fail(f"eval: {len(non_stereo)} campioni non-stereo (chunk di coda + eval lento)")
    else:
        ok("eval solo finestre stereo")
    short = [r for r in evalr if r["duration"] < EVAL_MIN_DUR]
    if short:
        fail(f"eval: {len(short)} campioni < {EVAL_MIN_DUR}s")
    else:
        ok(f"tutte le durate eval >= {EVAL_MIN_DUR}s (min "
           f"{min(r['duration'] for r in evalr):.0f}s)")
    train_eps = {ep_key(r["path"]) for r in train}
    leak = [r["path"] for r in evalr if ep_key(r["path"]) in train_eps]
    if leak:
        fail(f"LEAKAGE: {len(leak)} episodi eval presenti anche in train")
    else:
        ok("zero leakage train/eval (per episodio)")
    # ogni segmento da 100s di ogni eval deve avere >=1 parola SPEAKER_MAIN
    empty_crop = 0
    for r in evalr:
        al = json.loads((config.DATA_DIR / r["path"]).with_suffix(".json")
                        .read_text(encoding="utf-8"))["alignments"]
        starts = sorted(float(w[1][0]) for w in al if w[2] in ("A", "SPEAKER_MAIN"))
        nseg = int(r["duration"] // CROP_SEC) + (r["duration"] % CROP_SEC > 0)
        for k in range(int(nseg)):
            lo, hi = k * CROP_SEC, min((k + 1) * CROP_SEC, r["duration"])
            if not any(lo <= s < hi for s in starts):
                empty_crop += 1
    if empty_crop:
        fail(f"eval: {empty_crop} segmenti da {CROP_SEC:.0f}s SENZA parole del main")
    else:
        ok(f"nessun segmento da {CROP_SEC:.0f}s senza parole del main in eval")
    n_chunks = sum(int(r["duration"] // CROP_SEC) + (r["duration"] % CROP_SEC > 0)
                   for r in evalr)
    if n_chunks > 640:
        warn(f"eval = {n_chunks} chunk ma il training ne valuta solo i primi 640 "
             f"(40 batch): parte dell'eval non verrebbe mai usata")
    else:
        ok(f"eval = {n_chunks} chunk (<= 640: tutti valutati dal training)")

    # 4) spot-check wav: header 24k/2ch/PCM16 e durata coerente col manifest
    print(f"\n[4] Spot-check {WAV_SPOT} wav (header + durata vs manifest)")
    bad_wav = 0
    for r in random.sample(train + evalr, min(WAV_SPOT, len(train) + len(evalr))):
        try:
            info = sf.info(str(config.DATA_DIR / r["path"]))
            real = info.frames / info.samplerate
            if info.samplerate != config.TARGET_SR or info.channels != 2 \
               or "PCM_16" not in (info.subtype or "") or abs(real - r["duration"]) > 0.1:
                bad_wav += 1
                fail(f"{r['path']}: {info.samplerate}Hz/{info.channels}ch/"
                     f"{info.subtype} dur reale {real:.1f}s vs manifest {r['duration']:.1f}s")
        except Exception as e:
            bad_wav += 1
            fail(f"{r['path']}: illeggibile ({e})")
    if not bad_wav:
        ok("formato e durate coerenti")

    # 5) riepilogo volumi (informativo)
    print("\n[5] Riepilogo (informativo)")
    tot_h = sum(r["duration"] for r in train + evalr) / 3600
    gb = sum((config.DATA_DIR / r["path"]).stat().st_size
             for r in train + evalr) / 1024**3
    n_short = sum(1 for r in train if r["duration"] < CROP_SEC)
    print(f"  ore totali: {tot_h:.1f}h   wav: {gb:.0f}GB   "
          f"train<{CROP_SEC:.0f}s: {n_short} (ok: il loader li padda)")
    v1p = config.DATA_DIR / "v1_stereo_files.txt"
    v1 = {l.strip().replace(".json", ".wav") for l in
          v1p.read_text(encoding="utf-8-sig").splitlines() if l.strip()} if v1p.exists() else set()
    by_ch = defaultdict(float)
    for r in train + evalr:
        stem = Path(r["path"]).stem
        ch = stem.split("__")[1] if "__" in stem else "mie-ore"
        by_ch[ch] += r["duration"] / 3600
    for ch, h in sorted(by_ch.items(), key=lambda x: -x[1]):
        print(f"    {ch[:44]:<44} {h:7.1f}h")
    if v1:
        nv2 = sum(1 for r in train + evalr
                  if r["path"].startswith("stereo/") and Path(r["path"]).name not in v1)
        print(f"  delta v2 (stereo non in v1_stereo_files.txt): {nv2} finestre")

    print("\n" + "=" * 64)
    if fails:
        print(f"  PREFLIGHT: FAIL - {len(fails)} problemi, {len(warns)} warning")
        sys.exit(1)
    print(f"  PREFLIGHT: PASS - {len(warns)} warning. Pronto per upload/training.")


if __name__ == "__main__":
    main()
