"""
STEP 3 — Costruzione del dataset finale per moshi-finetune.

Scansiona data/stereo/ per le coppie .wav + .json valide e produce:
    data/train.jsonl
    data/eval.jsonl        (piccolo holdout, opzionale)

Ogni riga: {"path": "stereo/xxx.wav", "duration": <secondi>}
I path sono RELATIVI a data/ (= la cartella del jsonl), come si aspetta moshi-finetune.
Quindi su DigitalOcean basta caricare l'intera cartella data/ mantenendo la struttura.

Stampa anche le STATISTICHE: ore totali, n. file, durata media — così verifichi
di aver raggiunto le ~1000h prima di pagare la GPU.

Uso:
    python 03_build_dataset.py            # holdout eval di default = 50 file
    python 03_build_dataset.py --eval 0   # nessun eval
"""
import sys
import re
import json
import argparse

import soundfile as sf

import config

# Cartelle sorgente, IN ORDINE DI PRIORITÀ per la deduplica:
# i miei (data/stereo) vincono sugli stessi episodi nelle tue ore (filtrati 2-voci).
SOURCE_DIRS = [
    config.STEREO_DIR,
    config.DATA_DIR / "mie-ore" / "mydataset",
]


def video_id(name: str):
    """Estrae il video-ID YouTube (11 char) dai due formati di nome file."""
    m = re.search(r"__([A-Za-z0-9_-]{11})_\d{3}$", name)          # miei
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z0-9_-]{11})-tagliato_24k$", name)     # tuoi
    if m:
        return m.group(1)
    return None


def is_valid_json(jpath):
    """Vero solo se ci sono allineamenti e ENTRAMBI gli speaker hanno parlato."""
    try:
        data = json.loads(jpath.read_text(encoding="utf-8"))
    except Exception:
        return False
    aligns = data.get("alignments", [])
    if not aligns:
        return False
    speakers = {a[2] for a in aligns}
    # il canale principale può chiamarsi "SPEAKER_MAIN" (formato corretto per moshi-finetune)
    # o "A" (formato storico, pre-fix 06)
    has_main = "SPEAKER_MAIN" in speakers or "A" in speakers
    return has_main and "B" in speakers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=int, default=50, help="quanti file tenere per eval")
    args = ap.parse_args()

    rows = []
    skipped = 0
    dup_skipped = 0
    seen_ids = set()
    per_dir = {}
    for sdir in SOURCE_DIRS:
        if not sdir.exists():
            print(f"[i] salto {sdir} (non esiste)")
            continue
        n_dir = 0
        ids_this_dir = set()
        for wav in sorted(sdir.glob("*.wav")):
            jpath = wav.with_suffix(".json")
            if not jpath.exists() or not is_valid_json(jpath):
                skipped += 1
                continue
            # dedup per video-ID: se l'episodio è già stato preso da una cartella
            # a priorità più alta, lo saltiamo qui.
            vid = video_id(wav.stem)
            if vid and vid in seen_ids:
                dup_skipped += 1
                continue
            try:
                info = sf.info(str(wav))
                dur = info.frames / info.samplerate
            except Exception:
                skipped += 1
                continue
            if dur < config.MIN_DURATION_SEC:
                skipped += 1
                continue
            if vid:
                ids_this_dir.add(vid)
            rel = wav.relative_to(config.DATA_DIR).as_posix()  # path relativo a data/
            rows.append({"path": rel, "duration": dur})
            n_dir += 1
        seen_ids |= ids_this_dir
        per_dir[sdir.name] = n_dir

    if not rows:
        sys.exit("[X] Nessun campione valido. Hai lanciato 02_process.py?")
    print(f"[dedup] episodi duplicati scartati (già presi da cartella prioritaria): {dup_skipped}")
    for d, n in per_dir.items():
        print(f"   {d}: {n} campioni")

    # mescolo (seed fisso) così l'eval è un mix rappresentativo di tutte le sorgenti
    import random
    random.seed(0)
    random.shuffle(rows)

    # split deterministico train/eval
    n_eval = min(args.eval, max(0, len(rows) // 20))  # mai più del 5%
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]

    train_path = config.DATA_DIR / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    if eval_rows:
        eval_path = config.DATA_DIR / "eval.jsonl"
        with open(eval_path, "w", encoding="utf-8") as f:
            for r in eval_rows:
                f.write(json.dumps(r) + "\n")

    # statistiche
    tot_sec = sum(r["duration"] for r in rows)
    print("=" * 50)
    print(f"  Campioni validi : {len(rows)}   (scartati: {skipped})")
    print(f"  Ore TOTALI      : {tot_sec/3600:.1f} h")
    print(f"  Durata media    : {tot_sec/len(rows):.1f} s/campione")
    print(f"  Train / Eval    : {len(train_rows)} / {len(eval_rows)}")
    print("=" * 50)
    print(f"  -> {train_path}")
    if eval_rows:
        print(f"  -> {config.DATA_DIR / 'eval.jsonl'}")
    print("\nObiettivo ~1000h. Carica l'intera cartella data/ su DigitalOcean.")
    if tot_sec / 3600 < 1000:
        print(f"[i] Mancano ~{1000 - tot_sec/3600:.0f} h: aggiungi sorgenti e rilancia 01/02.")


if __name__ == "__main__":
    main()
