"""
STEP 5 — Manifest "CLEAN-ONLY" per caricare su DigitalOcean solo i campioni puliti,
lasciando INTATTO tutto il dataset in locale.

Produce in data/:
  clean_files.txt     -> elenco path (relativi a data/) di TUTTI i wav+json clean
                         + i jsonl: da dare a rsync con --files-from.
  train_clean.jsonl   -> training set che referenzia SOLO i campioni clean
  eval_clean.jsonl

"Clean" = identici criteri di 04_health_check (formato 24k/2ch/PCM16, dialogo a 2 voci,
wpm sano, niente allucinazioni, italiano). Dedup per video-ID come 03_build_dataset
(per gli episodi in comune vince data/stereo, già filtrato 2-voci).

Filtri applicati:
  - MULTI-VOCE: escluse le finestre `contaminated=True` in data/multivoice_full.csv
    (probe 07: 3ª voce >=10% del parlato — la diarization a 2 speaker le aveva spalmate
    sui 2 canali). Se il csv manca, il filtro è saltato con un warning.
  - CANALI ESCLUSI: prefissi in EXCLUDE_PREFIXES (sotto) buttati in blocco — per i
    canali che decidi di eliminare interi dopo il probe 3-voci.
  - EVAL ANTI-NAN/ANTI-LEAKAGE: l'eval usa SOLO finestre stereo piene (>=590s) da
    episodi a finestra singola, con parole del main in OGNI segmento da 100s ->
    nessun campione corto/vuoto che avveleni la media eval (loss mascherata: un solo
    batch degenere = nan permanente), e ZERO leakage train/eval.

    python 05_clean_manifest.py

Poi per copiare SOLO il clean sulla droplet (da WSL/Git-Bash, dalla cartella data/):
    rsync -a --info=progress2 --files-from=clean_files.txt . root@IP:/root/imoshi-data/
"""
import re
import sys
import csv
import json
import random
import importlib.util
from pathlib import Path

import config

# eval: solo finestre piene (>= questa durata). Il training usa duration_sec=100:
# campioni più corti in eval sono i sospettati dei nan del run A2.
EVAL_MIN_DUR = 590.0

# prefissi (nome file flat) da escludere in blocco dal manifest — es. canali che hai
# deciso di buttare interi dopo il probe 3-voci (07). Formato: "youtube__Nome-Canale".
EXCLUDE_PREFIXES: list[str] = []


def _load(mod_name, fname):
    spec = importlib.util.spec_from_file_location(mod_name, config.ROOT / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bd = _load("bd", "03_build_dataset.py")   # SOURCE_DIRS, video_id (dedup)
hc = _load("hc", "04_health_check.py")    # analyze (clean = stessi KPI dell'health check)


def main():
    # finestre bocciate dal probe multi-voce (07_multivoice_probe.py --per-channel 0)
    contaminated = set()
    mv_p = config.DATA_DIR / "multivoice_full.csv"
    if mv_p.exists():
        with open(mv_p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("contaminated") == "True":
                    contaminated.add(row["file"])
        print(f"[filtro multi-voce] {len(contaminated)} finestre contaminate da escludere")
    else:
        print(f"[!] {mv_p} NON trovato: filtro multi-voce SALTATO")

    seen_ids = set()
    clean_rows = []          # {"path", "duration"} solo clean
    manifest = []            # path relativi a data/ di wav + json clean
    n_total = n_clean = 0
    n_multivoice = n_excluded = 0
    per_dir = {}
    clean_bytes = 0

    for sdir in bd.SOURCE_DIRS:
        if not sdir.exists():
            print(f"[i] salto {sdir} (non esiste)")
            continue
        ids_dir = set()
        n_dir_clean = 0
        for wav in sorted(sdir.glob("*.wav")):
            jpath = wav.with_suffix(".json")
            if not jpath.exists():
                continue
            if any(wav.stem.startswith(p) for p in EXCLUDE_PREFIXES):
                n_excluded += 1
                continue
            if wav.name in contaminated:
                n_multivoice += 1
                continue
            vid = bd.video_id(wav.stem)
            if vid and vid in seen_ids:      # dedup: episodio già preso da dir prioritaria
                continue
            if vid:
                ids_dir.add(vid)
            n_total += 1
            r = hc.analyze(wav, jpath, want_audio=False)
            if r.get("error") or "dur" not in r or not r.get("clean"):
                continue
            n_clean += 1
            n_dir_clean += 1
            rel_w = wav.relative_to(config.DATA_DIR).as_posix()
            rel_j = jpath.relative_to(config.DATA_DIR).as_posix()
            manifest.append(rel_w)
            manifest.append(rel_j)
            clean_rows.append({"path": rel_w, "duration": r["dur"]})
            try:
                clean_bytes += wav.stat().st_size
            except OSError:
                pass
        seen_ids |= ids_dir
        per_dir[sdir.name] = n_dir_clean

    if not clean_rows:
        sys.exit("[X] Nessun campione clean trovato.")

    # --- split train/eval ANTI-NAN + ANTI-LEAKAGE ---
    # eval solo da episodi con UNA sola finestra clean, STEREO, di durata piena:
    #  - SOLO stereo/ (finestre ~600s): 600/100 = chunk pieni esatti, niente code corte.
    #    (il loader di moshi-finetune spezza ogni file in chunk da duration_sec=100 e
    #    valuta solo i primi 40 batch in ordine: 50 finestre = 300 chunk, tutti usati;
    #    i mie-ore interi da ~53min producevano 1594 chunk di cui solo i primi 640 usati)
    #  - niente campioni corti (< duration_sec del training) che avvelenano l'eval (nan A2)
    #  - ogni segmento da 100s deve contenere >=1 parola SPEAKER_MAIN (un chunk senza
    #    testo del main e' il sospetto residuo per la loss mascherata)
    #  - l'episodio scelto non ha altre finestre in train -> zero leakage, zero dati persi
    EVAL_CROP_SEC = 100.0     # deve combaciare con duration_sec del training

    def ep_key(path: str) -> str:
        stem = Path(path).stem
        return bd.video_id(stem) or re.sub(r"_\d{3}$", "", stem)

    def eval_ok(r) -> bool:
        if not r["path"].startswith("stereo/") or r["duration"] < EVAL_MIN_DUR:
            return False
        try:
            al = json.loads((config.DATA_DIR / r["path"]).with_suffix(".json")
                            .read_text(encoding="utf-8"))["alignments"]
        except Exception:
            return False
        starts = sorted(float(w[1][0]) for w in al if w[2] in ("A", "SPEAKER_MAIN"))
        nseg = int(r["duration"] // EVAL_CROP_SEC) + (r["duration"] % EVAL_CROP_SEC > 0)
        for k in range(int(nseg)):
            lo, hi = k * EVAL_CROP_SEC, min((k + 1) * EVAL_CROP_SEC, r["duration"])
            if not any(lo <= s < hi for s in starts):
                return False
        return True

    by_ep = {}
    for r in clean_rows:
        by_ep.setdefault(ep_key(r["path"]), []).append(r)
    eval_pool = [rs[0] for rs in by_ep.values() if len(rs) == 1 and eval_ok(rs[0])]

    random.seed(0)
    random.shuffle(eval_pool)
    n_eval = min(50, len(clean_rows) // 20, len(eval_pool))
    eval_rows = eval_pool[:n_eval]
    eval_paths = {r["path"] for r in eval_rows}
    train_rows = [r for r in clean_rows if r["path"] not in eval_paths]
    random.shuffle(train_rows)
    print(f"[eval] pool episodi-finestra-singola pieni: {len(eval_pool)} -> "
          f"scelti {len(eval_rows)} (min dur {min((r['duration'] for r in eval_rows), default=0):.0f}s)")

    # newline="\n": forziamo LF unix (altrimenti su Windows escono CRLF e rsync/Linux
    # si ritrova un \r finale nei path -> file "non trovati").
    train_p = config.DATA_DIR / "train_clean.jsonl"
    eval_p = config.DATA_DIR / "eval_clean.jsonl"
    with open(train_p, "w", encoding="utf-8", newline="\n") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with open(eval_p, "w", encoding="utf-8", newline="\n") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")

    # i jsonl vanno copiati anch'essi sulla droplet -> li metto nel manifest
    manifest.append("train_clean.jsonl")
    manifest.append("eval_clean.jsonl")

    manifest_p = config.DATA_DIR / "clean_files.txt"
    with open(manifest_p, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(manifest) + "\n")

    tot_h = sum(r["duration"] for r in clean_rows) / 3600
    gb = clean_bytes / (1024 ** 3)
    print("=" * 56)
    print(f"  CLEAN-ONLY MANIFEST")
    print("=" * 56)
    print(f"  esaminati (post-dedup): {n_total}")
    print(f"  esclusi multi-voce:     {n_multivoice}   esclusi per prefisso: {n_excluded}")
    print(f"  CLEAN:                  {n_clean}  ({100*n_clean/n_total:.1f}%)")
    for d, n in per_dir.items():
        print(f"     {d}: {n}")
    print(f"  ore clean:   {tot_h:.1f} h")
    print(f"  dimensione:  {gb:.0f} GB  (wav clean)")
    print(f"  train/eval:  {len(train_rows)} / {len(eval_rows)}")
    print("=" * 56)
    print(f"  -> {manifest_p}  ({len(manifest)} righe: wav+json+jsonl)")
    print(f"  -> {train_p}")
    print(f"  -> {eval_p}")
    print(f"\n  Upload selettivo (da WSL/Git-Bash, DENTRO la cartella data/):")
    print(f"     rsync -a --info=progress2 --files-from=clean_files.txt . root@IP:/root/imoshi-data/")
    print(f"  Sulla droplet il training userà train_clean.jsonl (path relativi già corretti).")


if __name__ == "__main__":
    main()
