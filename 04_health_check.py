"""
STEP 4 (opzionale ma consigliato) — Health check del dataset per Moshi.

Calcola KPI di qualità su tutte le coppie wav+json in data/stereo/ e dice quanto
è "pulito" il dataset e quante ore sono effettivamente buone per il training.

Riutilizzabile: lancialo di nuovo dopo aver aggiunto/processato altre ore.

    python 04_health_check.py                 # tutto data/stereo
    python 04_health_check.py --audio-sample 200   # quanti file per il check sui livelli audio
    python 04_health_check.py --csv out.csv   # dove scrivere il report per-campione

KPI calcolati per ogni campione:
  - formato: sample rate 24k / 2 canali / PCM16 (deve essere 100%)
  - bilanciamento speaker A/B (Moshi = dialogo a 2 voci: serve un 2° speaker presente)
  - densità di parlato (copertura) e sovrapposizione tra i due canali
  - parole/minuto (troppo basse = musica/silenzio; troppo alte = allucinazioni whisper)
  - ripetizione testo (allucinazioni tipo "Sottotitoli a cura di...")
  - italianità (euristica su stopword: scova segmenti in inglese/altra lingua)
  - livelli audio RMS per canale (su un campione): scova canali muti/clipping

Output: report a video + CSV per-campione + verdetto finale (ore "clean").
"""
import sys
import csv
import json
import argparse
import random
from collections import Counter

import numpy as np
import soundfile as sf

import config

# stopword italiane comuni: euristica leggera per stimare se il testo è italiano
IT_STOP = {
    "di","che","e","la","il","un","a","per","in","non","è","una","mi","con","si","ma",
    "ti","ci","lo","ho","le","i","da","se","sono","ha","come","più","cosa","io","tu",
    "no","sì","anche","perché","quando","della","del","gli","questo","questa","ne","me",
    "te","va","fa","poi","tutto","molto","ero","era","cioè","quindi","là","qua","noi","voi",
}

# soglie (tarate per dialogo conversazionale italiano in finestre ~10 min)
WPM_MIN, WPM_MAX = 40, 320          # parole/min plausibili (somma 2 canali)
MIN_SECONDARY_SHARE = 0.15          # il 2° speaker deve avere >=15% delle parole
MAX_BIGRAM_FREQ = 0.05              # freq del bigramma più ripetuto: sopra = allucinazione
MIN_IT_RATIO = 0.08                 # frazione di stopword IT: sotto = forse non italiano
# NB: uniq_ratio (unique/total) è ancora riportato ma NON usato come gate: cala sui file
# lunghi per saturazione del vocabolario, non per allucinazione -> dava falsi positivi.


def merge_intervals(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total_len(iv):
    return sum(e - s for s, e in iv)


def intersect_len(a, b):
    """durata totale in cui A e B sono entrambi attivi (a,b = liste mergiate)."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyze(wav, jpath, want_audio):
    try:
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate
    except Exception as e:
        return {"file": wav.name, "ok_format": False, "error": f"wav:{e}"}

    fmt_ok = (info.samplerate == config.TARGET_SR and info.channels == 2
              and "PCM_16" in (info.subtype or ""))

    try:
        al = json.loads(jpath.read_text(encoding="utf-8")).get("alignments", [])
    except Exception as e:
        return {"file": wav.name, "ok_format": fmt_ok, "error": f"json:{e}"}

    A = [w for w in al if w[2] in ("A", "SPEAKER_MAIN")]  # canale principale (sx)
    B = [w for w in al if w[2] == "B"]
    nA, nB = len(A), len(B)
    ntot = nA + nB
    words = [str(w[0]).lower().strip(".,!?;:\"'()[]") for w in al if w[0]]

    ivA = merge_intervals([(float(w[1][0]), float(w[1][1])) for w in A if w[1][1] > w[1][0]])
    ivB = merge_intervals([(float(w[1][0]), float(w[1][1])) for w in B if w[1][1] > w[1][0]])
    speech = total_len(ivA) + total_len(ivB)
    overlap = intersect_len(ivA, ivB)

    secondary_share = (min(nA, nB) / ntot) if ntot else 0.0
    wpm = (ntot / dur * 60) if dur else 0.0
    coverage = (speech / dur) if dur else 0.0
    overlap_pct = (overlap / dur) if dur else 0.0
    uniq_ratio = (len(set(words)) / len(words)) if words else 0.0
    bigrams = list(zip(words, words[1:]))
    top_bigram = (Counter(bigrams).most_common(1)[0][1] / len(bigrams)) if bigrams else 0.0
    it_ratio = (sum(1 for w in words if w in IT_STOP) / len(words)) if words else 0.0

    # --- turn-taking (botta e risposta): conta le ALTERNANZE tra i 2 speaker ---
    # NB: solo informativo, NON entra nel gate "clean" (le soglie si decidono dopo
    # aver visto la distribuzione reale sul dataset).
    seq = sorted(al, key=lambda w: w[1][0])
    turns = []  # [label, start, end] di ogni turno (parole consecutive stesso speaker)
    for w in seq:
        lab = "A" if w[2] in ("A", "SPEAKER_MAIN") else "B"
        if turns and turns[-1][0] == lab:
            turns[-1][2] = max(turns[-1][2], float(w[1][1]))
        else:
            turns.append([lab, float(w[1][0]), float(w[1][1])])
    turns_min = (len(turns) / dur * 60) if dur else 0.0
    turn_durs = [t[2] - t[1] for t in turns]
    med_turn = float(np.median(turn_durs)) if turn_durs else 0.0

    # canale/sorgente dal nome file flat: "youtube__<Canale>__<id>_<win>"
    parts = wav.stem.split("__")
    channel = parts[1] if len(parts) >= 3 else ""

    rms_L = rms_R = None
    if want_audio:
        try:
            n = min(info.frames, config.TARGET_SR * 60)  # primi 60s
            data, _ = sf.read(str(wav), frames=n, dtype="float32", always_2d=True)
            rms_L = float(np.sqrt(np.mean(data[:, 0] ** 2)))
            rms_R = float(np.sqrt(np.mean(data[:, 1] ** 2)))
        except Exception:
            pass

    # un campione è "clean" se passa tutti i filtri base
    clean = (fmt_ok and ntot > 0 and WPM_MIN <= wpm <= WPM_MAX
             and secondary_share >= MIN_SECONDARY_SHARE
             and top_bigram <= MAX_BIGRAM_FREQ and it_ratio >= MIN_IT_RATIO)

    return {
        "file": wav.name, "channel": channel, "dur": dur, "ok_format": fmt_ok,
        "nA": nA, "nB": nB, "secondary_share": secondary_share,
        "wpm": wpm, "coverage": coverage, "overlap_pct": overlap_pct,
        "turns_min": turns_min, "med_turn": med_turn,
        "uniq_ratio": uniq_ratio, "top_bigram": top_bigram, "it_ratio": it_ratio,
        "rms_L": rms_L, "rms_R": rms_R, "clean": clean, "error": "",
    }


def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "n/a"


def dist(vals):
    if not vals:
        return "n/a"
    a = np.array(vals)
    return (f"min={a.min():.2f} p25={np.percentile(a,25):.2f} "
            f"mediana={np.median(a):.2f} p75={np.percentile(a,75):.2f} max={a.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-sample", type=int, default=200,
                    help="quanti file leggere per il check livelli audio RMS (0 = nessuno)")
    ap.add_argument("--dir", default="",
                    help="cartella da analizzare (default: data/stereo). Usa per le tue ore.")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    from pathlib import Path
    scan_dir = Path(args.dir) if args.dir else config.STEREO_DIR
    csv_path = args.csv or str(scan_dir.parent / f"health_{scan_dir.name}.csv")

    wavs = sorted(scan_dir.glob("*.wav"))
    if not wavs:
        sys.exit(f"[X] Nessun wav in {scan_dir}")

    audio_idx = set(range(len(wavs)))
    if args.audio_sample and len(wavs) > args.audio_sample:
        random.seed(0)
        audio_idx = set(random.sample(range(len(wavs)), args.audio_sample))

    rows = []
    print(f"Analizzo {len(wavs)} campioni...")
    for i, wav in enumerate(wavs):
        jpath = wav.with_suffix(".json")
        if not jpath.exists():
            rows.append({"file": wav.name, "ok_format": False, "error": "json mancante", "clean": False})
            continue
        rows.append(analyze(wav, jpath, want_audio=(i in audio_idx)))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(wavs)}")

    valid = [r for r in rows if not r.get("error") and "dur" in r]
    n = len(rows)
    tot_h = sum(r["dur"] for r in valid) / 3600
    clean = [r for r in valid if r.get("clean")]
    clean_h = sum(r["dur"] for r in clean) / 3600

    def flag(cond):
        return [r for r in valid if cond(r)]

    bad_fmt = flag(lambda r: not r["ok_format"])
    monolog = flag(lambda r: r["secondary_share"] < MIN_SECONDARY_SHARE)
    low_wpm = flag(lambda r: r["wpm"] < WPM_MIN)
    high_wpm = flag(lambda r: r["wpm"] > WPM_MAX)
    repet = flag(lambda r: r["top_bigram"] > MAX_BIGRAM_FREQ)
    noit = flag(lambda r: r["it_ratio"] < MIN_IT_RATIO)
    audio_rows = [r for r in valid if r.get("rms_L") is not None]
    mute_ch = [r for r in audio_rows if min(r["rms_L"], r["rms_R"]) < 1e-4]

    print("\n" + "=" * 60)
    print(f"  HEALTH CHECK DATASET — {scan_dir}")
    print("=" * 60)
    print(f"\n[VOLUME]")
    print(f"  campioni: {n}   ore totali: {tot_h:.1f} h")
    print(f"  durata campione (s): {dist([r['dur'] for r in valid])}")
    print(f"\n[FORMATO]  (deve essere 24000Hz / 2ch / PCM16)")
    print(f"  conformi: {pct(len(valid)-len(bad_fmt), len(valid))}   non conformi: {len(bad_fmt)}")
    print(f"\n[DIALOGO A/B]  (2° speaker deve avere >= {int(MIN_SECONDARY_SHARE*100)}% parole)")
    print(f"  quota 2° speaker: {dist([r['secondary_share'] for r in valid])}")
    print(f"  quasi-monologo: {len(monolog)} campioni ({pct(len(monolog),len(valid))})")
    print(f"\n[DENSITÀ PARLATO / SOVRAPPOSIZIONE]")
    print(f"  copertura parlato: {dist([r['coverage'] for r in valid])}")
    print(f"  sovrapposizione A/B: {dist([r['overlap_pct'] for r in valid])}")
    print(f"\n[TURN-TAKING]  (informativo: alternanze speaker, il 'botta e risposta')")
    print(f"  turni/min: {dist([r.get('turns_min', 0) for r in valid])}")
    print(f"  durata mediana turno (s): {dist([r.get('med_turn', 0) for r in valid])}")
    print(f"\n[TRASCRIZIONE]")
    print(f"  parole/min: {dist([r['wpm'] for r in valid])}")
    print(f"  troppo poche parole (<{WPM_MIN} wpm, musica/silenzio): {len(low_wpm)} ({pct(len(low_wpm),len(valid))})")
    print(f"  troppe parole (>{WPM_MAX} wpm, possibili allucinazioni): {len(high_wpm)} ({pct(len(high_wpm),len(valid))})")
    print(f"\n[QUALITÀ TESTO]")
    print(f"  allucinazioni (bigramma ripetuto >{MAX_BIGRAM_FREQ}): {len(repet)} ({pct(len(repet),len(valid))})")
    print(f"  (uniq_ratio mediano, solo informativo: {np.median([r['uniq_ratio'] for r in valid]):.2f})")
    print(f"  forse non-italiano (stopword<{MIN_IT_RATIO}): {len(noit)} ({pct(len(noit),len(valid))})")
    if audio_rows:
        print(f"\n[AUDIO RMS]  (campione di {len(audio_rows)} file)")
        print(f"  canali quasi muti: {len(mute_ch)} ({pct(len(mute_ch),len(audio_rows))})")

    # riepilogo per canale/sorgente (il nome file flat contiene il canale)
    by_ch = {}
    for r in valid:
        by_ch.setdefault(r.get("channel") or "?", []).append(r)
    if len(by_ch) > 1:
        print(f"\n[PER CANALE]  (ordinati per ore totali)")
        print(f"  {'canale':<42} {'camp.':>6} {'ore':>8} {'clean':>14} {'turni/min':>10}")
        for ch, rs in sorted(by_ch.items(), key=lambda x: -sum(r['dur'] for r in x[1])):
            ch_clean = [r for r in rs if r.get("clean")]
            h = sum(r["dur"] for r in rs) / 3600
            h_cl = sum(r["dur"] for r in ch_clean) / 3600
            tm = np.median([r.get("turns_min", 0) for r in rs])
            print(f"  {ch[:42]:<42} {len(rs):>6} {h:>7.1f}h {h_cl:>6.1f}h ({pct(len(ch_clean),len(rs)):>6}) {tm:>9.1f}")
    print("\n" + "=" * 60)
    print(f"  VERDETTO: ore CLEAN = {clean_h:.1f} h su {tot_h:.1f} h  ({pct(len(clean),len(valid))} dei campioni)")
    print("=" * 60)
    health = 100 * clean_h / tot_h if tot_h else 0
    verdict = ("OTTIMO" if health >= 85 else "BUONO" if health >= 70
               else "DA RIPULIRE" if health >= 50 else "PROBLEMATICO")
    print(f"  Health score: {health:.0f}/100  -> {verdict}")
    print(f"  (per Moshi servono dialoghi a 2 voci puliti: punta a >=70% clean)")

    # CSV per-campione + lista peggiori
    fields = ["file","channel","dur","ok_format","nA","nB","secondary_share","wpm",
              "coverage","overlap_pct","turns_min","med_turn",
              "uniq_ratio","top_bigram","it_ratio","rms_L","rms_R","clean","error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n  Report per-campione: {csv_path}")
    print(f"  (filtra la colonna 'clean'=False per vedere/eliminare i campioni scartabili)")


if __name__ == "__main__":
    main()
