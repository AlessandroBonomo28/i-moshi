"""
STEP 8 (diagnostico, una tantum) — PROBE "FLIP DEL MAIN".

Problema (health.md §3, diary §14): SPEAKER_MAIN è scelto PER-FINESTRA (chi parla di più
in quei 10 min), quindi nello stesso episodio la voce sul canale sinistro — quella che
Moshi impara come "propria" — può cambiare persona tra una finestra e l'altra.

Misura: per un campione di episodi multi-finestra del manifest clean, estrae l'impronta
vocale (speaker embedding wespeaker, lo stesso della diarization) dal parlato di OGNI
canale di OGNI finestra. Tra finestre consecutive (k -> k+1) dello stesso episodio:

    flip se  cos(main_{k+1}, interloc_k)  >  cos(main_{k+1}, main_k)

cioè: il main di adesso somiglia più all'interlocutore di prima che al main di prima.
Niente soglie assolute: è un confronto relativo, robusto.

Uso:
    python 08_main_flip_probe.py --episodes 120
Output: data/main_flip_probe.csv (una riga per transizione) + report per canale.
Solo lettura: non modifica niente.
"""
import re
import csv
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import soundfile as sf

import config

MAX_EMBED_SEC = 30.0     # quanto parlato (concatenato) basta per un'impronta affidabile
MIN_EMBED_SEC = 3.0      # sotto questa soglia la finestra si salta
MAX_WIN_PER_EP = 8       # tetto finestre per episodio (bounda il costo)


def ep_key(stem: str) -> str:
    return re.sub(r"_\d{3}$", "", stem)


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


def speech_concat(chan: np.ndarray, words, sr: int) -> np.ndarray:
    """concatena il parlato del canale (dagli intervalli parola) fino a MAX_EMBED_SEC"""
    iv = merge_intervals([(float(w[1][0]), float(w[1][1])) for w in words
                          if w[1][1] > w[1][0]])
    parts, tot = [], 0.0
    for s, e in iv:
        a, b = int(s * sr), min(int(e * sr), len(chan))
        if b <= a:
            continue
        parts.append(chan[a:b])
        tot += (b - a) / sr
        if tot >= MAX_EMBED_SEC:
            break
    if tot < MIN_EMBED_SEC:
        return None
    return np.concatenate(parts)


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--out", default="data/main_flip_probe.csv")
    args = ap.parse_args()

    # episodi multi-finestra dal manifest clean (solo stereo/, mie-ore non è a finestre)
    wins = []
    for jl in ("train_clean.jsonl", "eval_clean.jsonl"):
        with open(config.DATA_DIR / jl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["path"].startswith("stereo/"):
                    wins.append(r["path"])
    by_ep = defaultdict(list)
    for p in wins:
        by_ep[ep_key(Path(p).stem)].append(p)
    multi = {k: sorted(v) for k, v in by_ep.items() if len(v) >= 2}
    print(f"Episodi multi-finestra nel clean: {len(multi)} (su {len(by_ep)} totali)")

    random.seed(0)
    picked = random.sample(sorted(multi), min(args.episodes, len(multi)))

    from pyannote.audio import Model, Inference
    print("Carico wespeaker-voxceleb-resnet34-LM ...")
    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM",
                                  token=config.HF_TOKEN or None)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inf = Inference(model, window="whole", device=torch.device(dev))
    print(f"    device: {dev}")

    def embed(chan_audio):
        wav_t = torch.from_numpy(chan_audio)[None, :]
        return np.asarray(inf({"waveform": wav_t, "sample_rate": config.TARGET_SR}))

    out_rows = []
    n_trans = n_flip = 0
    for ei, ep in enumerate(picked, 1):
        prev = None      # (main_emb, interloc_emb, win_name)
        ep_flips = 0
        for p in multi[ep][:MAX_WIN_PER_EP]:
            wavp = config.DATA_DIR / p
            jp = wavp.with_suffix(".json")
            try:
                data, sr = sf.read(str(wavp), dtype="float32", always_2d=True)
                al = json.loads(jp.read_text(encoding="utf-8"))["alignments"]
                mainw = [w for w in al if w[2] in ("A", "SPEAKER_MAIN")]
                intw = [w for w in al if w[2] == "B"]
                a_main = speech_concat(data[:, 0], mainw, sr)
                a_int = speech_concat(data[:, 1], intw, sr)
                if a_main is None or a_int is None:
                    prev = None
                    continue
                e_main, e_int = embed(a_main), embed(a_int)
            except Exception as e:
                print(f"  [!] {p}: {e}")
                prev = None
                continue
            if prev is not None:
                sim_same = cos(e_main, prev[0])
                sim_cross = cos(e_main, prev[1])
                flip = sim_cross > sim_same
                n_trans += 1
                n_flip += int(flip)
                ep_flips += int(flip)
                ch = Path(p).stem.split("__")[1] if "__" in Path(p).stem else "?"
                out_rows.append({"episode": ep, "channel": ch, "win": Path(p).name,
                                 "sim_same": round(sim_same, 3),
                                 "sim_cross": round(sim_cross, 3), "flip": flip})
            prev = (e_main, e_int, p)
        print(f"[{ei}/{len(picked)}] {ep[:60]:60s} flips {ep_flips}")

    out_p = config.ROOT / args.out
    with open(out_p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "channel", "win",
                                          "sim_same", "sim_cross", "flip"])
        w.writeheader()
        w.writerows(out_rows)

    print("\n" + "=" * 64)
    print("  PROBE FLIP DEL MAIN")
    print("=" * 64)
    print(f"  transizioni analizzate: {n_trans}")
    print(f"  FLIP: {n_flip} ({100*n_flip/max(n_trans,1):.1f}% delle transizioni)")
    eps_with = len({r['episode'] for r in out_rows if r['flip']})
    eps_tot = len({r['episode'] for r in out_rows})
    print(f"  episodi con almeno 1 flip: {eps_with}/{eps_tot} "
          f"({100*eps_with/max(eps_tot,1):.0f}%)")
    by_ch = defaultdict(list)
    for r in out_rows:
        by_ch[r["channel"]].append(r)
    print(f"\n  per canale:")
    for ch, rs in sorted(by_ch.items(), key=lambda x: -len(x[1])):
        fl = sum(1 for r in rs if r["flip"])
        print(f"    {ch[:42]:<42} flip {fl}/{len(rs)} ({100*fl/len(rs):.0f}%)")
    same = [r["sim_same"] for r in out_rows]
    cross = [r["sim_cross"] for r in out_rows]
    print(f"\n  sanity: sim_same mediana {np.median(same):.2f} / "
          f"sim_cross mediana {np.median(cross):.2f} "
          f"(se fossero vicine, l'embedding non distingue le voci)")
    print(f"  Dettaglio: {out_p}")


if __name__ == "__main__":
    main()
