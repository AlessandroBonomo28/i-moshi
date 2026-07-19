"""
STEP 1 — Download dei sorgenti audio (YouTube + RSS) nella cartella data/raw.

Uso:
    python 01_download.py youtube      # scarica tutte le URL in sources/youtube_sources.txt
    python 01_download.py rss          # scarica tutti i feed in sources/podcast_rss.txt
    python 01_download.py all          # entrambi

Caratteristiche:
    - Ripartibile: yt-dlp tiene un archivio (data/raw/_yt_archive.txt); l'RSS salta i file
      già scaricati. Puoi interrompere con Ctrl+C e rilanciare senza riscaricare.
    - Scarica solo l'audio (opus/m4a) per risparmiare spazio: la decodifica vera la fa
      lo step 2 con ffmpeg, che legge qualsiasi formato.

NB: le tue 300h già pronte (BSMT, Dario Moccia, ecc.) NON serve riscaricarle: copiale
direttamente dentro data/raw/ (in qualsiasi formato audio) e lo step 2 le processerà.

Richiede: yt-dlp e ffmpeg nel PATH; feedparser e requests (pip install -r requirements-local.txt).
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

import config

# file sorgenti sovrascrivibile via env (per i batch v2/v3...): IMOSHI_YT_SOURCES
YT_LIST = config.ROOT / "sources" / os.environ.get("IMOSHI_YT_SOURCES", "youtube_sources.txt")
RSS_LIST = config.ROOT / "sources" / os.environ.get("IMOSHI_RSS_SOURCES", "podcast_rss.txt")
# l'archivio segue la RAW_DIR: con IMOSHI_RAW_DIR=data/raw_v2 diventa data/raw_v2/_yt_archive.txt
YT_ARCHIVE = config.RAW_DIR / "_yt_archive.txt"


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        print(f"[!] Manca {path}")
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def download_youtube() -> None:
    if shutil.which("yt-dlp") is None:
        sys.exit("[X] yt-dlp non trovato nel PATH. Installa con: pip install yt-dlp")
    if shutil.which("ffmpeg") is None:
        sys.exit("[X] ffmpeg non trovato nel PATH (serve a yt-dlp per estrarre l'audio).")

    urls = _read_lines(YT_LIST)
    if not urls:
        print("[!] Nessuna URL in sources/youtube_sources.txt"); return

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(config.RAW_DIR / "youtube" / "%(uploader)s" / "%(id)s.%(ext)s")

    # ANTI-BOT: YouTube blocca i download sequenziali troppo rapidi ("Sign in to
    # confirm you're not a bot" / rate-limit per ~1h). Due contromisure:
    #  1) cookies dal browser dove sei loggato a YouTube (la più efficace).
    #     Imposta YT_COOKIES_BROWSER=chrome|firefox|edge|brave prima di lanciare.
    #  2) pause randomizzate tra i download (sempre attive sotto).
    # Due modi (in OR): file cookies.txt esportato, oppure cookie letti dal browser.
    cookies_file = os.environ.get("YT_COOKIES_FILE", "").strip()
    cookies_browser = os.environ.get("YT_COOKIES_BROWSER", "").strip()
    if cookies_file:
        cookie_args = ["--cookies", cookies_file]
        print(f"    (uso cookies.txt: {cookies_file})")
    elif cookies_browser:
        cookie_args = ["--cookies-from-browser", cookies_browser]
        print(f"    (uso cookies dal browser: {cookies_browser})")
    else:
        cookie_args = []
        print("    (NESSUN cookie: rischio rate-limit. Imposta YT_COOKIES_FILE=cookies.txt "
              "oppure YT_COOKIES_BROWSER=firefox)")

    for i, url in enumerate(urls, 1):
        print(f"\n=== [{i}/{len(urls)}] YouTube: {url} ===")
        cmd = [
            "yt-dlp",
            "--download-archive", str(YT_ARCHIVE),  # ripartibilità
            "--ignore-errors",                       # un video privato/rimosso non blocca tutto
            "--no-overwrites",
            *cookie_args,
            "--sleep-requests", "1",                 # pausa tra le richieste API
            "--sleep-interval", "5",                 # pausa min tra i download (sec)
            "--max-sleep-interval", "30",            # pausa max -> randomizzata 5-30s
            "--retries", "10",
            "--socket-timeout", "30",                # se la connessione si appende, molla dopo 30s
            "--fragment-retries", "20",              # e ritenta il frammento invece di restare bloccato
            "--extractor-args", "youtube:player_client=default,web_safari",
            "-f", "bestaudio/best",
            "-x", "--audio-format", "opus",          # opus = qualità ottima, file piccoli
            "--audio-quality", "0",
            "--concurrent-fragments", "4",
            "-o", out_tmpl,
            url,
        ]
        # NB: niente check=True -> se un canale fallisce, passiamo al prossimo
        subprocess.run(cmd)

    print("\n[OK] Download YouTube completato (o ripreso).")


def download_rss() -> None:
    try:
        import feedparser
        import requests
    except ImportError:
        sys.exit("[X] Manca feedparser/requests. pip install feedparser requests")

    feeds = _read_lines(RSS_LIST)
    if not feeds:
        print("[!] Nessun feed in sources/podcast_rss.txt"); return

    base = config.RAW_DIR / "rss"
    base.mkdir(parents=True, exist_ok=True)

    for i, feed_url in enumerate(feeds, 1):
        print(f"\n=== [{i}/{len(feeds)}] RSS: {feed_url} ===")
        parsed = feedparser.parse(feed_url)
        show = (parsed.feed.get("title", f"feed_{i}") or f"feed_{i}")
        safe_show = "".join(c if c.isalnum() or c in " -_" else "_" for c in show).strip()[:60]
        show_dir = base / safe_show
        show_dir.mkdir(parents=True, exist_ok=True)
        print(f"    Podcast: {show} — {len(parsed.entries)} episodi")

        for entry in parsed.entries:
            mp3_url = None
            for enc in entry.get("enclosures", []):
                if "audio" in enc.get("type", "") or enc.get("href", "").endswith((".mp3", ".m4a")):
                    mp3_url = enc["href"]; break
            if not mp3_url:
                continue
            ep_id = entry.get("id", entry.get("title", "ep"))
            safe_id = "".join(c if c.isalnum() else "_" for c in ep_id)[:80]
            ext = ".mp3" if ".mp3" in mp3_url else ".m4a"
            dest = show_dir / f"{safe_id}{ext}"
            if dest.exists() and dest.stat().st_size > 1000:
                continue  # già scaricato
            try:
                with requests.get(mp3_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            f.write(chunk)
                    tmp.rename(dest)
                print(f"    + {dest.name}")
            except Exception as e:
                print(f"    [!] errore su {mp3_url}: {e}")

    print("\n[OK] Download RSS completato (o ripreso).")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("youtube", "all"):
        download_youtube()
    if mode in ("rss", "all"):
        download_rss()
    if mode not in ("youtube", "rss", "all"):
        print(__doc__)
