"""
Configurazione centrale della pipeline i-moshi (italian-moshi).

Tutti gli script (download, process, build_dataset) importano da qui.
NIENTE segreti hardcodati: il token HuggingFace si legge da variabile d'ambiente.

    Windows PowerShell:  $env:HF_TOKEN = "hf_xxx"
    Linux/macOS:         export HF_TOKEN=hf_xxx
"""
import os
from pathlib import Path

# Radice della pipeline = questa cartella
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Cartelle dati
# ---------------------------------------------------------------------------
DATA_DIR    = ROOT / "data"
# RAW_DIR e STEREO_DIR sono sovrascrivibili via variabile d'ambiente, così ogni
# "sotto-dataset" (batch di canali diverso) può stare in cartelle separate senza
# duplicare gli script. Esempi:
#   $env:IMOSHI_RAW_DIR    = "data/raw_v2"     (PowerShell)
#   export IMOSHI_RAW_DIR=data/raw_v2          (bash)
# I default restano data/raw e data/stereo (il dataset originale).
RAW_DIR     = Path(os.environ.get("IMOSHI_RAW_DIR", DATA_DIR / "raw"))
STEREO_DIR  = Path(os.environ.get("IMOSHI_STEREO_DIR", DATA_DIR / "stereo"))
# train.jsonl / eval.jsonl vengono scritti in DATA_DIR (i path nel jsonl sono relativi a qui)

# ---------------------------------------------------------------------------
# Segreti
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
TARGET_SR = 24_000          # sample rate richiesto da Moshi/Mimi (NON cambiare)
WHISPER_SR = 16_000         # whisper lavora a 16k
LANGUAGE = "it"

# Modello whisper-timestamped. "medium" è il consigliato per audio stereo/diarizzato
# (vedi annotate.py di Kyutai). "large-v3" è più preciso ma ~2-3x più lento.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")

# ---------------------------------------------------------------------------
# Diarization (pyannote 3.1)
# ---------------------------------------------------------------------------
# Per podcast/interviste a 2 voci forzare esattamente 2 speaker migliora ENORMEMENTE
# lo split L/R. Se hai materiale con 3+ persone, alza DIAR_MAX_SPEAKERS.
DIAR_MIN_SPEAKERS = int(os.environ.get("DIAR_MIN_SPEAKERS", "2"))
DIAR_MAX_SPEAKERS = int(os.environ.get("DIAR_MAX_SPEAKERS", "2"))

# ---------------------------------------------------------------------------
# Finestratura (chunking) e filtri qualità
# ---------------------------------------------------------------------------
# I file lunghi vengono spezzati in finestre: tiene bassa la RAM della diarization,
# velocizza whisper e migliora lo shuffling in training. Ogni finestra = 1 campione.
WINDOW_SEC = float(os.environ.get("WINDOW_SEC", "600"))   # 10 minuti
MIN_DURATION_SEC = 30.0     # scarta finestre/file più corti di così

# Il 2° speaker deve parlare almeno questa frazione del tempo: garantisce VERO dialogo
# (scarta monologhi mascherati da conversazione). 0.15 = 15%.
MIN_SECONDARY_SPEAKER_RATIO = float(os.environ.get("MIN_SECONDARY_RATIO", "0.15"))

# Estensioni audio riconosciute nella cartella raw
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".opus", ".webm", ".flac", ".ogg", ".aac", ".mp4"}
