# Italian moshi
Moshi è un IA con cui puoi parlare in tempo reale, senze le classiche pause tipiche delle conversazioni a turni con l'AI. La puoi eseguire in locale con una GPU con 24GB di vram.

<img width="721" height="581" alt="222" src="https://github.com/user-attachments/assets/d9ebe90b-5175-455a-b6a6-dc1162927adc" />

Questo repo contiene gli script per costruirti il dataset da
solo, controllarne la qualità, e le config di training
validate su 1×H100 80GB. Con questi strumenti e ~1500h di parlato dialogico italiano
riproduci l'intero esperimento.


[Moshi](https://github.com/kyutai-labs/moshi) è il modello vocale full-duplex di Kyutai, `kyutai/moshiko-pytorch-bf16`, 7B. I-Moshi è un tentativo di replicare quello che hanno fatto i giapponesi con [J-Moshi](https://arxiv.org/abs/2506.02979), ovvero il finetuning in italiano del modello.
```
sorgenti audio (scegli tu) ──▶ pipeline dati (questo repo, GPU locale)
                                      │  wav stereo 24k + json allineamenti
                                      ▼
                          manifest clean (train/eval jsonl)
                                      │  upload (S3/Spaces o rsync)
                                      ▼
                  moshi-finetune su 1×H100 (config A o B, questo repo)
                                      │  checkpoint fuso ~15GB
                                      ▼
                          inferenza locale (moshi.server)
```
Puoi supportare lo sviluppo su Ko-Fi [offrendo un caffè ☕](https://ko-fi.com/goodmann) 
### Modelli su huggingface

- Primo modello trainato su 1400h con config_A: https://huggingface.co/goodman117/moshi-ita-A2
- Secondo modello trainato su 2400h con configB (work in progress)
---

## 1. Il formato dati di Moshi (perché la pipeline è fatta così)

moshi-finetune si aspetta **dialoghi a 2 voci** in questo formato:

- **wav stereo 24kHz PCM16**: canale SINISTRO = la voce che Moshi impara a imitare
  ("main"), canale DESTRO = l'interlocutore;
- **json gemello** con gli allineamenti word-level:
  `{"alignments": [[parola, [start, end], "SPEAKER_MAIN"], ...]}`;
- **manifest jsonl**: una riga `{"path": "...", "duration": secondi}` per campione,
  path relativi alla cartella del jsonl.

⚠️ **La regola più importante di tutto il repo**: il training tiene SOLO le parole
etichettate **`SPEAKER_MAIN`** (`keep_main_only=True` nell'interleaver) e scarta
silenziosamente ogni altra etichetta. Se etichetti i canali "A"/"B", il flusso testo
arriva **VUOTO** al training, la loss sembra normale, e il modello esce rotto — l'abbiamo
scoperto dopo un run intero da $100 (vedi §7, Run 1). Gli script di questo repo emettono
già l'etichetta giusta; se costruisci dati per altre vie, verificala col preflight (09).

I podcast sono mono: la pipeline ricrea lo stereo "finto" con la diarization
(l'approccio J-CHAT/J-Moshi): speaker dominante → canale SX, secondo → DX.

## 2. Requisiti

**Pipeline dati (locale)**
- GPU NVIDIA (usata una RTX 5090; qualunque GPU ≥12GB regge diarization+whisper)
- Python 3.10+, ffmpeg nel PATH
- `pip install torch torchaudio` (build adatta alla tua GPU), poi
  `pip install -r requirements-local.txt`
- Account HuggingFace con i gate accettati (gratis, un click):
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  e [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
  poi `hf auth login` (o `HF_TOKEN` nell'ambiente)
- Spazio disco: ~3× le ore grezze (raw compresso + stereo PCM16). 1500h ≈ 600GB.

**Training (cloud)**
- 1×H100 80GB (usata una GPU Droplet DigitalOcean, immagine "AI/ML Ready",
  720GB NVMe). Full finetuning NON è possibile su 1 GPU (servono ~8×H100):
  si usa LoRA, che per l'italiano basta (scrittura latina → il tokenizer di
  Moshi la copre già; il giapponese invece richiese full FT).
- [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) (repo Kyutai)

## 3. Pipeline dati, passo-passo

Tutti gli script leggono `config.py` (path sovrascrivibili via variabili d'ambiente:
`IMOSHI_RAW_DIR`, `IMOSHI_STEREO_DIR`, `IMOSHI_YT_SOURCES`, `IMOSHI_RSS_SOURCES` —
utile per tenere separati batch successivi di sorgenti).

### 3.1 Scegli le sorgenti → `sources/`
Copia `sources/youtube_sources.example.txt` in `youtube_sources.txt` e metti i TUOI
URL (canali/playlist). **Criterio d'oro: dialoghi 1-a-1** (host+ospite). Vedi i
commenti nel file example per cosa evitare (3+ voci, monologhi) e perché.

### 3.2 Download → `01_download.py`
```bash
python 01_download.py youtube     # e/o: rss, all
```
Riprendibile (archivio yt-dlp), estrae solo l'audio (opus). Anti-bot: imposta
`YT_COOKIES_BROWSER=firefox` (o `YT_COOKIES_FILE=cookies.txt`) o YouTube ti
rate-limita dopo poche decine di video.

### 3.3 Macinamento → `02_process.py` (il cuore, giorni di GPU)
```bash
python 02_process.py
```
Per ogni file: decodifica → finestre da 10 min → diarization forzata a 2 speaker
(pyannote 3.1) → stereo finto SX/DX → filtro anti-monologo (2° speaker ≥15% del
tempo) → trascrizione word-level per canale (faster-whisper `medium`, lingua=it) →
wav+json. **Idempotente**: interrompi e rilanci quando vuoi, salta il già fatto.
Ordine di grandezza sulla 5090: ~250-350 file/h di processing per file da ~1h.

### 3.4 Health check → `04_health_check.py`
```bash
python 04_health_check.py --csv data/health.csv
```
KPI per campione: formato, quota 2° speaker, parole/min, allucinazioni whisper
(bigrammi ripetuti), italianità, canali muti, **turni/minuto e durata mediana del
turno** (il "botta-e-risposta": la metrica che più predice l'utilità per il
turn-taking di Moshi), riepilogo per canale. Verdetto: ore clean.

### 3.5 (Fortemente consigliato) Probe 3-voci → `07_multivoice_probe.py`
```bash
python 07_multivoice_probe.py --per-channel 25 --channels "Canale-1,Canale-2"
```
**Il punto cieco n°1 della pipeline**: la diarization forzata a 2 speaker NON scarta
i contenuti a 3+ voci — li spalma sui 2 canali, e nessun filtro testuale se ne
accorge. Questo probe ridiarizza un campione con `max_speakers=4` e misura la
contaminazione reale per canale. Sul nostro dataset: canali "da interviste" che
sembravano puliti erano contaminati fino al **75%**. Con `--per-channel 0` fai la
scansione completa dei canali incriminati e il CSV di verdetti viene usato da 05
per escludere le finestre contaminate.

### 3.6 (Opzionale) Probe flip del main → `08_main_flip_probe.py`
Misura con speaker-embedding quante volte la voce "main" cambia canale tra finestre
consecutive dello stesso episodio (succede coi co-host paritari: dominanza che
oscilla → l'identità di Moshi "salta" a metà conversazione). Solo misura, il fix
(swap canali) è documentato ma non incluso.

### 3.7 Manifest clean → `05_clean_manifest.py`
```bash
python 05_clean_manifest.py
```
Produce `train_clean.jsonl`, `eval_clean.jsonl`, `clean_files.txt` (lista upload).
Applica: criteri clean dell'health check, esclusione finestre multi-voce (dal CSV
del probe 07, se presente), dedup per video-ID. **L'eval è costruito con regole
severe imparate a caro prezzo** (vedi §7): solo finestre piene ≥590s, episodi a
finestra singola (zero leakage col train), ≥1 parola main in ogni segmento da 100s.

### 3.8 Gate finale → `09_preflight.py`
```bash
python 09_preflight.py     # exit 0 = via libera
```
Valida i MANIFEST contro il contratto reale di moshi-finetune (verificato sul
sorgente): coppie esistenti, etichette SPEAKER_MAIN ovunque, regole eval, zero
leakage, spot-check formato wav. **Lancialo sempre prima di upload/training.**

## 4. Trasferimento dati → cloud

Con un object storage S3-compatible (usato DigitalOcean Spaces, stessa regione
della GPU per la rete interna veloce):
```bash
# dal PC (dentro la cartella data/): carica SOLO i file del manifest
s3cmd sync --files-from=clean_files.txt . s3://TUO-BUCKET/
# dalla droplet: scarica (rete interna: nel nostro caso 244MB/s reali)
s3cmd sync --files-from=clean_files.txt s3://TUO-BUCKET/ /root/imoshi/data/
```
In alternativa diretto: `rsync -a --info=progress2 --files-from=clean_files.txt . root@IP:/root/imoshi/data/`.
Lo storage intermedio conviene se prevedi di distruggere/ricreare la GPU (la GPU
fattura anche da spenta: si paga solo se DISTRUTTA — e i dati sopravvivono nel bucket).


## 5. Training su H100

### Setup ambiente
```bash
tmux new -s train        # il training dura decine di ore, tienilo in tmux

conda create -n moshi_env python=3.10 -y
conda activate moshi_env
pip install torch torchaudio

git clone https://github.com/kyutai-labs/moshi-finetune
cd moshi-finetune
pip install -e .
```

### Scegli la config: A o B

| | `configs/moshi_A.yaml` | `configs/moshi_B.yaml` |
|---|---|---|
| Embedding | congelati | **sbloccati** (`ft_embed: true`) |
| Cosa impara | accento, prosodia | + lessico/semantica (adattamento di lingua vero) |
| Checkpoint | adapter LoRA leggero (`save_adapters: true`) | modello fuso ~15GB (`save_adapters: false`) |
| Inferenza | `--lora-weight` | `--moshi-weight` |
| VRAM di picco | ~38GB | ~55GB |

Copia `configs/moshi_A.yaml` o `configs/moshi_B.yaml` nella cartella di
`moshi-finetune`, e aggiorna i path (`data.train_data`, `data.eval_data`,
`run_dir`) con quelli reali della tua macchina.

### Smoke-test (50 step, ~15 min)
Prima del run lungo, verifica che tutto funzioni con pochi step:
```bash
# nel tuo file di config, temporaneamente: max_steps: 50, ckpt_freq: 25, run_dir diverso
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m train moshi_smoke.yaml 2>&1 | tee smoke.log
```
Controlla: nessun errore, VRAM di picco in linea con la tabella sopra, checkpoint
scritto nella cartella `run_dir/checkpoints/checkpoint_000050/consolidated/`
(per la config B deve pesare ~15GB).

### Run vero
Ripristina i valori definitivi nel config (`max_steps`, `ckpt_freq`, `run_dir`) e lancia:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m train moshi_A.yaml 2>&1 | tee run.log
# oppure moshi_B.yaml
```

`max_steps` per ~2 epoche:
```
max_steps ≈ (ore_dataset × 3600) / (batch_size × duration_sec) × 2
```
Con le config qui incluse (`batch_size: 16`, `duration_sec: 100`) e ~1500h di
dataset: ~6300 step, ~30h su 1×H100, stimabile in ~$100-110 al prezzo corrente
della GPU.

Monitoraggio durante il run (da un'altra finestra/pane tmux):
```bash
tail -f run.log          # train_loss deve scendere
watch -n5 nvidia-smi      # uso GPU/VRAM
df -h                     # spazio disco (i checkpoint ruotano)
```

Per staccarti da tmux senza interrompere il training: `Ctrl+B` poi `D`. Per
rientrare: `tmux attach -t train`.

## 9. Papers

- [Kyutai](https://kyutai.org) per Moshi e moshi-finetune
- [J-Moshi](https://arxiv.org/abs/2506.02979) (Nagoya Univ.) per aver mostrato la strada dell'adattamento di lingua
- pyannote.audio (diarization), faster-whisper/CTranslate2 (trascrizione)


