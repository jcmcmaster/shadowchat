## YouTube Audio QA (local, Windows-friendly)

End-to-end local pipeline to download audio from YouTube, transcribe with Whisper, index with a local vector DB, and ask questions with citations + timestamps.

### Quickstart (PowerShell)

1) Create venv and install deps

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Prepare URLs or playlists

- Put one per line in `urls.txt` (video or playlist URLs are fine). Or use flags.

3) Run pipeline

```
# Download individual URLs
python main.py download --urls-file urls.txt
# Or: download a playlist by ID or URL
python main.py download --playlist-id PLAYLIST_ID
python main.py download --playlist-url https://www.youtube.com/playlist?list=PLAYLIST_ID

# Transcribe all downloaded audio files
python main.py transcribe --model-size medium

# Index transcripts
python main.py index --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Ask questions
python main.py ask --question "What did they say about X?"
```

 ### Environment variables (.env)

 - Create a `.env` at repo root (not committed) to store secrets like `OPENAI_API_KEY`.
 - Example keys:
   - `OPENAI_API_KEY=sk-...`
   - `OLLAMA_HOST=http://localhost:11434`
 - PowerShell helper to load `.env` into current session:
   ```powershell
   Get-Content .env | ForEach-Object {
     if (-not ($_ -match '^(\s*#|\s*$)')) { $k,$v = $_.Split('=',2); [Environment]::SetEnvironmentVariable($k,$v,'Process') }
   }
   ```

Citations include clickable timecodes like `https://youtu.be/VIDEO_ID?t=123`.

### Data layout

- `data/audio/{id}.mp3` and `data/audio/{id}.info.json`
- `data/transcripts/{id}.json`
- `data/index/` (Chroma persistent store)

### Notes

- For best quality, use `--model-size large-v3` (requires more RAM and time). `medium` is a good CPU default.
- Re-runs skip already-downloaded audio and transcripts when possible.
- Fully local: no API costs.
