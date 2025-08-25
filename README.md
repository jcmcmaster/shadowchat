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

# Transcribe with speaker diarization (requires Hugging Face token)
python main.py transcribe --model-size medium --enable-diarization --diarization-token YOUR_HF_TOKEN

# Index transcripts
python main.py index --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Ask questions
python main.py ask --question "What did they say about X?"
```

 ### Environment variables (.env)

 - Create a `.env` at repo root (not committed) to store secrets like `OPENAI_API_KEY`.
 - Example keys:
   - `OPENAI_API_KEY=sk-...`
   - `HUGGINGFACE_TOKEN=hf_...` (required for speaker diarization)
 - PowerShell helper to load `.env` into current session:
   ```powershell
   Get-Content .env | ForEach-Object {
     if (-not ($_ -match '^(\s*#|\s*$)')) { $k,$v = $_.Split('=',2); [Environment]::SetEnvironmentVariable($k,$v,'Process') }
   }
   ```

Citations include clickable timecodes like `https://youtu.be/VIDEO_ID?t=123`.

### Speaker Diarization

Speaker diarization identifies different speakers in audio and assigns speaker labels to transcript segments. This feature uses pyannote.audio and is optional.

#### Setup:
1. Get a Hugging Face token from https://huggingface.co/settings/tokens
2. Accept the user agreement for the pyannote model at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Use the token with `--diarization-token` or set `HUGGINGFACE_TOKEN` in your `.env` file

#### Usage:
```bash
# Enable diarization during transcription
python main.py transcribe --enable-diarization --diarization-token YOUR_HF_TOKEN
```

#### Output Format:
With diarization enabled, transcript segments include an optional `speaker` field:
```json
{
  "text": "Hello there",
  "start": 1.23,
  "end": 3.45,
  "speaker": "SPEAKER_00"
}
```

The speaker field is backward compatible - existing transcripts without speaker information continue to work normally.

### Data layout

- `data/audio/{id}.mp3` and `data/audio/{id}.info.json`
- `data/transcripts/{id}.json`
- `data/index/` (Chroma persistent store)

### Notes

- For best quality, use `--model-size large-v3` (requires more RAM and time). `medium` is a good CPU default.
- Re-runs skip already-downloaded audio and transcripts when possible.
- Fully local: no API costs.
- Speaker diarization adds processing time but provides speaker identification in multi-speaker recordings.
- Diarization requires a Hugging Face token and internet access during initial model download.
