## YouTube Audio QA (local, Windows-friendly)

End-to-end local pipeline to download audio from YouTube, transcribe with Whisper, index with a local vector DB, and ask questions with citations + timestamps.

### Quickstart (PowerShell)

1) Create venv and install deps

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Note for GPU acceleration:** The requirements.txt includes PyTorch with CUDA 12.1 support for GPU-accelerated transcription and diarization. If you encounter CUDA compatibility issues or prefer CPU-only operation, you can install the CPU version:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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

# Transcribe with speaker diarization (requires Hugging Face token in .env)
python main.py transcribe --model-size medium --enable-diarization

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
3. Add `HUGGINGFACE_TOKEN=hf_your-token-here` to your `.env` file
4. For GPU acceleration, ensure PyTorch with CUDA support is installed (included in requirements.txt)

#### Usage:
```bash
# Enable diarization during transcription (uses HUGGINGFACE_TOKEN from .env)
python main.py transcribe --enable-diarization
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

**See [Vector Database Management](#vector-database-management) below for detailed information on updating the database with new sessions.**

### Data layout

- `data/audio/{id}.mp3` and `data/audio/{id}.info.json`
- `data/transcripts/{id}.json`
- `data/index/` (Chroma persistent store)

### Vector Database Management

The system uses ChromaDB as a persistent vector database to store and search through transcribed audio content. Understanding how to manage and update this database is crucial for maintaining an effective QA system.

#### Understanding Sessions

In this system, a **session** refers to a single YouTube video or audio recording that has been:
- Downloaded as an MP3 file (`data/audio/{video_id}.mp3`)
- Transcribed into segments (`data/transcripts/{video_id}.json`)  
- Indexed into the vector database as searchable chunks

Each session is identified by its unique YouTube video ID and can contain multiple text chunks that are searchable independently.

#### How the Vector Database Works

1. **Chunking**: Transcripts are broken into overlapping chunks (default: 1500 characters) to optimize retrieval
2. **Embedding**: Each chunk is converted to a vector using sentence-transformers (default: `all-MiniLM-L6-v2`)
3. **Storage**: Vectors and metadata are stored in ChromaDB collections
4. **Retrieval**: Queries are embedded and matched against stored vectors using semantic similarity

#### Adding New Sessions to the Database

When you have new audio content to add to your database:

1. **Download and transcribe new content first**:
   ```bash
   # Add new URLs to your urls.txt or use direct flags
   python main.py download --url "https://youtube.com/watch?v=NEW_VIDEO_ID"
   python main.py transcribe --model-size medium
   ```

2. **Index the new transcripts**:
   ```bash
   # This will automatically detect and index only new transcripts
   python main.py index
   ```

The indexer automatically handles incremental updates:
- **Skip existing**: Already-indexed videos are not re-processed unless explicitly overwritten
- **Preserve consistency**: Uses the same embedding model and chunking strategy
- **Maintain timeline**: Preserves chronological ordering across all sessions

#### Practical Workflow Examples

**Example 1: Adding a single new video**
```bash
# Add one new video to existing database
python main.py download --url "https://youtube.com/watch?v=abc123"
python main.py transcribe
python main.py index
python main.py stats  # Verify it was added
```

**Example 2: Batch adding multiple videos**
```bash
# Add multiple URLs to urls.txt, then process all at once
echo "https://youtube.com/watch?v=video1" >> urls.txt
echo "https://youtube.com/watch?v=video2" >> urls.txt
python main.py download --urls-file urls.txt
python main.py transcribe --model-size medium
python main.py index
```

**Example 3: Adding a playlist incrementally**
```bash
# Process new playlist while preserving existing database
python main.py download --playlist-url "https://youtube.com/playlist?list=PLxxxxxx"
python main.py transcribe
python main.py index  # Only new videos get indexed
python main.py ask --question "What topics were covered?"
```

#### Database Update Process

The `index` command performs these steps for each new transcript:

1. **Load transcript**: Reads the JSON file containing segments and metadata
2. **Extract metadata**: Pulls video title, upload date, and original URL from `.info.json`
3. **Create chunks**: Segments text into optimal sizes for retrieval
4. **Generate embeddings**: Converts text chunks to vectors using the specified model
5. **Store in database**: Adds documents, embeddings, and metadata to the ChromaDB collection

Each chunk gets stored with rich metadata:
```python
{
    "video_id": "abc123",
    "title": "Video Title",
    "start": 45.2,           # Timestamp in video
    "end": 67.8,
    "url": "https://youtube.com/watch?v=abc123",
    "chunk_index": 0,
    "upload_date": "20241201",
    "upload_timestamp": 1733011200,
    "global_start": 1733011245,  # Global timeline position
    "global_end": 1733011267
}
```

#### Collection Management

**Default collection**: `youtube_audio`
**Custom collections**: Use `--collection` flag to organize different content types

```bash
# Create separate collections for different topics
python main.py index --collection "tech_talks"
python main.py ask --collection "tech_talks" --question "What about AI?"
```

#### Database Maintenance

**Monitoring database growth**:
```bash
# Check how many sessions are indexed
python main.py stats

# Check database size on disk
du -sh data/index/
```

**Re-indexing**: If you need to rebuild the entire database:
```bash
# This will re-process all transcripts (useful after model changes)
# Note: This recreates the collection from scratch
rm -rf data/index/  # Remove existing database
python main.py index
```

**Model changes**: If you switch embedding models, you must re-index:
```bash
python main.py index --embedding-model "sentence-transformers/all-mpnet-base-v2"
```

**Consistency**: Always use the same embedding model for querying as you used for indexing.

**Partial cleanup**: To remove specific sessions, delete the corresponding files and re-index:
```bash
# Remove a specific session completely
rm data/audio/{video_id}.*
rm data/transcripts/{video_id}.json
# Then recreate index to remove from database
rm -rf data/index/
python main.py index
```

#### Database Inspection and Verification

**Check indexed content**:
```bash
# List all indexed sessions with titles and chronological order
python main.py stats

# Test database with a simple query
python main.py ask --question "test" --top-k 1

# Interactive exploration
python main.py chat
```

**Verify successful indexing**: After adding new sessions, check:
1. New files appear in `data/transcripts/` (one `.json` per video)
2. `python main.py stats` shows increased session count
3. Test queries return content from new sessions
4. No error messages during the indexing process

**Understanding the file structure**:
```
data/
├── audio/
│   ├── {video_id}.mp3        # Downloaded audio
│   └── {video_id}.info.json  # Video metadata from YouTube
├── transcripts/
│   └── {video_id}.json       # Whisper transcription with timestamps
└── index/                    # ChromaDB persistent database
    └── chroma.sqlite3        # Vector embeddings and metadata
```

#### Performance Considerations

- **Incremental updates**: Adding new sessions is fast - only new content is processed
- **Memory usage**: Larger embedding models require more RAM but provide better semantic understanding
- **Storage**: ChromaDB compresses vectors efficiently; expect ~1-2MB per hour of audio content
- **Query speed**: Searches are typically sub-second even with thousands of chunks

#### Requirements and Setup

**Network access**: First-time setup requires internet connection to download embedding models from Hugging Face
**Model caching**: Once downloaded, models are cached locally for offline use
**Storage space**: Embedding models typically require 100-500MB of disk space

#### Troubleshooting

**Connection errors**: Ensure internet access for initial model download. Once cached, the system works offline
**Empty results**: Ensure query and index use the same embedding model
**Memory errors**: Use smaller models like `all-MiniLM-L6-v2` or reduce chunk size  
**Inconsistent results**: Verify all transcripts were indexed successfully - check for error messages during indexing
**Model not found**: Delete model cache (`~/.cache/huggingface/`) if models become corrupted

### Notes

- For best quality, use `--model-size large-v3` (requires more RAM and time). `medium` is a good CPU default.
- Re-runs skip already-downloaded audio and transcripts when possible.
- Fully local: no API costs.
- Speaker diarization adds processing time but provides speaker identification in multi-speaker recordings.
- Diarization requires a Hugging Face token in your `.env` file and internet access during initial model download.
