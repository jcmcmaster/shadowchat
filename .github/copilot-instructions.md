# Copilot Instructions for shadowchat

## Repository Summary

This is a **YouTube Audio QA system** that provides an end-to-end local pipeline for downloading audio from YouTube, transcribing it with Whisper, indexing transcripts in a vector database, and enabling question-answering with citations and timestamps. The system is designed to be Windows-friendly and runs entirely locally without requiring external API costs (except for optional OpenAI LLM features).

## High-Level Repository Information

- **Type**: Python CLI application
- **Size**: Small-to-medium Python project (~7 core modules)
- **Languages**: Python 3.12+
- **Key Frameworks**: 
  - yt-dlp (YouTube downloading)
  - faster-whisper (speech transcription)
  - ChromaDB (vector database)
  - sentence-transformers (embeddings)
  - OpenAI API (optional LLM analysis)
  - Rich (CLI formatting)
- **Target Runtime**: Local desktop/server environments (CPU or CUDA)

## Build Instructions

### Environment Setup
**ALWAYS follow this exact sequence:**

1. **Create virtual environment** (required):
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/Mac:
   source .venv/bin/activate
   ```

2. **Install dependencies** (expect 5-10 minutes for full install):
   ```bash
   pip install -r requirements.txt
   ```
   
   **Known Issues**:
   - Network timeouts are common - retry if pip fails
   - CUDA dependencies are large (~2GB) - use `--timeout 1000` if needed
   - On first run, Whisper models will download automatically (~100MB-1.5GB per model)

3. **Environment variables setup** (required for LLM features):
   ```bash
   cp .env.example .env
   # Edit .env and add: OPENAI_API_KEY=sk-your-key-here
   ```

### Basic Validation Commands

**Always run these to verify setup**:
```bash
# Test CLI is working
python main.py --help

# Verify data directories are created
python main.py stats  # Should show "Total sessions: 0"
```

### Complete Pipeline Test
To test the full pipeline with a short video:
```bash
# 1. Download test video (creates data/audio/)
python main.py download --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# 2. Transcribe (creates data/transcripts/, may take 1-5 minutes)
python main.py transcribe --model-size small.en

# 3. Index transcripts (creates data/index/)
python main.py index

# 4. Query the database
python main.py ask --question "What is being said?"

# 5. Check stats
python main.py stats
```

### Command Timing Expectations
- **Download**: 30 seconds - 5 minutes (depends on video length/quality)
- **Transcribe**: 1-10 minutes (depends on audio length and model size)
- **Index**: 10-60 seconds (depends on transcript length)
- **Query**: 1-5 seconds for retrieval, 10-30 seconds if using LLM analysis

### Build Troubleshooting

**Common Issues**:
1. **Network timeouts during pip install**: Retry with `pip install --timeout 1000 -r requirements.txt`
2. **CUDA memory errors**: Use `--device cpu` flag for transcription
3. **Permission errors on data/**: Ensure write permissions to repository directory
4. **Missing .env**: Copy `.env.example` to `.env` (LLM features won't work without API key)
5. **Model download failures**: Whisper models auto-download on first use - ensure internet connection

## Project Layout

### Core Architecture
```
shadowchat/
├── main.py                 # CLI entry point with subcommands
├── yt_audio_ai/           # Core package
│   ├── downloader.py      # YouTube audio downloading (yt-dlp)
│   ├── transcriber.py     # Audio → text transcription (faster-whisper)
│   ├── indexer.py         # Text → vector database (ChromaDB + sentence-transformers)
│   ├── qa.py              # Question answering and LLM analysis
│   ├── llm.py             # OpenAI API interface
│   └── utils.py           # Shared utilities
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variable template
└── data/                 # Generated at runtime
    ├── audio/            # Downloaded MP3 files + metadata JSON
    ├── transcripts/      # Whisper transcription JSON files
    └── index/            # ChromaDB persistent vector database
```

### Main Commands (CLI Interface)
All commands accessed via `python main.py <subcommand>`:

- **download**: YouTube URLs → MP3 + metadata
  - `--url`, `--urls-file`, `--playlist-id`, `--playlist-url`
- **transcribe**: MP3 → JSON transcripts
  - `--model-size` (small.en, medium, large-v3)
  - `--device` (cpu, cuda)
- **index**: Transcripts → vector database
  - `--embedding-model` (sentence-transformers model)
  - `--collection` (database collection name)
- **ask**: Query vector database
  - `--question` (required)
  - `--top-k` (number of results)
- **chat**: Interactive Q&A mode
- **stats**: Show database statistics

### Configuration Files
- **requirements.txt**: All Python dependencies (no lock file)
- **.env**: Environment variables (not committed, copy from .env.example)
- **.gitignore**: Excludes .env, .venv/, data/, urls.txt
- **.github/dependabot.yml**: Automated dependency updates

### Data Flow
1. **Input**: YouTube URLs or playlists
2. **Download**: yt-dlp extracts audio → `data/audio/{video_id}.mp3` + `{video_id}.info.json`
3. **Transcribe**: faster-whisper → `data/transcripts/{video_id}.json` (with timestamps)
4. **Index**: sentence-transformers embeddings → `data/index/` (ChromaDB persistent store)
5. **Query**: Natural language → semantic search with timestamp citations

### Testing and Validation
**No formal test suite exists** - validation is done through:
- Manual pipeline testing with sample videos
- `python main.py stats` to verify database state
- Query testing with known content

### Dependencies Not Obvious from File Structure
- **Model downloads**: Whisper models (100MB-1.5GB) download automatically to user cache
- **CUDA runtime**: Optional GPU acceleration requires NVIDIA drivers + CUDA toolkit
- **ChromaDB**: Creates SQLite database files in `data/index/`
- **OpenAI API**: Only required for `--analysis` mode in chat/ask commands

### Key Implementation Details
- **Incremental processing**: Re-runs skip existing files (download/transcribe/index)
- **Chunking strategy**: Transcripts split into ~1500 character chunks with timestamp preservation
- **Citation format**: Results include YouTube URLs with timestamp parameters (`?t=123`)
- **Collection support**: Multiple vector databases can coexist with `--collection` flag
- **Model consistency**: Always use same embedding model for indexing and querying

### Error Patterns to Watch For
1. **Network timeouts**: Both pip install and model downloads can fail
2. **Memory issues**: Large Whisper models may OOM on smaller systems
3. **Path issues**: Windows path separators, permission errors on data/ directory
4. **API key missing**: OpenAI features fail silently without proper .env setup
5. **Model mismatch**: Changing embedding models requires full re-indexing

### Files in Repository Root
```
├── .env.example           # Environment variable template
├── .gitignore            # Git ignore rules
├── README.md             # Detailed usage documentation
├── main.py               # CLI entry point (243 lines)
├── requirements.txt      # 12 Python dependencies
├── urls.sample.txt       # Example URL file (single YouTube URL)
└── yt_audio_ai/          # Core Python package (7 modules)
```

## Key Guidance for Agents

1. **Always use virtual environments** - dependency conflicts are common
2. **Expect long installation times** - budget 10-15 minutes for full setup
3. **Test incrementally** - run each pipeline step individually to isolate issues
4. **Check data/ permissions** - the application must create directories and files
5. **Verify model downloads** - first transcription will download Whisper models
6. **Use small test videos** - validate pipeline before processing large content
7. **Monitor disk space** - audio files, models, and vector databases consume storage
8. **Trust these instructions** - only search/explore if information is incomplete or incorrect

### Recommended Development Workflow
1. Set up environment (venv + pip install)
2. Test with single short video end-to-end
3. Verify each command works independently
4. Use `stats` command to validate database state
5. For changes: test download → transcribe → index → query cycle