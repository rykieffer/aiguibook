# AIGUIBook - AI Audiobook Generator

Transform EPUB ebooks into audiobooks (M4B) with AI character voices and emotion acting.

## Features

- **EPUB Parsing**: Extracts chapters, metadata, TOC from any EPUB 2/3 file
- **Smart Segmentation**: Splits text into TTS-friendly segments (40-100 words), never mid-sentence
- **Character & Emotion Analysis**: LLM-powered detection of speakers and their emotions (angry, sad, whisper, etc.)
- **Two Voice Modes**:
  - **Single Narrator** (default): One voice acts out ALL characters with emotional shifts
  - **Multi-Cast**: Unique AI-generated voice per character, each with acting
- **Voice Design**: Describe a voice in text → AI generates a unique WAV reference
- **Voice Cloning**: Use any 3+ second WAV as reference, clone it with emotion acting
- **faster-qwen3-tts Engine**: CUDA-graph optimized inference, no flash-attn needed
- **Text-Embedded JSON**: Analysis results include the full book text — no need to re-parse the EPUB
- **Whisper Validation**: Optional quality check via faster-whisper (WER scoring)
- **M4B Output**: Chapter markers, metadata, loudness normalization

## Architecture

```
EPUB → Parse → Segment → Analyze (LLM) → Voice Design → Generate (TTS) → Validate → M4B
                                   │                              │
                                   └── Saves to JSON ─────────────┘
                                      (includes full text)
```

### Voice Pipeline

1. **VoiceDesign** (`1.7B-VoiceDesign`): Generate a reference WAV from a text description
2. **VoiceClone** (`1.7B-Base`): Clone the reference WAV with emotion instructions for audiobook generation

The emotion instructions are passed via the `instruct` parameter of `generate_voice_clone()`, which guides the model's tone and style without modifying the text.

## Installation

```bash
# Clone the repo
git clone https://github.com/rykieffer/aiguibook.git
cd aiguibook

# Create conda environment
conda create -n aiguibook python=3.12
conda activate aiguibook

# Install dependencies
pip install -r requirements.txt
pip install faster-qwen3-tts

# System dependencies (Linux)
sudo apt install ffmpeg sox libsox-dev
```

### RTX 5080 / Blackwell Note

RTX 50xx GPUs need CUDA 12.8+ PyTorch wheels:
```bash
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Usage

### GUI Mode (Recommended)

```bash
python main.py
```

Open http://localhost:7860 in your browser.

#### Workflow

1. **Tab 1 - Analysis**: Upload EPUB → Parse → Run Character Analysis → Save JSON
2. **Tab 2 - Voice Design**: Design narrator voice (describe it) → Optionally design character voices → Set voice strategy (Single Narrator or Multi-Cast)
3. **Tab 3 - Production**: Click START GENERATION → Progress bar tracks each segment

### CLI Mode

```bash
# Parse an EPUB and show metadata
python cli.py parse --input book.epub

# Full pipeline: EPUB to M4B
python cli.py generate --input book.epub --output ./output

# Character analysis only
python cli.py analyze --input book.epub
```

### The JSON Workflow

Once you save the analysis JSON, you never need the EPUB again:

1. Parse EPUB + Run Analysis → Save JSON (includes all text + emotions)
2. Next session: Load JSON → Go straight to Production

## Configuration

Config is stored at `~/.aiguibook/config.yaml`. Key settings:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `tts` | `model` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | TTS model variant |
| `tts` | `device` | `cuda` | Compute device |
| `analysis` | `llm_backend` | `lmstudio` | LLM backend (lmstudio/ollama/openrouter) |
| `analysis` | `openrouter_api_key` | | OpenRouter API key |
| `output` | `format` | `m4b` | Output format |
| `output` | `bitrate` | `128k` | Audio bitrate |
| `validation` | `enabled` | `true` | Whisper validation |
| `general` | `language` | `french` | Primary language |

## Project Structure

```
audiobook_ai/
├── core/
│   ├── config.py           # YAML configuration
│   ├── epub_parser.py      # EPUB extraction
│   ├── project.py          # Project/state management
│   └── text_segmenter.py   # Sentence-aware text splitting
├── analysis/
│   └── character_analyzer.py  # LLM-based character/emotion detection
├── tts/
│   ├── qwen_engine.py      # faster-qwen3-tts wrapper
│   └── voice_manager.py    # Voice profile management
├── audio/
│   ├── assembly.py         # M4B assembly with chapter markers
│   └── validation.py       # Whisper-based quality check
└── gui/
    └── app.py              # Gradio web interface
```

## License

MIT License
