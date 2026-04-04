UNDER DEVELOPMENT!!!

# AIGUIBook - AI-Powered Audiobook Generator

**EPUB -> High-Quality French Audiobook with Character Voices & Emotions**

Transform your EPUB ebooks into professional-quality audiobooks with multiple character voices, emotion-aware delivery, and full chapter metadata. Runs entirely on your local GPU.

---

## Features

- **Multi-voice audiobooks**: Each character gets a distinct AI voice
- **Emotion-aware delivery**: Detects anger, sadness, excitement, whispered passages, tension, etc.
- **Voice cloning**: Clone any voice from 3 seconds of audio
- **Voice design**: Create unique voices from text descriptions
- **Character detection**: LLM automatically identifies who speaks what
- **Quality validation**: Whisper-based verification with auto-retry on errors
- **Chapter metadata**: Full M4B output with proper chapter markers
- **Resume support**: Pause and resume generation at any point
- **Beautiful GUI**: Gradio-based web interface, runs at localhost:7860
- **French-first**: Optimized for French language with English fallback

---

## Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (16GB+ recommended for RTX 5080/4080/4090)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free (for models: ~4GB)
- **FFmpeg**: Must be installed and in PATH

---

## Quick Installation (WSL2 on Windows 11 - Recommended)

No need for Python on Windows or Docker/Podman. WSL2 has direct GPU passthrough.

```bash
# 1. In WSL2 Ubuntu terminal:
sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev ffmpeg

# 2. Verify GPU is visible
nvidia-smi   # Should show your RTX 5080

# 3. Navigate to the project
cd /path/to/audiobook-ai

# 2. Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.4 support
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install qwen-tts
pip install qwen-tts

# 6. Optional: Install flash-attention for faster inference
pip install flash-attn --no-build-isolation

# 7. Make sure FFmpeg is installed
# Ubuntu/Debian:
sudo apt install ffmpeg
# Arch Linux:
sudo pacman -S ffmpeg
# Nix:
nix-env -iA nixpkgs.ffmpeg

# 8. Set environment variables (optional, for OpenRouter API character analysis)
export OPENROUTER_API_KEY="sk-or-..."

# 9. Launch the GUI
python main.py
```

---

## Usage

### GUI Mode (Recommended)

```bash
python main.py
```

Opens at **http://localhost:7860** with 5 tabs:

1. **Setup** - Upload EPUB, view book info, configure output settings
2. **Voices** - Assign voices to characters, create new voices, preview them
3. **Preview** - Test TTS with sample text and parameters
4. **Generate** - Run the full pipeline with progress tracking, pause/resume
5. **Settings** - TTS, LLM, output, and validation configuration

### CLI Mode

```bash
# Full pipeline: EPUB -> M4B
python cli.py generate --input "book.epub" --output "./output/"

# Preview mode (first 3 chapters only)
python cli.py generate --input "book.epub" --output "./output/" --preview-only

# Just parse and inspect the EPUB
python cli.py parse --input "book.epub"

# Run character/emotion analysis only
python cli.py analyze --input "book.epub"

# First-time setup: download TTS models, test GPU
python cli.py setup --download-models

# List available voice profiles
python cli.py voices --list

# Create default voices
python cli.py voices --create default
```

---

## Pipeline Overview

```
EPUB File
    |
    v
[1] EPUB Parser -- Extract chapters, metadata, TOC
    |
    v
[2] Text Segmenter -- Split into TTS-friendly segments (~150 words each)
    |
    v
[3] Character Analyzer (LLM) -- Detect speakers, emotions, assign voices
    |
    v
[4] Voice Assignment -- Link each character to a voice profile
    |
    v
[5] TTS Generation -- Qwen3-TTS generates audio per segment
    |                                         |
    |                              [5b] Whisper Validation (retry if WER > threshold)
    |
    v
[6] Audio Assembly -- Concatenate, crossfade, normalize (-16 LUFS)
    |
    v
[7] M4B Encoding -- Chapter markers, metadata, AAC compression
    |
    v
Finished Audiobook (.m4b)
```

---

## Voice Management

### Built-in Default Voices

The system comes with 7 pre-configured French-friendly voice profiles:

| Voice ID | Description |
|----------|-------------|
| `narrator_male` | Deep warm male, mature, authoritative |
| `narrator_female` | Soft warm female, clear and elegant |
| `young_male` | Young energetic male, bright |
| `young_female` | Young cheerful female, animated |
| `elder_male` | Older deep male, grave and wise |
| `elder_female` | Older compassionate female, gentle |
| `robotic` | Mechanical synthetic voice for sci-fi |

### Creating Custom Voices

**Option A: Voice Cloning** - Upload 3+ seconds of reference audio
```bash
# In the GUI: Voices tab -> Upload reference audio -> Assign name
```

**Option B: Voice Design** - Describe the voice in text
```bash
# In the GUI: Voices tab -> Describe voice -> Click Generate
# Example: "Deep male voice, gravelly, slightly British accent, menacing"
```

### Assigning Voices to Characters

After character analysis, the GUI shows each discovered character. For each:
- Select from existing voices
- Upload a reference sample for cloning
- Generate a new voice from description

---

## Emotion System

The LLM analyzes context to detect emotions in each segment:

| Emotion | French Instruction |
|---------|-------------------|
| calm | "Parlez d'un ton calme et posé" |
| excited | "Parlez avec excitation et enthousiasme" |
| angry | "Parlez avec colère, voix ferme et intense" |
| sad | "Parlez d'une voix triste et mélancolique" |
| whisper | "Chuchotez d'une voix mystérieuse" |
| tense | "Parlez d'une voix tendue et nerveuse" |
| urgent | "Parlez rapidement, avec urgence" |
| amused | "Parlez avec amusement, ton léger" |
| contemptuous | "Parlez avec mépris, voix froide" |
| surprised | "Parlez avec surprise et étonnement" |

---

## Configuration

Edit `~/.aiguibook/config.yaml`:

```yaml
tts:
  model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
  backend_local: true
  device: cuda
  dtype: bfloat16
  batch_size: 4

analysis:
  llm_backend: openrouter        # or "ollama"
  openrouter_api_key: ""         # Set via OPENROUTER_API_KEY env var
  openrouter_model: anthropic/claude-sonnet-4-20250514
  ollama_model: qwen3:32b
  ollama_base_url: http://localhost:11434

voices:
  narrator_ref: ""               # Path to narrator reference audio
  character_refs: {}             # {character_name: audio_path}

output:
  format: m4b
  bitrate: 128k
  sample_rate: 24000
  chapter_markers: true
  normalize_audio: true
  crossfade_duration: 0.5

validation:
  enabled: true
  whisper_model: distil-medium.en
  max_wer: 15
  max_retries: 2

general:
  language: french
  language_fallback: english
  max_segments: 99999
  preview_mode: false
```

---

## Models

AIGUIBook downloads these models automatically on first run (or use `python cli.py setup --download-models`):

| Model | Size | Purpose |
|-------|------|---------|
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | ~3.4GB | Main TTS with voice cloning |
| Qwen/Qwen3-TTS-12Hz-0.6B-VoiceDesign | ~1.2GB | Voice design from descriptions |
| faster-whisper (distil-medium.en) | ~300MB | Quality validation |

All models downloaded to `~/.cache/huggingface/`.

---

## Performance

On RTX 5080 (~16GB VRAM) with Qwen3-TTS-1.7B-Base:

- **TTS speed**: ~300% real-time (varies with text complexity)
- **VRAM usage**: ~4-6GB during generation
- **Typical book**: 10-hour audiobook in ~3-4 hours
- **Validation**: Adds ~20% overhead (depends on settings)

Speed tips:
- Increase `batch_size` in config (4-8 is good for 16GB VRAM)
- Disable validation if you trust the output (`validation.enabled: false`)
- Use `dtype: float16` instead of `bfloat16` for slight speedup
- Set `whisper_device: cpu` to free GPU VRAM if needed

---

## Troubleshooting

### "CUDA Out of Memory"
- Reduce `batch_size` to 2 or 1
- Set `dtype: float16` instead of `bfloat16`
- Close other GPU applications
- Use the 0.6B model variant

### "FFmpeg not found"
```bash
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS
```

### Character analysis not working
- Check your `OPENROUTER_API_KEY` environment variable
- Or switch to `llm_backend: ollama` with local Ollama running

### French pronunciation issues
- Use `language: french` in config
- The model auto-detects but explicitly setting it helps
- Word substitutions can be added in the GUI for problematic names

### Audio sounds robotic or flat
- Ensure emotion detection is enabled
- Try different reference audio for voice cloning
- Increase the segment overlap/crossfade duration

---

## Project Structure

```
audiobook-ai/
├── main.py                         # GUI entry point (python main.py)
├── cli.py                          # CLI entry point
├── requirements.txt                # All dependencies
├── pyproject.toml                  # Package metadata
├── audiobook_ai/
│   ├── __init__.py
│   ├── core/
│   │   ├── epub_parser.py          # EPUB2/3 parsing
│   │   ├── text_segmenter.py       # Text segmentation for TTS
│   │   ├── project.py              # Project state management
│   │   └── config.py               # Configuration management
│   ├── analysis/
│   │   └── character_analyzer.py   # LLM character/emotion detection
│   ├── tts/
│   │   ├── qwen_engine.py          # Qwen3-TTS wrapper
│   │   └── voice_manager.py        # Voice profiles & design
│   ├── audio/
│   │   ├── assembly.py             # Audio merging, M4B output
│   │   └── validation.py           # Whisper quality validation
│   └── gui/
│       └── app.py                  # Gradio web interface
├── voices/                         # Voice profile storage
├── output/                         # Generated audiobooks
└── work/                           # Temporary working files
```

---

## License

MIT License. Built for Roland's sci-fi audiobook collection.

---

## Future Ideas

- [ ] Audiobookshelf server integration for library management
- [ ] PDF/mobi/azw3 input support
- [ ] Background music and sound effects for sci-fi scenes
- [ ] Voice marketplace / community voice sharing
- [ ] Docker container for easy deployment
- [ ] Whisper model fine-tuning for French name pronunciation
- [ ] Multi-GPU support for faster generation
- [ ] Streaming/audio-while-generating playback
