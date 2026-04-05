# AIGUIBook - AI Audiobook Generator

Transform EPUB ebooks into multi-voice audiobooks (M4B) with AI character voices, emotion detection, and voice cloning.

## Architecture

AIGUIBook uses a **two-stage TTS pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STAGE 1: Character Analysis                 │
│                                                                     │
│  EPUB → Parser → Segmenter → LLM (LM Studio / OpenRouter)           │
│              Detects: narrator vs dialogue, character names,         │
│              emotions per segment                                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 2: Reference Voice Generation             │
│                                                                     │
│  Bark TTS → Auto-generates WAV samples for each character           │
│  (10 built-in French voices, no recording needed)                   │
│                                                                     │
│  ┌─ User can override Bark samples with: ───────────────────────┐   │
│  │ • Upload custom WAV file (3+ seconds)                        │   │
│  │ • Use ElevenLabs / any other TTS as reference source         │   │
│  │ • Record your own voice                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: Audiobook Synthesis                     │
│                                                                     │
│  Qwen3-TTS → Voice cloning using Bark WAV samples                   │
│  → Generates full audiobook with character-appropriate voices       │
│  → Validates with Whisper STT                                       │
│  → Assembles chapters with crossfade + normalization                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                        Final M4B with chapter markers
```

## Features

- **EPUB Parsing**: Extracts chapters, metadata, TOC from EPUB2/EPUB3
- **Character Analysis**: LLM detects who is speaking in each text segment
- **Emotion Detection**: Calm, excited, angry, sad, whisper, tense, urgent, amused, etc.
- **Automatic Voice Assignment**: Maps characters to the best Bark voice preset
- **Bark Reference Generation**: Creates WAV voice samples automatically
- **Voice Cloning**: Qwen3-TTS clones voices from Bark reference audio
- **Manual Override**: Upload custom WAV files for any character
- **Quality Validation**: Whisper STT verifies generated audio accuracy
- **M4B Output**: Chapter markers, metadata, normalized audio, crossfade
- **Gradio GUI**: Web interface with 5 tabs (Setup, Voices, Preview, Generate, Settings)
- **CLI Interface**: Full pipeline from command line
- **Resumable Progress**: Save/analysis results to JSON, reuse without re-scanning

## Installation

```bash
git clone https://github.com/rykieffer/aiguibook.git
cd aiguibook

# Create conda environment
conda create -n aiguibook python=3.12
conda activate aiguibook

# Install dependencies
pip install -r requirements.txt

# Install Bark (if not auto-installed)
pip install git+https://github.com/suno-ai/bark.git

# Install Qwen3-TTS (for final synthesis)
pip install qwen-tts
```

### System Requirements
- **GPU**: NVIDIA with CUDA support (8GB+ VRAM recommended)
- **OS**: Linux (tested), Windows via WSL2
- **Python**: 3.10+
- **FFmpeg**: Required for audio assembly (`apt install ffmpeg`)

## Usage

### GUI Mode
```bash
python main.py
# Opens at http://localhost:7860
```

**Tab 1: Setup**
- Upload EPUB file
- Parse book metadata
- Set output format, language, TTS model
- Run character analysis

**Tab 2: Voices**
- View detected characters with segment counts
- Bark auto-generates reference WAV samples
- Upload custom WAV files to override Bark samples
- Voice design with text descriptions

**Tab 3: Preview**
- Test TTS with sample text
- Select voice, language, emotion
- Listen to generated audio

**Tab 4: Generate**
- Full audiobook generation
- Progress bar, chapter-by-chapter status
- Start/Pause/Resume/Stop
- Estimated time remaining

**Tab 5: Settings**
- TTS batch size, quality, device
- Validation settings (WER threshold, retries)
- LLM backend (LM Studio, OpenRouter, Ollama)
- Save/Load configuration

### CLI Mode
```bash
# Full generation
python cli.py generate --input book.epub --output ./audiobooks

# Parse and show metadata
python cli.py parse --input book.epub

# Character analysis only
python cli.py analyze --input book.epub

# Setup and model download
python cli.py setup --download-models

# Voice management
python cli.py voices --list
python cli.py voices --create narrator ref_audio.wav
python cli.py voices --create-design "marcus" "Deep male voice, French accent" "Sample text"
```

## Voice Options

### Bark Built-in Voices (10 French presets)
| Voice ID | Description | Use Case |
|----------|-------------|----------|
| narrator_male | Deep authoritative male | Main narrator |
| narrator_female | Warm female | Female narrator |
| young_male | Energetic young male | Young male characters |
| young_female | Cheerful young female | Young female characters |
| elder_male | Mature, warm male | Older male characters |
| elder_female | Grave, warm female | Older female characters |
| angry_male | Firm, intense male | Angry/controlling characters |
| soft_female | Gentle, soft female | Timid/calm characters |
| robotic | Neutral, mechanical | Robots, AI voices |

### Custom Voice Sources
1. **Bark auto-generation**: Default, no recording needed
2. **Upload WAV**: Any 3+ second WAV file
3. **ElevenLabs**: Export from ElevenLabs → upload as WAV reference
4. **Record yourself**: Use any recorder, export as WAV
5. **Edge TTS**: Generate samples with Microsoft Edge voices (free)

## Character Analysis Speed

| Model | Speed | Notes |
|-------|-------|-------|
| Ministral 3B (LM Studio) | ~14 min/1400 segs | Great balance |
| Qwen3.5-9B (LM Studio) | ~1-2 hours | Higher accuracy |
| OpenRouter (GPT-4o) | ~5-10 min | Fastest, costs tokens |

### Pre-filter Optimization
~70-80% of segments are pure narration and are skipped by the LLM entirely (detected by absence of dialogue markers). Only dialogue segments hit the LLM.

## Saving/Loading Analysis

After character analysis completes, results are saved automatically to:
```
/tmp/aiguibook/{book_title}/character_analysis.json
```

Reload this file later to skip re-analysis.

## Configuration

Config file location: `~/.aiguibook/config.yaml`

```yaml
tts:
  model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
  device: cuda
  dtype: bfloat16
  batch_size: 4

analysis:
  llm_backend: lmstudio
  lmstudio_base_url: http://localhost:1234/v1
  openrouter_api_key: ""  # or set OPENROUTER_API_KEY env var

output:
  format: m4b
  bitrate: 128k
  normalize_audio: true
  crossfade_duration: 0.5

validation:
  enabled: true
  whisper_model: distil-small.en
  max_wer: 15
  max_retries: 2
```

## Project Structure

```
audiobook-ai/
├── audiobook_ai/
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── character_analyzer.py   # LLM character/emotion analysis
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── assembly.py              # M4B assembly with chapter markers
│   │   └── validation.py            # Whisper STT validation
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py                # YAML configuration manager
│   │   ├── epub_parser.py           # EPUB2/EPUB3 parser
│   │   ├── project.py               # Project state management
│   │   └── text_segmenter.py        # Text segmentation for TTS
│   ├── gui/
│   │   ├── __init__.py
│   │   └── app.py                   # Gradio web interface
│   └── tts/
│       ├── __init__.py
│       ├── bark_engine.py           # Bark reference voice generator
│       └── voice_manager.py         # Voice profiles management
├── main.py                          # GUI entry point
├── cli.py                           # CLI entry point
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Package metadata
└── README.md                        # This file
```

## Dependencies

| Package | Purpose |
|---------|---------|
| bark | Reference voice generation |
| qwen-tts | Final audiobook TTS with voice cloning |
| torch | Deep learning backend |
| faster-whisper | Audio validation |
| gradio | Web GUI |
| ebooklib | EPUB parsing |
| pydub | Audio manipulation |
| ffmpeg-python | M4B assembly |
| jiwer | WER calculation |
| openai | LLM API (OpenRouter) |

## License

MIT License — see LICENSE file.

## Credits

- **Bark** - Suno AI (https://github.com/suno-ai/bark)
- **Qwen3-TTS** - Alibaba (https://huggingface.co/Qwen)
- **faster-whisper** - Guillaume Klein
- **Gradio** - Hugging Face
