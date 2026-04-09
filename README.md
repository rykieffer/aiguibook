# AIGUIBook - AI Audiobook Generator

Transform EPUB ebooks into multi-voice audiobooks (M4A) with AI character voices, emotion detection, and voice acting.

## Architecture

```
EPUB -> Parse -> Segment -> LLM Analysis (characters/emotions) -> Voice Design -> Voice Clone + Acting -> M4A
```

### Two Production Modes

1. **Single Narrator** (Recommended): One voice reads everything. The AI dynamically applies emotions (angry, sad, whisper, tense) to act out different characters. Fast and consistent.

2. **Full Ensemble**: Each character gets a unique AI-designed voice via VoiceDesign. Characters are instantly recognizable by their distinct vocal identity.

### Key Technologies

- **TTS Engine**: [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — 10x faster Qwen3-TTS using CUDA graph capture (no flash-attn required)
- **Voice Design**: Qwen3-TTS VoiceDesign model creates unique voices from text descriptions
- **Voice Clone + Acting**: Qwen3-TTS Base model clones voices and applies `instruct`-based emotion acting
- **Analysis**: Local LLM (LM Studio/Ollama) or OpenRouter detects speakers and emotions per segment
- **Output**: M4A with loudness normalization (LUFS -16), AAC encoding

## Installation

```bash
git clone https://github.com/rykieffer/aiguibook.git
cd aiguibook

# Create environment
conda create -n aiguibook python=3.12
conda activate aiguibook

# System dependencies
sudo apt install ffmpeg sox libsox-dev -y

# Python dependencies
pip install -r requirements.txt
pip install faster-qwen3-tts

# RTX 50xx / Blackwell GPUs need nightly PyTorch:
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Usage

### GUI (Gradio)
```bash
python main.py
```

### CLI
```bash
python cli.py generate --input book.epub --output ./my_audiobook
```

## Workflow

1. **Tab 1 - Analysis**: Upload EPUB, set project folder, run character analysis. Everything saves to `analysis.json`.
2. **Tab 2 - Voice Design**: Design narrator voice + character voices. Voices auto-save to `project/voices/`.
3. **Tab 3 - Production**: Click Start. The Base model generates each segment with emotion acting, then assembles the final M4A.

### Project Folder Structure
```
~/audiobooks/my_book/
  analysis.json    # Contains ALL text + tags + voice descriptions
  voices/
    narrator.wav    # Designed or uploaded narrator voice
    John.wav        # Designed character voice
    Alice.wav       # Designed character voice
  segments/
    ch0_s000.wav    # Generated audio per segment
    ch0_s001.wav
    ...
  My_Book.m4a      # Final assembled audiobook
```

### Key Design Decisions

- **Text in JSON**: The full book text is embedded in `analysis.json`. After analysis, you never need to re-parse the EPUB.
- **Emotion via `instruct`**: Emotions are passed to the TTS model's `instruct` parameter (not prepended to text). This is the correct faster-qwen3-tts API.
- **Resume Support**: If generation is interrupted, click RESUME to skip already-generated segments.
- **Segmenter**: Splits at sentence boundaries only. Never breaks mid-sentence. Dialogue paragraphs kept together.

## License
MIT
