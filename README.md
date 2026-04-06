# AIGUIBook - AI Audiobook Generator

Transform EPUB ebooks into professional M4B audiobooks using AI character analysis and Qwen3-TTS voice cloning.

**Project:** [GitHub - rykieffer/aiguibook](https://github.com/rykieffer/aiguibook)

## Project Vision
An end-to-end pipeline that:
1.  **Analyzes** the book using an LLM (LM Studio/OpenRouter) to detect characters, emotions, and speaking roles.
2.  **Assigns Voices** to characters. (Single Narrator mode: one voice with dynamic emotional range; Multi-Cast mode: different voices per character).
3.  **Generates Audio** using **Qwen3-TTS** (Voice Cloning). It uses 3-second reference audio clips to create unique voices for each character and uses the detected emotions to modulate the speech style.
4.  **Validates** generated audio quality using **Whisper** and Word Error Rate (WER).
5.  **Assembles** the final M4B with chapter markers, metadata, and loudness normalization.

## Architecture

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Text Analysis** | LM Studio / OpenRouter | LLM-based character detection, emotion tagging (e.g., "Holden is tense"). |
| **Voice Engine** | Qwen3-TTS + Transformers | High-quality voice synthesis using reference audio for voice cloning. |
| **Voice Reference** | Bark / Custom WAV | Generates or stores 3s reference clips for each character. |
| **Validation** | Faster-Whisper | Ensures generated audio matches the text (WER check). |
| **Assembly** | FFmpeg | Merges chunks into a single M4B with chapter markers. |

## Setup

### Prerequisites
*   **Python:** 3.10+
*   **GPU:** NVIDIA RTX (8GB+ VRAM recommended)

### Installation
```bash
pip install -r requirements.txt
pip install qwen-tts
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 # For RTX 50 Series
```

### System Dependencies
```bash
sudo apt install sox ffmpeg libsox-dev -y
```

### Usage
```bash
# GUI Mode
python main.py

# CLI Mode
python cli.py generate --input book.epub
```
