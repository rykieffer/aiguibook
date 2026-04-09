# AIGUIBook v8 - AI Audiobook Generator

Transform EPUB ebooks into multi-voice audiobooks (M4A) with AI character voices, emotion detection, and voice acting.

## Project Folder Structure

Everything for one book lives in a single project folder:

```
~/audiobooks/my_book/
├── analysis.json          # Characters, emotions, and full book text
├── voices/
│   ├── narrator.wav        # Narrator reference voice
│   ├── Jean.wav           # Character voice (if ensemble mode)
│   └── Marie.wav          # Character voice
├── segments/
│   ├── ch0_seg0.wav        # Generated audio per text segment
│   ├── ch0_seg1.wav
│   ├── ch1_seg0.wav
│   └── ...
└── My_Book.m4a            # Final assembled audiobook
```

No more hunting for temp files. One folder = one book. Load it, resume it, share it.

## Workflow

1. **Tab 1 - Analysis**: Set your project folder, upload EPUB, run character analysis. Everything auto-saves.
2. **Tab 2 - Voice Design**: Design or upload the narrator voice. Optionally design character voices for ensemble mode.
3. **Tab 3 - Production**: Hit START. WAVs go to `segments/`, final M4A goes to the project root. If it crashes, hit RESUME.

### Resuming After a Crash
Just click "Load Project from Folder" in Tab 1, paste the same folder path, then hit RESUME in Tab 3. It skips already-generated WAVs automatically.

## Features
- **Single Narrator Mode**: One voice acts out all roles with emotion
- **Full Ensemble Mode**: Different AI-designed voices per character
- **Configurable Silence**: 0-2 seconds between segments (default 0.75s)
- **Resume Support**: Skip existing WAVs after a crash
- **Automatic M4A Assembly**: Chapter markers, AAC encoding, loudness normalization
- **Text-Embedded JSON**: analysis.json contains the full book text - no need to re-parse EPUB

## Installation

```bash
git clone https://github.com/rykieffer/aiguibook.git
cd aiguibook

conda create -n aiguibook python=3.12
conda activate aiguibook

sudo apt install ffmpeg sox libsox-dev -y
pip install -r requirements.txt
pip install faster-qwen3-tts
```

## Usage

```bash
python main.py
```

## License
MIT
