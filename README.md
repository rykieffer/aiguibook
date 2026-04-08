# AIGUIBook v7 - AI Audiobook Generator

Transform EPUB ebooks into multi-voice audiobooks (M4B) with AI character voices, emotion detection, and voice acting.

## Architecture

AIGUIBook uses a tailored TTS pipeline focused on **emotional voice acting**:

```
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 1: Character Analysis                 │
│                                                             │
│ EPUB → Parser → Segmenter → LLM (LM Studio / OpenRouter)    │
│        Detects: narrator vs dialogue, character names,      │
│        and specific emotions per segment.                   │
│        *Results & Text are saved to a single JSON file.*    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 2: Voice Strategy                     │
│                                                             │
│ Single Narrator Mode (Recommended):                         │
│   Pick ONE base voice (e.g. Qwen built-in or custom WAV).   │
│   The AI applies the detected emotions (angry, sad,         │
│   whisper) dynamically to act out the different roles.      │
│                                                             │
│ Multi-Cast Mode (Ensemble):                                 │
│   Assign specific WAV reference files to specific           │
│   characters. Uses VoiceDesign to generate unique voices.   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 STAGE 3: Audiobook Synthesis                │
│                                                             │
│ Qwen3-TTS (1.7B) generates audio for each segment, applying │
│ the emotion instructions.                                   │
│ Files are assembled into an M4B with chapter markers.       │
└─────────────────────────────────────────────────────────────┘
```

## Features
- **EPUB Parsing & Segmentation**: Automatically splits chapters into TTS-safe lengths at sentence boundaries.
- **Fast Pre-filtering**: Skips LLM calls for 80% of segments that are purely narration, saving hours of GPU time.
- **LLM Character/Emotion Scan**: Local (LM Studio/Ollama) or remote (OpenRouter) LLM detects the speaker and their emotion.
- **Text-Embedded JSON**: Your parsed book text is saved directly inside the `character_analysis.json` file. You never have to re-upload the EPUB once it's scanned!
- **Dynamic Voice Acting**: Qwen3-TTS dynamically shifts the tone of the narrator's voice based on the context of the dialogue.
- **ElevenLabs Integration**: Automatically generates optimized Voice Design prompts for ElevenLabs if you wish to generate external reference voices.

## Installation

```bash
git clone https://github.com/rykieffer/aiguibook.git
cd aiguibook

# Create conda environment
conda create -n aiguibook python=3.11
conda activate aiguibook

# Install dependencies (FFmpeg required system-wide)
sudo apt install ffmpeg sox libsox-dev -y
pip install -r requirements.txt
pip install qwen-tts
```

## Usage

Launch the Gradio Web UI:
```bash
python main.py
```

### The 4-Step Workflow:
1. **Analysis Tab**: Upload your `.epub` and click "Run Character Analysis". When done, click "Save Analysis File".
2. **Voice Strategy Tab**: Choose "Single Narrator". Select "Ryan" or upload your own 3-second `.wav` file as the narrator. 
3. **Production Tab**: To render the book, just drop in your `.json` analysis file and click **START GENERATION**.
4. **Settings Tab**: Configure your LM Studio endpoint here.

### Working with JSON
Because the entire text of the book is now saved inside your `character_analysis.json` file, you can close the app, come back tomorrow, load *just the JSON file*, and immediately start rendering the audiobook.

## License
MIT License
