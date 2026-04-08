# AIGUIBook - Project Context & Architecture
**Repository:** https://github.com/rykieffer/aiguibook
**Target Hardware:** Linux/WSL2, NVIDIA RTX 5080 (16GB VRAM)

---

## 1. Core Strategy
The software is a complete AI-powered audiobook production pipeline.
**Current Workflow:**
1.  **Ingest:** Parse EPUB to chapters/text.
2.  **Analyze:** Use LLM (LM Studio/Ollama) to detect Characters & Emotions for every text segment.
3.  **Design:** Use **Qwen3-TTS (VoiceDesign model)** to generate unique Reference WAV files for the Narrator and every Character based on text descriptions.
4.  **Production:** Use **Qwen3-TTS (Base model)** + `faster-qwen3-tts` (CUDA graphs) to clone the Reference WAVs while synthesizing the text with the detected Emotions.

## 2. Module Architecture

### `core/config.py`
- Loads `~/.aiguibook/config.yaml`.
- Manages API keys (OpenRouter), LLM endpoints (LM Studio), and TTS settings (Device, Batch, Model variants).

### `core/epub_parser.py`
- **Function:** Extracts OPF metadata (title, author) and spine items from EPUB.
- **Status:** Working perfectly. Do not modify.

### `core/text_segmenter.py`
- **Function:** Splits raw chapter text into manageable chunks (~150 words) for the TTS engine. Respects sentence boundaries.
- **Status:** Working perfectly. Do not modify.

### `analysis/character_analyzer.py`
- **Function:**
  - Receives text chunks.
  - Sends them to a local LLM (LM Studio) to identify `Speaker`, `Character Name`, and `Emotion`.
  - **Deduplication:** Merges name variants (e.g., "Naomi" and "Naomi Nagata").
- **Input:** List of `TextSegment` objects.
- **Output:** JSON structure (saved to disk) mapping Segment IDs to `SpeechTag`s.
- **Status:** **CRITICAL STABLE DO NOT MODIFY.**

### `tts/qwen_engine.py`
- **Core Engine:** Handles model loading and inference.
- **Libraries:** Supports `faster-qwen3-tts` (Primary, uses CUDA Graphs for speed) and falls back to `qwen-tts` if needed.
- **Modes:**
  - **VoiceDesign:** `model.generate_voice_design(text, instruct, language)` -> Creates Reference WAVs.
  - **VoiceClone:** `model.generate_voice_clone(text, ref_audio, ref_text, language)` -> Synthesizes book audio using the Reference WAV.
  - **Emotion:** Injects emotion instructions (e.g., "Parlez avec colère") into the prompt for generation.
- **Pooling:** Uses `TTSEnginePool` for multi-threaded generation.

### `gui/app.py`
- **Stack:** Gradio v6+.
- **Tabs:**
  1.  **Analysis:** File input for EPUB. Parses and runs LLM scan. Loads/Saves analysis JSON.
  2.  **Voice Design:** UI for Narrator and Character descriptions. Trigger "Design Voice" (creates WAV).
  3.  **Production:** Runs the full generation loop using the WAVs from Tab 2.

## 3. Detailed Pipeline Logic

### Step 1: Parsing & Analysis (Immutable)
- User uploads EPUB.
- `TextSegmenter` cuts text into ~400 segments.
- `CharacterAnalyzer` loops over segments via LLM.
- Result is a JSON file: `{"tags": {segment_id: {"char": "Barry", "emotion": "angry"}}, "chars": ["Barry", "Joan"]}`.

### Step 2: Voice Design (Reference Generation)
- User types a description for "Barry" (e.g., "Deep male French voice").
- Clicking "Design" loads **Qwen3-TTS-12Hz-1.7B-VoiceDesign** model.
- Generates a 5-second WAV file.
- **Storage:** Saves WAV path in `self.character_voice_paths["Barry"]`.
- Same for Narrator (`self.narrator_wav_path`).

### Step 3: Production (Voice Cloning)
- User switches to "Production" tab.
- Loads **Qwen3-TTS-12Hz-1.7B-Base** via `faster-qwen3-tts` (FasterQwen3TTS).
- **Loop:**
  1.  Read segment text.
  2.  Identify Character (e.g., "Barry").
  3.  Lookup WAV for Barry.
  4.  Identify Emotion (e.g., "Angry").
  5.  Construct Prompt: `[Emotion Instruction]. [Segment Text]`.
  6.  Call `generate_voice_clone(text=prompt, ref_audio=Barry_WAV, ref_text=Barry_Context)`.
  7.  Save WAV to output dir.
- **Assembly:** Stitch WAVs together (not fully implemented yet).

## 4. Critical Rules for New Session
1.  **ALWAYS** run `git pull` and check `PROJECT_CONTEXT.md` before starting.
2.  **NEVER** touch `character_analyzer.py`, `epub_parser.py`, or `text_segmenter.py`.
3.  **PRIORITY:** Use `faster-qwen3-tts` for all voice generation tasks.
4.  **GOAL:** The ultimate goal is a "Single Narrator" mode where the narrator reads everything but changes tone based on emotion tags.
