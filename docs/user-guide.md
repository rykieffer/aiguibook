# AIGUIBook User Guide for AIGUIBook (v6)

## Overview
AIGUIBook is a Gradio web interface that transforms your EPUB books into audiobooks. The system uses an LLM to analyze the text for character voices and emotions, and then uses Qwen3-TTS to generate the audio.

## Prerequisites (Before you start)
1.  **EPUB File:** You must have the EPUB file of your book.
    *   *Important:* Even if you load an analysis file later, the app needs the EPUB to generate the audio.
2.  **Analysis File (`.json`):** 
    *   If you have run an analysis previously, you will have a `.json` file (e.g., `aiguibook_analysis.json`).
    *   If you have never analyzed this book, you will need to run the **Analysis** step first.
3.  **Qwen-TTS:** Ensure you have run `pip install qwen-tts` and have a GPU with PyTorch installed (RTX 50 Series users must use the nightly build).

## Getting Started
1.  Open your terminal and navigate to the project folder:
    `cd /path/to/aiguibook`
2.  Launch the interface:
    `python main.py`
3.  Open your browser to `http://localhost:7860`.

## The 4-Tab Workflow

### Tab 1: Setup & Analysis
This tab is for **understanding** your book.
*   **Upload EPUB:** Drag and drop your `.epub` file.
*   **Parse Book:** Click this to extract the chapters and author.
*   **Run Character Analysis:** This runs the LLM to find out *who* is speaking and *how* (emotions). This takes 10-20 minutes.
*   **Save/Load Analysis:** 
    *   **Save:** Saves the analysis to a `.json` file.
    *   **Load:** Loads a saved `.json` file so you don't have to re-analyze.
    *   *Note:* Loading a JSON file loads the **emotions/characters** but **not the book text**.

### Tab 2: Voices & Casting
This tab is for **assigning voices**.
*   **Single Narrator Mode (Default):** One voice for everyone, but they will act out the emotions (e.g., whispering, angry) based on the analysis.
    *   **Select Voice:** Pick a base voice (Ryan, Aiden, etc.).
*   **Ensemble Cast Mode:** Unique voices for each character.
    *   **Auto-Assign:** Automatically assigns the best voice based on the analysis.
*   **Preview Narrator:** Listen to a sample of the voice before you start.

### Tab 3: Production
This tab is for **generating** the book.
*   **Book Source (EPUB):** 
    *   **CRITICAL STEP:** You must upload the EPUB file here **again**.
    *   *Why?* If you loaded a JSON file in Tab 1, the system has the emotions but forgot the text. This field gives it the text.
*   **Preview Mode:** If checked, it will only generate the first chapter (for testing).
*   **Start Generation:** Clicking this begins the process. It will show the progress in the log box.
    *   *First Run:* It may take several minutes to download the TTS model.

### Tab 4: Settings
*   **LLM Settings:** 
    *   **LM Studio URL:** Where the app looks for the local LLM (default: `http://localhost:1234/v1`).
    *   **Test Connection:** Checks if LM Studio is running.

## Troubleshooting

### Error: "No EPUB path available"
**Cause:** You loaded a JSON analysis file but didn't provide the book text.
**Fix:** Go to **Tab 3 (Production)** and upload your EPUB file in the "Book Source" box.

### Error: "'Qwen3TTSModel' object has no attribute 'generate'"
**Cause:** This is a library compatibility issue.
**Fix:** Ensure you are using the latest version of the AIGUIBook code (v6+) which has the patch to use `transformers.AutoModel`.

### Error: "CUDA capability sm_120 is not compatible"
**Cause:** You have an RTX 5080/5090 and an old version of PyTorch.
**Fix:** 
```bash
pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Error: "TTS generation failed: 'NoneType' object has no attribute 'len'"
**Cause:** The text-to-speech engine is receiving a blank text field.
**Fix:** Check your EPUB file for empty chapters or very short chapters that might be skipped by the segmenter.
