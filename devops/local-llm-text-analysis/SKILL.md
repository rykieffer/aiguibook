---
name: local-llm-text-analysis
description: Strategy for processing long documents with local LLMs (LM Studio/Ollama). Focuses on single-segment prompting, regex pre-filtering, and rule-based post-processing to avoid context limits and empty responses.
category: devops
---

# Local LLM Text Analysis Strategy

When using local LLMs (e.g., Qwen3.5-9B on LM Studio) to analyze long documents (like EPUBs) for tasks like character extraction or sentiment analysis, standard batching often fails due to context limits and model instability. Use this architecture for reliability and speed.

## 1. Single-Segment Prompting
- **Do not batch** multiple text segments into a single prompt. Local models often return empty responses or truncated JSON when the input context is too large.
- **Rule:** Send one text segment per API call.
- **Prompt Template:** Keep it concise (
</think>
