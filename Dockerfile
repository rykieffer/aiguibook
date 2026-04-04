FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Roland"
LABEL description="AIGUIBook - AI Audiobook Generator with Character Voices"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    python3-pip ffmpeg git curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash aiguibook
USER aiguibook
WORKDIR /home/aiguibook/app

# Create venv
RUN python3.12 -m venv /home/aiguibook/venv
ENV PATH="/home/aiguibook/venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/aiguibook/venv"

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir qwen-tts

# Optional: flash attention (skip if build fails)
RUN pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn skipped"

# Copy application
COPY --chown=aiguibook:aiguibook . .

# Create directories
RUN mkdir -p /home/aiguibook/.aiguibook
RUN mkdir -p /home/aiguibook/app/{voices,output,work}

# Volumes for persistence
VOLUME ["/home/aiguibook/.aiguibook", "/home/aiguibook/app/voices", "/home/aiguibook/app/output"]

# Default port
EXPOSE 7860

# Default command: launch GUI
CMD ["python3.12", "main.py", "--port", "7860", "--server-name", "0.0.0.0"]
