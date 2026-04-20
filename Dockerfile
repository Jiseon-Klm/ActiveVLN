# ============================================================
# ActiveVLN SocialACT Offline Evaluator — vllm API client
# Base: ROS2 Jazzy (Ubuntu 24.04)
# No GPU required in container — inference runs on vllm server
#
# Build:
#   docker build -t activevln-socialact-eval .
#
# Run (vllm server must be running on host before starting):
#   docker run --rm \
#     -v /home/aprl/Desktop/when2reason:/workspace/when2reason \
#     -e VLLM_API_BASE=http://host-gateway:8000/v1 \
#     activevln-socialact-eval
# ============================================================

FROM ros:jazzy-ros-base

# ── system packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ── isolated Python environment ───────────────────────────────
RUN python3 -m venv /opt/eval-venv

RUN /opt/eval-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/eval-venv/bin/pip install --no-cache-dir \
    "numpy>=1.26,<3" \
    "Pillow>=10.0" \
    "opencv-python-headless>=4.9" \
    "matplotlib>=3.8" \
    "qwen-vl-utils>=0.0.8" \
    "openai>=1.30"

# ── workspace ─────────────────────────────────────────────────
WORKDIR /workspace

COPY run_infer_socialact_by_activevln.py /workspace/run_infer_socialact_by_activevln.py

# ── default paths (override via -e at runtime) ────────────────
ENV DATASET_DIR=/workspace/when2reason/OmniNav_setup/SocialACT_revision/07_New_cube_and_panorama_images_2hz/Labeled
ENV INSTR_DIR=/workspace/when2reason/OmniNav_setup/SocialACT_revision/05_Refactored_data/High-level_Instructions/Labeled_v1
ENV OUTPUT_DIR=/workspace/when2reason/ActiveVLN_setup/experiment_results/ActiveVLN/SocialACT_High_Labeled_v1

# ── ROS2 env ──────────────────────────────────────────────────
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
