# Training image (not needed for inference). The original training ran with PyTorch 2.7.0 + CUDA 12.8
# on a single NVIDIA RTX PRO 6000 (Blackwell); see requirements/training-local.txt for the observed stack.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
WORKDIR /workspace/upar
RUN pip install --no-cache-dir numpy==1.26.4 Pillow==10.2.0 pycocotools==2.0.10 mmcv==1.7.2 \
        opencv-python-headless==4.10.0.84 addict==2.4.0 yapf==0.40.2 PyYAML==6.0.2 tomli==2.2.1
COPY src/ src/
COPY scripts/ scripts/
COPY artifacts/ artifacts/
COPY peta_file_mapping.txt .
ENV PYTHONPATH=/workspace/upar/src:/workspace/upar/scripts
# Mount the organizer data at /workspace/upar/data (see TRAINING.md for the expected layout).
CMD ["python", "scripts/train_rsn_domain_continuation.py", "--help"]
