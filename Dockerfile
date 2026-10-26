# Inference image: the official Codabench runtime (Python 3.11, PyTorch 2.4.1, CUDA 12.1) plus this
# repository's offline runner. The submitted archive is mounted at run time, unmodified.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /opt/upar
# pycocotools is only needed for inference/evaluate_keypoints.py (the submission itself does not use it).
RUN pip install --no-cache-dir pycocotools==2.0.8
COPY inference/ /opt/upar/inference/
ENTRYPOINT ["python", "/opt/upar/inference/run_codabench_layout.py"]
CMD ["--submission", "/submission", "--input", "/input_data", "--output", "/output/predictions.json"]
