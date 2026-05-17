# Conformal YOLOv8n Reliability

This project evaluates FP32, FP16, and INT8 OpenVINO versions of YOLOv8n under a conformal prediction wrapper.

## Goals
- Compare prediction coverage across precision levels.
- Measure inference time on CPU.
- Evaluate calibration behavior on COCO128 and video.

## Structure
- `src/`: code modules
- `data/`: dataset and video input
- `models/`: exported models
- `results/`: outputs and tables
- `notebooks/`: experiments
