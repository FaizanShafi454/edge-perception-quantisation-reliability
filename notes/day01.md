# Day 1

## Environment setup
- Conda environment: edge-perception-env
- python version:3.10

## Baseline verification (YOLO on image)
- Purpose: verify YOLO runs correctly on this system
- Model: YOLOv8n (pretrained)
- Input: sample image provided by Ultralytics (bus.jpg)
- Observed detections: persons, bus, stop sign
- Latency breakdown (approx):
  - preprocess: ~20 ms
  - inference: ~206 ms
  - postprocess: ~23 ms
- Conclusion: baseline inference successful on CPU