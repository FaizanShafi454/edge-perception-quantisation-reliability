# Edge Perception Quantisation Reliability

**MSc Data Science and Analytics — University of Hertfordshire**  
**Student:** Faizan Shafi Seroo (23088135)  
**Supervisor:** Dr. Grigorios Skaltsas  

---

## Repository Structure

This repository contains two research phases:

| Branch | Phase | Description |
|--------|-------|-------------|
| `main` | Phase 1 | Confidence calibration study — ECE, reliability diagrams, confidence convergence |
| `conformal-yolov8n` | Phase 2 | Conformal prediction coverage study — split CP, 24-condition evaluation |

---

## Phase 1 — main branch

**Title:** An Empirical Evaluation of Quantisation Effects on Reliability and
Confidence Calibration of Real-Time Edge-Based Detection Systems

Key finding: Discovered the **confidence convergence phenomenon** — a 0.382
FP32–INT8 confidence gap on clean data collapses to <0.034 under any degradation.

## Phase 2 — conformal-yolov8n branch

**Title:** Conformal Coverage Under Quantisation: A Case Study on Edge-Deployed YOLOv8

Key finding: FP16 and INT8 break the nominal 90% coverage guarantee on clean
COCO val2017 data and violate coverage in 6/8 and 3/8 degradation conditions
vs 1/8 for FP32.

---

## Hardware

Intel Core i5 CPU laptop, Windows 11, no GPU acceleration

## Framework

Intel OpenVINO, Ultralytics YOLOv8, Python 3.10

## How to Run Phase 2 Experiments

### 1. Create and activate environment

```bash
conda create -n edge-conf python=3.10 -y
conda activate edge-conf
pip install -r requirements.txt
```

### 2. Prepare models

Place the exported models in the repository root (or update paths in the notebook):

- `yolov8n.onnx`               – FP32 ONNX export from Ultralytics
- `yolov8n_openvino_model/`    – FP32 OpenVINO IR
- `yolov8n_int8_openvino_model/` – INT8 OpenVINO IR (post‑training quantisation)

### 3. Run Phase 2 notebook

Open Jupyter:

```bash
jupyter notebook
```

Then run:

- `conformal_yolov8n/Phase2_Conformal_Coverage.ipynb`

This notebook:

1. Builds COCO128 calibration and COCO val2017 test splits.
2. Computes LAC nonconformity scores and conformal thresholds for FP32, FP16, and INT8.
3. Evaluates coverage on clean test data and 7 degradation conditions (blur, JPEG, noise).
4. Writes all tables to `conformal_yolov8n/results/tables/` (including `clean_test_summary.csv` and `full_24_condition_results.csv`).[file:113][file:114]
