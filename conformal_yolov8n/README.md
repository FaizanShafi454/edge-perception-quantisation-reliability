# Edge Perception Quantisation Reliability

**MSc Data Science and Analytics — University of Hertfordshire**
**Student:** Faizan Shafi Seroo (23088135)
**Supervisor:** Dr. Grigorios Skaltsas
**Module:** 7COM1039

---

## Repository Structure

| Branch | Phase | Description |
|--------|-------|-------------|
| `main` | Phase 1 | Confidence calibration — ECE, reliability diagrams, confidence convergence |
| `conformal-yolov8n` | Phase 2 | Conformal prediction coverage — split CP, 24-condition evaluation |

---

## Phase 1 — main branch

**Title:** An Empirical Evaluation of Quantisation Effects on Reliability and
Confidence Calibration of Real-Time Edge-Based Detection Systems

Key finding: Discovered the **confidence convergence phenomenon** — a 0.382
FP32–INT8 confidence gap on clean data collapses to under 0.034 under any degradation.

## Phase 2 — this branch (conformal-yolov8n)

**Title:** Conformal Coverage Under Quantisation: A Case Study on Edge-Deployed YOLOv8

Key finding: FP16 and INT8 break the nominal 90% coverage guarantee on clean
COCO val2017 data and violate coverage in 6/8 and 3/8 degradation conditions
vs 1/8 for FP32.

---

## Hardware

Intel Core i5 CPU laptop, Windows 11, no GPU acceleration

## Framework

Intel OpenVINO, Ultralytics YOLOv8, Python 3.10