# Edge Perception Quantisation Reliability

**MSc Data Science and Analytics — University of Hertfordshire**
**Student:** Faizan Shafi Seroo (23088135)
**Supervisor:** Dr. Grigorios Skaltsas
**Module:** 7COM1039

## Project Title
An Empirical Evaluation of Quantisation Effects on Reliability 
and Confidence Calibration of Real-Time Edge-Based Detection Systems

## Overview
This repository contains all experimental code and results for the 
MSc Final Project evaluating FP32, FP16, and INT8 quantisation 
effects on YOLOv8n confidence calibration via Intel OpenVINO 
on CPU-based edge hardware.

## Notebooks
- evaluation_day03.ipynb — FP32 baseline video inference
- confidence_analysis.ipynb — Confidence distribution analysis
- fp16_evaluation.ipynb — FP16 model export and evaluation
- degradation_pipeline.ipynb — Degradation dataset construction
- degradation_experiments.ipynb — 24-condition experiments
- ece_calibration.ipynb — ECE and reliability diagram analysis
- statistical_analysis.ipynb — Statistical significance testing

## Hardware
Intel Core i5 CPU laptop, Windows 11, no GPU acceleration

## Framework
Intel OpenVINO toolkit for FP16 and INT8 model export and inference

## Research Papers

- **Phase 1 — Confidence Calibration:** [`paper/Seroo_Skaltsas_Phase1_Calibration_2026.pdf`](paper/Seroo_Skaltsas_Phase1_Calibration_2026.pdf)
- **Phase 2 — Conformal Coverage:** Available on the [`conformal-yolov8n`](https://github.com/FaizanShafi454/edge-perception-quantisation-reliability/tree/conformal-yolov8n) branch
