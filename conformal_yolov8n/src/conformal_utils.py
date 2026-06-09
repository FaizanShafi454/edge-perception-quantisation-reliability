from pathlib import Path
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. LABEL LOADING
# ─────────────────────────────────────────────

def load_yolo_labels(label_path, image_width, image_height):
    """
    Load a YOLO-format label file and return ground-truth boxes in xyxy pixels.

    YOLO format per line:  class_id  x_center  y_center  width  height
    All values are normalised [0, 1] and are converted to absolute pixel coords.

    Returns
    -------
    list of dict  ->  {"class_id": int, "bbox": [x1, y1, x2, y2]}
                      Empty list if the file does not exist.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_c, y_c, w, h = map(float, parts)
            x_c *= image_width
            y_c *= image_height
            w   *= image_width
            h   *= image_height
            boxes.append({
                "class_id": int(class_id),
                "bbox": [x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2]
            })
    return boxes


# ─────────────────────────────────────────────
# 2. IoU HELPER
# ─────────────────────────────────────────────

def compute_iou(box_a, box_b):
    """
    Compute Intersection-over-Union for two boxes in xyxy format.
    Returns a float in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# 3. NONCONFORMITY SCORE
# ─────────────────────────────────────────────

def compute_nonconformity(confidence):
    """
    LAC (Least Ambiguous Classifier) nonconformity score.

    Formula:  s = 1 - confidence

    Only used for matched detections (IoU >= iou_threshold). False positives
    are excluded during calibration/test evaluation.
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return 1.0 - confidence


# ─────────────────────────────────────────────
# 4. CONFORMAL THRESHOLD (q_hat)
# ─────────────────────────────────────────────

def conformal_quantile(scores, alpha=0.10):
    """
    Compute the conformal threshold q_hat from calibration nonconformity scores.

    Finite-sample corrected quantile level:
        level = ceil( (n + 1) * (1 - alpha) ) / n
    """
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot compute quantile: scores array is empty.")

    n     = len(scores)
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


# ─────────────────────────────────────────────
# 5. CONFORMAL ACCEPT / REJECT
# ─────────────────────────────────────────────

def conformal_accept(nonconformity_score, qhat):
    """
    Accept a detection if its nonconformity score is within the threshold.
    """
    return float(nonconformity_score) <= float(qhat)


# ─────────────────────────────────────────────
# 6. CALIBRATION DATA COLLECTION
# ─────────────────────────────────────────────

def collect_calibration_data(model, image_paths, label_dir, iou_threshold=0.50):
    """
    Run the YOLO model on a list of calibration images and collect nonconformity scores.

    Only MATCHED detections (best_iou >= iou_threshold) are included.
    False positives (unmatched detections) are excluded.

    Parameters
    ----------
    model         : Ultralytics YOLO model (FP32, FP16, or INT8 via OpenVINO)
    image_paths   : list of Path objects  — the CALIBRATION split only
    label_dir     : Path to the directory containing YOLO .txt label files
    iou_threshold : float  Minimum IoU to count a detection as a true positive match

    Returns
    -------
    pd.DataFrame with columns:
        image, class_id, confidence, best_iou, nonconformity
    """
    label_dir = Path(label_dir)
    rows = []

    for image_path in image_paths:
        results = model(str(image_path), verbose=False)
        if not results:
            continue

        result  = results[0]
        orig_h, orig_w = result.orig_shape
        gt_boxes = load_yolo_labels(label_dir / f"{image_path.stem}.txt", orig_w, orig_h)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy  = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        matched_gt = set()

        for box, conf, cls in zip(xyxy, confs, clss):
            best_iou    = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if int(cls) != gt["class_id"]:
                    continue
                iou = compute_iou(box.tolist(), gt["bbox"])
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                rows.append({
                    "image":          image_path.name,
                    "class_id":       int(cls),
                    "confidence":     float(conf),
                    "best_iou":       round(best_iou, 6),
                    "nonconformity":  round(compute_nonconformity(conf), 6)
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 7. TEST SET EVALUATION
# ─────────────────────────────────────────────

def evaluate_on_test(model, image_paths, label_dir, qhat, iou_threshold=0.50):
    """
    Run the YOLO model on test images and measure conformal coverage.

    Parameters
    ----------
    model         : Ultralytics YOLO model
    image_paths   : list of Path objects  — TEST split only
    label_dir     : Path to label directory
    qhat          : float  Conformal threshold from calibration set
    iou_threshold : float  Same as in collect_calibration_data

    Returns
    -------
    pd.DataFrame with columns:
        image, class_id, confidence, best_iou, nonconformity, accepted
    """
    label_dir = Path(label_dir)
    rows = []

    for image_path in image_paths:
        results = model(str(image_path), verbose=False)
        if not results:
            continue

        result  = results[0]
        orig_h, orig_w = result.orig_shape
        gt_boxes = load_yolo_labels(label_dir / f"{image_path.stem}.txt", orig_w, orig_h)

        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy  = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        matched_gt = set()

        for box, conf, cls in zip(xyxy, confs, clss):
            best_iou    = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                if int(cls) != gt["class_id"]:
                    continue
                iou = compute_iou(box.tolist(), gt["bbox"])
                if iou > best_iou:
                    best_iou    = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None and best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                score    = round(compute_nonconformity(conf), 6)
                accepted = conformal_accept(score, qhat)
                rows.append({
                    "image":         image_path.name,
                    "class_id":      int(cls),
                    "confidence":    float(conf),
                    "best_iou":      round(best_iou, 6),
                    "nonconformity": score,
                    "accepted":      accepted
                })

    return pd.DataFrame(rows)