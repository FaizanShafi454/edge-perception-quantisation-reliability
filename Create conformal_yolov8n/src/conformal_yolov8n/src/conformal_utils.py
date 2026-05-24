from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def load_yolo_labels(label_path, image_shape):
    """
    Read YOLO txt labels and convert to xyxy pixel boxes.
    YOLO format: class_id x_center y_center width height (normalized).
    """
    h, w = image_shape[:2]
    boxes = []

    if not Path(label_path).exists():
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, xc, yc, bw, bh = map(float, parts)
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h

        boxes.append({
            "class_id": int(cls_id),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

    return boxes


def compute_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


def collect_calibration_data(model, image_paths, label_paths, iou_threshold=0.5, conf_threshold=0.001, imgsz=640):
    rows = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        gt_boxes = load_yolo_labels(lbl_path, image.shape)

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_threshold,
            verbose=False,
            device="cpu"
        )

        if not results or len(results) == 0:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        pred_conf = result.boxes.conf.cpu().numpy()

        used_pred = set()

        for gt in gt_boxes:
            gt_box = [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]
            gt_class = gt["class_id"]

            best_iou = 0.0
            best_idx = -1

            for i, pbox in enumerate(pred_boxes):
                if i in used_pred:
                    continue
                iou = compute_iou(gt_box, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= iou_threshold:
                used_pred.add(best_idx)
                rows.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "true_class": gt_class,
                    "pred_class": int(pred_cls[best_idx]),
                    "confidence": float(pred_conf[best_idx]),
                    "iou": float(best_iou),
                    "matched": 1
                })
            else:
                rows.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "true_class": gt_class,
                    "pred_class": -1,
                    "confidence": 0.0,
                    "iou": 0.0,
                    "matched": 0
                })

    return pd.DataFrame(rows)


def compute_nonconformity(df):
    df = df.copy()

    df["nonconformity"] = np.where(
        (df["matched"] == 1) & (df["pred_class"] == df["true_class"]),
        1.0 - df["confidence"],
        1.0
    )

    return df


def conformal_quantile(scores, alpha=0.1):
    scores = np.asarray(scores)
    n = len(scores)

    if n == 0:
        raise ValueError("No calibration scores found.")

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    qhat = np.quantile(scores, q_level, method="higher")
    return qhat, q_level, n


def conformal_accept(confidence, qhat):
    return confidence >= (1.0 - qhat)


def evaluate_on_test(
    model,
    img_test,
    lbl_test,
    iou_threshold=0.5,
    conf_threshold=0.001,
    imgsz=640,
    qhat=None
):
    rows = []

    for img_path, lbl_path in zip(img_test, lbl_test):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        gt_boxes = load_yolo_labels(lbl_path, image.shape)

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_threshold,
            verbose=False,
            device="cpu"
        )

        if not results or len(results) == 0:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        pred_boxes = result.boxes.xyxy.cpu().numpy()
        pred_cls = result.boxes.cls.cpu().numpy().astype(int)
        pred_conf = result.boxes.conf.cpu().numpy()

        used_pred = set()

        for gt in gt_boxes:
            gt_box = [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]
            gt_class = gt["class_id"]

            best_iou = 0.0
            best_idx = -1

            for i, pbox in enumerate(pred_boxes):
                if i in used_pred:
                    continue
                iou = compute_iou(gt_box, pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= iou_threshold:
                used_pred.add(best_idx)
                conf = float(pred_conf[best_idx])
                pred_class = int(pred_cls[best_idx])

                accepted = True if qhat is None else conformal_accept(conf, qhat)
                covered = int(accepted and (pred_class == gt_class))

                rows.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "true_class": gt_class,
                    "pred_class": pred_class,
                    "confidence": conf,
                    "iou": float(best_iou),
                    "matched": 1,
                    "accepted": accepted,
                    "covered": covered
                })
            else:
                conf = 0.0
                accepted = True if qhat is None else conformal_accept(conf, qhat)
                covered = 0

                rows.append({
                    "image_path": str(img_path),
                    "label_path": str(lbl_path),
                    "true_class": gt_class,
                    "pred_class": -1,
                    "confidence": conf,
                    "iou": 0.0,
                    "matched": 0,
                    "accepted": accepted,
                    "covered": covered
                })

    return pd.DataFrame(rows)
