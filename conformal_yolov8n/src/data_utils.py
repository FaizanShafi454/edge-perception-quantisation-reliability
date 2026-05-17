from pathlib import Path
import cv2

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
