from ultralytics import YOLO
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load a small YOLO model (fast for CPU)
model = YOLO("yolov8n.pt")  # 'n' = nano (small and fast)

# 2) Run on a built-in sample image URL (so you don't need files yet)
img_url = "https://ultralytics.com/images/bus.jpg"

results = model.predict(
    source=img_url,
    save=True,          # saves annotated image
    project=str(OUTPUT_DIR),
    name="day01_yolo_image",
    conf=0.25
)

print("Done.")
print("Saved outputs to:", OUTPUT_DIR / "day01_yolo_image")