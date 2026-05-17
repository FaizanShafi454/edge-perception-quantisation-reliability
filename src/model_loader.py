from ultralytics import YOLO
from .config import YOLO_WEIGHTS

def load_fp32_model(weights_path: str = YOLO_WEIGHTS):
    return YOLO(weights_path)

def load_fp16_model(weights_path: str = YOLO_WEIGHTS):
    model = YOLO(weights_path)
    return model

def load_openvino_int8_model(openvino_model_path: str):
    return YOLO(openvino_model_path)
