from ultralytics import YOLO
from utils import create_directory
import os

# 데이터셋 설정 파일 경로
DATA_CONFIG_PATH = "../data/data_config.yaml"

# 결과 저장 디렉토리
MODEL_DIR = "../models/"
create_directory(MODEL_DIR)

# YOLOv8 모델 초기화
model = YOLO("yolov8n.pt")  # Pretrained YOLOv8 모델 로드

# 학습
model.train(data=DATA_CONFIG_PATH, epochs=50, imgsz=640)

# 학습된 가중치 저장
model_path = os.path.join(MODEL_DIR, "trained_model.pt")
model.save(model_path)
print(f"모델이 {model_path}에 저장되었습니다.")
