from ultralytics import YOLO
from utils import create_directory
import os

# 데이터셋 설정 파일 경로
DATA_CONFIG_PATH = "../data/data_config.yaml"

# 결과 저장 디렉토리
MODEL_DIR = "../models/"
create_directory(MODEL_DIR)

# Fine-Tuning에 사용할 사전 학습된 YOLOv8 가중치 파일 경로
PRETRAINED_MODEL_PATH = "../models/yolov8_pretrained.pt"

# YOLOv8 모델 초기화
model = YOLO(PRETRAINED_MODEL_PATH)  # 사전 학습된 YOLOv8 가중치 로드

# Fine-Tuning (옵션 설명)
# - data: Fine-Tuning에 사용할 데이터셋 설정 파일 경로.
# - epochs: 학습 반복 횟수 (Fine-Tuning에서는 일반적으로 10~50 정도 추천).
# - optimizer: Adam 또는 SGD 등 최적화 알고리즘 선택.
# - lr0: 초기 학습률. Fine-Tuning에서는 작은 학습률(예: 1e-4)을 설정하여 안정적인 학습을 유도.
model.train(data=DATA_CONFIG_PATH, epochs=30, imgsz=640, optimizer='Adam', lr0=1e-4)

# Fine-Tuning된 모델 저장
fine_tuned_model_path = os.path.join(MODEL_DIR, "fine_tuned_model.pt")
model.save(fine_tuned_model_path)
print(f"Fine-Tuning 완료된 모델이 {fine_tuned_model_path}에 저장되었습니다.")
