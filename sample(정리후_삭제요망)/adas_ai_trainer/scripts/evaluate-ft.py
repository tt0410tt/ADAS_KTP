from ultralytics import YOLO
from utils import create_directory, save_metrics
import os

# Fine-Tuned 모델 경로 및 데이터 설정
MODEL_PATH = "../models/fine_tuned_model.pt"
DATA_CONFIG_PATH = "../data/data_config.yaml"
OUTPUT_DIR = "../outputs/"
create_directory(OUTPUT_DIR)

# 모델 로드
model = YOLO(MODEL_PATH)

# 검증 및 평가
metrics = model.val(data=DATA_CONFIG_PATH)

# 성능 지표 저장
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
save_metrics(metrics, metrics_path)
print(f"평가 결과가 {metrics_path}에 저장되었습니다.")
