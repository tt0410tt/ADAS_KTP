import os
from pathlib import Path

def create_directory(path: str):
    """지정된 경로가 없으면 생성합니다."""
    os.makedirs(path, exist_ok=True)

def save_metrics(metrics: dict, file_path: str):
    """평가 지표를 JSON 형식으로 저장."""
    import json
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
