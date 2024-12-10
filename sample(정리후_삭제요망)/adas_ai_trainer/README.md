# YOLOv8 Object Detection

## 프로젝트 설명
이 프로젝트는 YOLOv8을 사용하여 자동차 및 사람을 탐지하는 모델을 학습하고 평가합니다.

## 디렉토리 구조
- `data/`: 데이터셋 디렉토리
- `models/`: 모델 가중치 파일 저장
- `outputs/`: 예측 결과 및 평가 지표 저장
- `scripts/`: Python 스크립트
- `requirements.txt`: Python 의존성 목록

## 사용 방법
1. 필요한 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```
2. 모델 학습:
    ```bash
    python scripts/train.py
    ```
3. 모델 평가:
    ```bash
    python scripts/evaluate.py
    ```
