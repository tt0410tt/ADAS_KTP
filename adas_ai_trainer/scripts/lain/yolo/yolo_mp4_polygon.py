from ultralytics import YOLO
import cv2
import numpy as np
import os

# YOLO 모델 불러오기 (YOLOv8-seg 가중치 파일 필요)
model = YOLO("/data_KTP/SO/runs/segment/train11/weights/222.pt")

# 비디오 파일 읽기
video_path = "/data_KTP/SO/분당판교_드라이브_2.mp4"
cap = cv2.VideoCapture(video_path)

# FPS 및 프레임 크기 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 저장 디렉토리 생성
output_dir = "output_masks"
os.makedirs(output_dir, exist_ok=True)

# 비디오 저장 설정
output_video_path = os.path.join(output_dir, "output_with_overlay2.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0  # 현재 처리 중인 프레임 인덱스

while True:
    ret, frame = cap.read()
    if not ret:
        print("더 이상 읽을 수 있는 프레임이 없습니다.")
        break

    # YOLO 추론
    results = model.predict(source=frame, save_txt=True)

    # 복호화된 폴리곤 데이터를 읽어서 오버레이
    label_path = os.path.join("/data_KTP/SO/runs/segment/predict/labels", f"image0.txt")
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        # 유효한 라인만 필터링
        valid_lines = [line for line in lines if len(line.strip().split()) >= 5]

        for line in valid_lines:
            data = line.strip().split()
            if len(data) > 1:
                # 폴리곤 좌표 (0~1로 정규화된 값)
                points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
                # 원본 이미지 크기로 복원
                points[:, 0] *= frame_width
                points[:, 1] *= frame_height
                points = points.astype(np.int32)
                # 원본 프레임에 복원된 폴리곤 오버레이 (초록색 단색)
                cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # 라벨 파일 내용 삭제
        with open(label_path, "w") as f:
            f.write("")

    # 처리된 프레임 저장
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"전체 영상 처리가 완료되었습니다. 결과 비디오는 {output_video_path}에 저장되었습니다.")
