import os
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_2 = os.path.dirname(os.path.dirname(current_dir))
parent_dir_3 = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
model_path = os.path.abspath(os.path.join(parent_dir_2, "model/lain/train_yolo11xseg_SO.pt"))
video_path = os.path.join(parent_dir_3, "video/4_mv_1.mp4")
output_video_path = os.path.join(parent_dir_3, "video/4_mv_1test.mp4")
label_path = os.path.join(parent_dir_3, "runs/segment/predict/labels/image0.txt")

def process_line(line, frame_width, frame_height):
    data = line.strip().split()
    if len(data) < 2:
        return None, None
    class_id = int(data[0])
    points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= frame_width
    points[:, 1] *= frame_height
    return class_id, points.astype(np.int32)

def fit_line(points):
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    reg = LinearRegression().fit(x, y)
    return reg.coef_[0], reg.intercept_

def is_polygon_in_roi(polygon, roi, threshold=0.8):
    inside_count = 0
    for point in polygon:
        if cv2.pointPolygonTest(roi, (int(point[0]), int(point[1])), False) >= 0:
            inside_count += 1
    return inside_count / len(polygon) >= threshold

# YOLO 모델 로드
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

roi_left_ratio = np.array([
    [0.2, 1.0],
    [0.5, 1.0],
    [0.5, 0.75],
    [0.45, 0.75]
], dtype=np.float32)

roi_right_ratio = np.array([
    [0.5, 1.0],
    [0.8, 1.0],
    [0.55, 0.75],
    [0.5, 0.75]
], dtype=np.float32)

roi_left = (roi_left_ratio * np.array([frame_width, frame_height])).astype(np.int32)
roi_right = (roi_right_ratio * np.array([frame_width, frame_height])).astype(np.int32)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

total_frames_to_process = int(fps * 10)
current_frame = 0

while current_frame < total_frames_to_process:
    ret, frame = cap.read()
    if not ret:
        print("더 이상 읽을 수 있는 프레임이 없습니다.")
        break

    results = model.predict(source=frame, save_txt=True)

    left_polygons = []
    right_polygons = []

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            class_id, points = process_line(line, frame_width, frame_height)
            if class_id == 80 and points is not None and len(points) > 0:
                if is_polygon_in_roi(points, roi_left):
                    left_polygons.append(points)
                elif is_polygon_in_roi(points, roi_right):
                    right_polygons.append(points)

    if not left_polygons and not right_polygons:
        current_frame += 1
        continue

    left_a, left_b = None, None
    right_a, right_b = None, None

    if left_polygons:
        left_points = np.vstack(left_polygons)
        left_a, left_b = fit_line(left_points)
        start_point = (0, int(left_b))
        end_point = (frame_width, int(left_a * frame_width + left_b))
        cv2.line(frame, start_point, end_point, (255, 0, 0), thickness=2)

    if right_polygons:
        right_points = np.vstack(right_polygons)
        right_a, right_b = fit_line(right_points)
        start_point = (0, int(right_b))
        end_point = (frame_width, int(right_a * frame_width + right_b))
        cv2.line(frame, start_point, end_point, (0, 255, 0), thickness=2)

    # 중앙 점 및 빨간선 계산
    if left_a is not None and right_a is not None:
        # 초록선과 파란선의 교차점
        center_x = (right_b - left_b) / (left_a - right_a)
        center_y = left_a * center_x + left_b

        # (0.5, 0.5)를 기준으로 한 직선의 기울기 및 절편 계산
        x_center_point = 0.5 * frame_width
        y_center_point = 1.0  * frame_height

        red_slope = (center_y - y_center_point) / (center_x - x_center_point)
        red_intercept = center_y - red_slope * center_x

        # 빨간선 각도 계산 및 방향 표시
        # 빨간선 각도 계산 및 방향 표시 (세로축 기준)
        red_angle = np.degrees(np.arctan(abs(red_slope)))
        vertical_angle = 90 - red_angle  # 세로축 기준 핸들 각도
        if red_slope > 0:
            direction = "left"
        else:
            direction = "right"
        angle_text = f"{direction}: {vertical_angle:.2f} degrees"
        cv2.putText(frame, angle_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 빨간선 그리기 (화면의 좌우 끝까지)
        start_point = (0, int(red_intercept))  # x = 0일 때 y값
        end_point = (frame_width, int(red_slope * frame_width + red_intercept))  # x = frame_width일 때 y값
        cv2.line(frame, start_point, end_point, (0, 0, 255), thickness=2)

        # 흰색 점 (0.5, 0.5) 표시
        cv2.circle(frame, (int(x_center_point), int(y_center_point)), 5, (255, 255, 255), -1)

    for polygon in left_polygons:
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    for polygon in right_polygons:
        cv2.polylines(frame, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.polylines(frame, [roi_left], isClosed=True, color=(255, 255, 255), thickness=2)
    cv2.polylines(frame, [roi_right], isClosed=True, color=(255, 255, 255), thickness=2)

    out.write(frame)
    current_frame += 1

    if os.path.exists(label_path):
        with open(label_path, "w") as f:
            f.truncate(0)

cap.release()
out.release()

runs_path = os.path.join(parent_dir_3, "runs")
if os.path.exists(runs_path):
    import shutil
    shutil.rmtree(runs_path)
    print(f"runs 디렉토리({runs_path})가 삭제되었습니다.")

print(f"10초 동안의 프레임 처리가 완료되었습니다. 결과 비디오는 {output_video_path}에 저장되었습니다.")
