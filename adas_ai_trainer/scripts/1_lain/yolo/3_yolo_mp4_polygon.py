import os
from ultralytics import YOLO
import cv2
import numpy as np
import math
import shutil

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir_2 = os.path.dirname(os.path.dirname(current_dir))
parent_dir_3 = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
parent_dir_4 = os.path.dirname(parent_dir_3)
model_path = os.path.abspath(os.path.join(parent_dir_2, "model/lain/train_yolo11xseg_SO.pt"))
video_path = os.path.join(parent_dir_3, "video/4_mv_1.mp4")
output_video_path = os.path.join(parent_dir_3, "video/4_mv_1_line_curve.mp4")
label_path = os.path.join(parent_dir_3, "runs/segment/predict/labels/image0.txt")
del_path = os.path.join(parent_dir_3, "runs")

# YOLO 모델 로드
model = YOLO(model_path)

# 비디오 파일 읽기
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 10초 동안 처리할 프레임 수 계산
total_frames_to_process = int(fps * 10)
current_frame = 0

def process_line(line, frame_width, frame_height):
    """
    YOLO 텍스트 라인의 데이터를 처리하여 클래스 ID와 폴리곤 좌표를 반환.
    """
    data = line.strip().split()
    if len(data) < 2:
        return None, None  # 유효하지 않은 데이터
    class_id = int(data[0])
    # 좌표 데이터 복원
    points = np.array(data[1:], dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= frame_width  # X 좌표 복원
    points[:, 1] *= frame_height  # Y 좌표 복원
    return class_id, points.astype(np.int32)

def find_roi_from_closest_polygons(points, all_polygons, frame_width, frame_height):
    """
    맨 아래 중앙점에서 가장 가까운 폴리곤의 좌우에서 각각 가장 왼쪽, 오른쪽 점을 찾아
    ROI 영역을 생성.

    Parameters:
        points (ndarray): 현재 폴리곤 좌표 배열.
        all_polygons (list): 모든 폴리곤 리스트.
        frame_width (int): 프레임 너비.
        frame_height (int): 프레임 높이.

    Returns:
        roi_coords (list): ROI 사각형 좌표 [(left_x, top_y), (right_x, bottom_y)].
    """
    # 맨 아래 중앙점 계산
    bottom_central = np.array([frame_width // 2, frame_height])

    # 현재 폴리곤을 제외한 나머지 폴리곤과 거리 계산
    left_polygon = None
    right_polygon = None
    min_left_dist = float('inf')
    min_right_dist = float('inf')

    for poly in all_polygons:
        if np.array_equal(poly, points):
            continue

        # 중심점과 폴리곤의 평균 좌표 계산
        poly_center = np.mean(poly, axis=0)
        dist = np.linalg.norm(poly_center - bottom_central)

        if poly_center[0] < bottom_central[0] and dist < min_left_dist:
            left_polygon = poly
            min_left_dist = dist
        elif poly_center[0] > bottom_central[0] and dist < min_right_dist:
            right_polygon = poly
            min_right_dist = dist

    if left_polygon is None or right_polygon is None:
        return None  # 왼쪽 또는 오른쪽 폴리곤이 없으면 ROI 생성 불가

    # 왼쪽 폴리곤의 가장 왼쪽 점과 오른쪽 폴리곤의 가장 오른쪽 점 계산
    left_most_point = left_polygon[np.argmin(left_polygon[:, 0])]
    right_most_point = right_polygon[np.argmax(right_polygon[:, 0])]

    # ROI 사각형 생성
    roi_top = int(frame_height * 0.5)
    roi_bottom = frame_height

    return [
        (int(left_most_point[0]), roi_top),
        (int(right_most_point[0]), roi_bottom)
    ]

def generate_control_points_with_middle(polygon):
    """
    폴리곤에서 맨 아래와 맨 위 두 점의 중간점을 시작점과 끝점으로 설정하고,
    나머지 점들을 중간 제어점으로 활용.
    """
    if len(polygon) < 4:  # 최소 4개의 점이 필요
        return None

    # Y 좌표 기준 정렬
    sorted_points = polygon[np.argsort(polygon[:, 1])]
    bottom_two = sorted_points[:2]  # 맨 아래 두 점
    top_two = sorted_points[-2:]   # 맨 위 두 점

    # 중간점 계산
    bottom_mid = np.mean(bottom_two, axis=0)
    top_mid = np.mean(top_two, axis=0)

    # 나머지 점들을 중간 제어점으로 사용
    middle_points = sorted_points[2:-2]

    # 최종 제어점 배열 생성 (시작점, 중간 제어점들, 끝점)
    control_points = np.vstack(([bottom_mid], middle_points, [top_mid]))
    return control_points

def bezier_curve(control_points, num_points=100):
    """
    다중 제어점 베지어 곡선 생성.
    """
    control_points = np.array(control_points, dtype=np.float64)  # 제어점의 데이터 타입 명확히 설정
    n = len(control_points) - 1  # 제어점 개수
    t = np.linspace(0, 1, num_points)

    # 베지어 곡선 계산
    curve = np.zeros((num_points, 2), dtype=np.float64)  # float64로 초기화
    for i in range(n + 1):
        binomial_coeff = float(math.comb(n, i))  # math.comb 반환값을 float으로 변환
        curve += binomial_coeff * (1 - t)[:, None]**(n - i) * t[:, None]**i * control_points[i]
    return curve.astype(np.int32)

def dilate_and_erode_curve(frame, curve, kernel_size=12, dilation_iterations=3, erosion_iterations=2):
    """
    베지어 곡선을 팽창시키고 침식으로 다듬는 함수.

    Parameters:
        frame (ndarray): 원본 프레임.
        curve (ndarray): 베지어 곡선의 좌표 배열.
        kernel_size (int): 팽창/침식 커널 크기.
        dilation_iterations (int): 팽창 반복 횟수.
        erosion_iterations (int): 침식 반복 횟수.

    Returns:
        processed_frame (ndarray): 팽창 및 침식된 곡선을 포함한 프레임.
    """
    # 빈 마스크 생성
    binary_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # 곡선을 마스크에 그리기
    cv2.polylines(binary_mask, [curve], isClosed=False, color=255, thickness=2)

    # 커널 생성 (사각형 모양)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # 팽창 연산
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=dilation_iterations)

    # 침식 연산
    eroded_mask = cv2.erode(dilated_mask, kernel, iterations=erosion_iterations)

    # 침식된 마스크를 원본 프레임에 오버레이
    processed_frame = frame.copy()
    processed_frame[eroded_mask > 0] = [255, 0, 255]  # 보라색으로 표시

    return processed_frame

while current_frame < total_frames_to_process:
    ret, frame = cap.read()
    if not ret:
        print("더 이상 읽을 수 있는 프레임이 없습니다.")
        break

    # YOLO 추론
    results = model.predict(source=frame, save_txt=True)

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        polygons = []
        for line in lines:
            class_id, points = process_line(line, frame_width, frame_height)
            if points is not None and class_id == 80:
                polygons.append(points)

        for points in polygons:
            # **초록색 폴리곤**: 모든 클래스에 대해 표시
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # **파란색 베지어 곡선 및 ROI 사각형**: 클래스가 80인 경우
            roi_coords = find_roi_from_closest_polygons(points, polygons, frame_width, frame_height)
            if roi_coords is not None:
                left_point, right_point = roi_coords

                # 사각형 시각화
                cv2.rectangle(
                    frame,
                    left_point,
                    right_point,
                    color=(0, 0, 255),  # 빨간색
                    thickness=2,
                )

                # ROI 영역 내에서 베지어 곡선 생성
                control_points = generate_control_points_with_middle(points)
                if control_points is None:
                    continue  # 점이 부족하면 스킵

                curve = bezier_curve(control_points)
                if curve is not None:
                    # 팽창 및 침식된 곡선 표시
                    frame = dilate_and_erode_curve(frame, curve, kernel_size=12, dilation_iterations=3, erosion_iterations=2)

        # 라벨 파일 초기화
        with open(label_path, "w") as f:
            f.write("")

    # 처리된 프레임 저장
    out.write(frame)
    current_frame += 1

cap.release()
out.release()

if os.path.exists(del_path):
    shutil.rmtree(del_path)
    print(f"폴더 {del_path}가 삭제되었습니다.")
else:
    print(f"폴더 {del_path}를 찾을 수 없습니다.")

print(f"10초 동안의 프레임 처리가 완료되었습니다. 결과 비디오는 {output_video_path}에 저장되었습니다.")
