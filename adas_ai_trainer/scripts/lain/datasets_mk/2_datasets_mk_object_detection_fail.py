from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os

# YOLO11n 모델 불러오기
model = YOLO("yolo11x-seg.pt")

# 입력 이미지 경로 및 출력 폴더 경로
input_image_path = "/data_KTP/승용_자율주행차_데이터셋/train/"
output_text_path = "/data_KTP/승용_자율주행차_데이터셋/label_detecting/"

# 이미지 파일 리스트 가져오기
image_file_list = glob.glob(input_image_path + "*.jpg", recursive=True)

# YOLO 모델 추론
for image_file in image_file_list:
    results = model(image_file)
    image = cv2.imread(image_file)
    original_height, original_width = image.shape[:2]

    ouput_file_name = image_file.replace(input_image_path, "").replace("jpg", "txt")
    # 폴더 생성 및 내용있을시 내용제거
    with open(output_text_path + ouput_file_name, "w+") as f:
        f.close()

    for result in results:  # 결과 처리
        if len(result) == 0:
            continue

        for mask, box in zip(result.masks.data, result.boxes):
            class_id = int(box.cls[0].item())  # 클래스 ID
            confidence = float(box.conf[0].item())  # 신뢰도

            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 바운딩 박스 중심 계산
            bounding_box_center_x = (x1 + x2) / 2
            bounding_box_center_y = (y1 + y2) / 2

            # 바운딩 박스 그리기
            color = (0, 255, 0)  # 녹색
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{class_id} Conf:{confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 마스크 데이터 처리
            mask = mask.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)

            # 마스크 크기 조정
            # YOLO 모델의 입력 크기 확인
            yolo_input_size = model.overrides.get('imgsz', 640)  # 기본값: 640
            yolo_height, yolo_width = yolo_input_size, yolo_input_size
            resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

            # 마스크 윤곽 추출
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # 폴리곤 외곽점(왼쪽, 오른쪽, 위, 아래) 찾기
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                topmost = tuple(contour[contour[:, :, 1].argmin()][0])
                bottommost = tuple(contour[contour[:, :, 1].argmax()][0])

                # 폴리곤 외곽 사각형 중심 계산
                polygon_center_x = (leftmost[0] + rightmost[0]) / 2
                polygon_center_y = (topmost[1] + bottommost[1]) / 2

                # 중심 차이 계산
                delta_x = bounding_box_center_x - polygon_center_x
                delta_y = bounding_box_center_y - polygon_center_y

                # 폴리곤 이동
                aligned_contour = []
                for point in contour:
                    new_x = point[0][0] + delta_x
                    new_y = point[0][1] + delta_y
                    aligned_contour.append([[new_x, new_y]])
                aligned_contour = np.array(aligned_contour, dtype=np.int32)

                # 이동된 폴리곤 그리기
                cv2.drawContours(image, [aligned_contour], -1, (255, 0, 0), 2)  # 파란색 폴리곤

                # 폴리곤 데이터 저장
                with open(output_text_path + ouput_file_name, "a+") as f:
                    f.write(str(class_id) + " ")
                    for point in aligned_contour:
                        x, y = point[0][0] / original_width, point[0][1] / original_height
                        x, y = round(x, 6), round(y, 6)
                        f.write(str(x) + " " + str(y) + " ")
                    f.write("\n")

print("변환 끝났습니다.")