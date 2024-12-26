from ultralytics import YOLO
import cv2
import numpy as np
import glob
import os

# YOLO11n 모델 불러오기
model = YOLO("yolo11x-seg.pt")

# 입력 이미지 경로 및 출력 TXT 파일 경로
input_image_path = "/data_KTP/승용_자율주행차_데이터셋/train/"  # 감지할 이미지 경로
output_text_path = "/data_KTP/승용_자율주행차_데이터셋/label_detecting/"  # 폴리곤 데이터를 저장할 폴더 경로
ouput_file_name=""
def save_polygons_to_txt(polygons, txt_file):
    """
    폴리곤 좌표를 TXT 파일로 저장하는 함수
    """
    with open(txt_file, "a") as f:
        for poly in polygons:
            poly_str = " ".join([f"{int(x[0])},{int(x[1])}" for x in poly])
            f.write(poly_str + "\n")

CATS_ID = 99  # 차선 카테고리 아이디 기입. 그런데 그냥 9번으로 따로 저장할 필요는 없는 것 같다.

image_file_list = glob.glob(input_image_path + "*.jpg", recursive=True)

# YOLO 모델 추론
for image_file in image_file_list:
    results = model(image_file)
    image = cv2.imread(image_file)
    original_height, original_width = image.shape[:2]  # 원본 이미지 크기

    ouput_file_name = image_file.replace(input_image_path, "").replace("jpg", "txt")
    with open(output_text_path + ouput_file_name, "w+") as f:
        f.close()

    for result in results:  # results에서 각 객체 정보 추출
        if len(result) == 0:
            # 물체 검출이 실패했을 경우
            continue

        for mask, box in zip(result.masks.data, result.boxes):
            class_id = int(box.cls[0].item())  # 클래스 ID
            confidence = float(box.conf[0].item())  # 신뢰도

            # 마스크 데이터 처리
            mask = mask.cpu().numpy()  # 텐서를 NumPy 배열로 변환
            mask = (mask * 255).astype(np.uint8)  # 이진 마스크를 0~255 범위로 변환

            # YOLO 모델 입력 크기 설정 (기본값: 640x640)
            yolo_input_size = model.overrides.get('imgsz', 640)  # 모델 입력 크기 가져오기 (없으면 기본값 640)
            yolo_height, yolo_width = yolo_input_size, yolo_input_size
            # 마스크 크기 조정 (원본 이미지 크기와 일치하도록)
            resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

            # 마스크 좌표 추출
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not os.path.exists(output_text_path + ouput_file_name):
                os.system('touch ' + output_text_path + ouput_file_name)

            with open(output_text_path + ouput_file_name, "a+") as f:
                for contour in contours:
                    f.write(str(class_id) + " ")
                    for point in contour:
                        x, y = point[0][0] / original_width, point[0][1] / original_height
                        x, y = round(x, 6), round(y, 6)
                        f.write(str(x) + " " + str(y) + " ")
                    f.write("\n")

print("변환이 끝났습니다")
