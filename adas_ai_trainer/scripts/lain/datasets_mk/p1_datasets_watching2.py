import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 경로 설정
b_folder = "/data_KTP/datasets/labels/"  # 텍스트 파일이 있는 폴더
a_folder = "/data_KTP/datasets/images/"  # 이미지가 있는 폴더

# 결과 저장 폴더
output_folder = "/data_KTP/승용_자율주행차_데이터셋/datasets/check"
os.makedirs(output_folder, exist_ok=True)
# 이미지 매칭이 안된 텍스트 파일 정리
unmatched_file = os.path.join(output_folder, "unmatched_files.txt")
with open(unmatched_file, "w") as uf:
    uf.write("이미지 매칭이 안된 텍스트 파일 목록:\n")

# 텍스트 파일 처리 함수
def decode_polygon(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    decoded_data = []
    for line in lines:
        parts = line.strip().split(" ")
        object_id = int(parts[0])  # 객체 ID
        polygon = list(map(float, parts[1:]))  # 폴리곤 데이터

        # 폴리곤 좌표를 (x, y) 쌍으로 변환
        points = [(polygon[i] * 1920, polygon[i + 1] * 1080) for i in range(0, len(polygon), 2)]
        decoded_data.append((object_id, points))

    return decoded_data

# 이미지와 폴리곤 데이터 시각화 및 저장
def visualize_and_save(image_path, decoded_data, output_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    # 폴리곤 그리기
    for object_id, points in decoded_data:
        try:
            print(f"객체 ID: {object_id}, 폴리곤 좌표: {points}")
            points = np.array(points, dtype=np.int32)
            print(f"변환된 numpy 배열: {points}")
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(image, str(object_id), (int(points[0][0]), int(points[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        except Exception as e:
            print(f"폴리곤 그리기 중 오류 발생 - 객체 ID: {object_id}, 오류: {e}")

    # 결과 이미지 저장
    try:
        cv2.imwrite(output_path, image)
        print(f"저장 완료: {output_path}")
    except Exception as e:
        print(f"이미지 저장 중 오류 발생: {e}")

# B 폴더의 텍스트 파일 처리 및 A 폴더의 이미지와 매칭
for txt_file in os.listdir(b_folder):
    if txt_file.endswith(".txt"):
        txt_path = os.path.join(b_folder, txt_file)

        # 폴리곤 데이터 복호화
        decoded_data = decode_polygon(txt_path)

        # 매칭되는 이미지 경로 찾기
        base_name = os.path.splitext(txt_file)[0]
        image_path_jpg = os.path.join(a_folder, base_name + ".jpg")
        image_path_png = os.path.join(a_folder, base_name + ".png")

        if os.path.exists(image_path_jpg):
            output_path = os.path.join(output_folder, base_name + ".jpg")
            visualize_and_save(image_path_jpg, decoded_data, output_path)
        elif os.path.exists(image_path_png):
            output_path = os.path.join(output_folder, base_name + ".png")
            visualize_and_save(image_path_png, decoded_data, output_path)
        else:
            print(f"이미지 파일이 없습니다: {base_name}.jpg 또는 {base_name}.png")
            with open(unmatched_file, "a") as uf:
                uf.write(f"{txt_file}\n")
