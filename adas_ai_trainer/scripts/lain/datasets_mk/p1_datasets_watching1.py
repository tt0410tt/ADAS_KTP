import cv2
import os
import numpy as np

def draw_polygons(image_path, txt_path, output_path):
    """
    텍스트 파일에 정의된 폴리곤을 이미지 위에 그려 저장하는 함수
    :param image_path: 이미지 파일 경로
    :param txt_path: 텍스트 파일 경로
    :param output_path: 결과 이미지를 저장할 경로
    """
    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 파일을 읽을 수 없습니다: {image_path}")
        return

    # 텍스트 파일 읽기
    if not os.path.exists(txt_path):
        print(f"텍스트 파일이 존재하지 않습니다: {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 각 라인에서 폴리곤 정보를 파싱
    for line in lines:
        data = line.strip().split()
        if len(data) < 2:
            continue

        object_id = int(data[0])  # 물체 번호
        coordinates = list(map(float, data[1:]))

        # 좌표를 (x, y) 형식으로 변환
        points = []
        for i in range(0, len(coordinates), 2):
            x = int(coordinates[i] * image.shape[1])
            y = int(coordinates[i + 1] * image.shape[0])
            points.append((x, y))

        # 폴리곤 그리기
        points = np.array(points, dtype=np.int32)
        color = (0, 255, 0)  # 초록색
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)

        # 물체 번호 표시
        text_position = (points[0][0], points[0][1] - 10)
        cv2.putText(image, str(object_id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 결과 저장
    cv2.imwrite(output_path, image)
    print(f"결과 이미지를 저장했습니다: {output_path}")

a_folder = "/data_KTP/승용_자율주행차_데이터셋/train"  # 사진 폴더
b_folder = "/data_KTP/승용_자율주행차_데이터셋/label_get"  # 텍스트 파일 폴더
output_folder = "22_111609_220623_19"  # 결과 이미지 저장 폴더

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# A폴더와 B폴더의 파일 이름을 기준으로 매칭
for image_file in os.listdir(a_folder):
    if image_file.endswith((".jpg", ".png")):  # 이미지 파일 필터링
        image_path = os.path.join(a_folder, image_file)
        txt_file = os.path.splitext(image_file)[0] + ".txt"  # 동일한 이름의 txt 파일 찾기
        txt_path = os.path.join(b_folder, txt_file)

        # 출력 파일 경로 설정
        output_path = os.path.join(output_folder, image_file)

        # 폴리곤을 이미지에 대입
        draw_polygons(image_path, txt_path, output_path)
