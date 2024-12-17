import json
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob

# 사용한 데이터셋 : 특이도로환경주행데이터
# 데이터셋이 들어간 폴더 경로. 지정해줄것.
json_folder = '/home/kgj0096/라벨링데이터/TL1.1920x1080'
png_folder = '/home/kgj0096/원천데이터/TS1.1920x1080'

# 변환된 값들을 저장하는 폴더 생성(폴더 존재하면 다시 안만듬). 그림 만들때 씀
img_folder = "./contours"
os.makedirs(img_folder, exist_ok=True)
text_folder = "./yolo_seg/labels/train/특이_도로_환경_주행데이터_라벨링_변환"

# 데이터셋 폴더 안에 있는 이미지,라벨링 파일 리스트 불러오기
json_file_list = glob.glob(json_folder + "/**/*.json",
                           recursive=True)
png_file_list = glob.glob(png_folder + "/**/*.png",
                          recursive=True)
# 리스트 정렬. 같은 데이터가 같은 위치에 배치하기 위함
json_file_list = sorted(json_file_list)
png_file_list = sorted(png_file_list)

# 차선 아이디(임시), 해상도(json 파일 하나 참고해서 가져오기. <x> <y> 값임)
CATS_ID = 99
RESOUTION = 1920, 1080

# 라벨링 데이터, 원본데이터를 차례대로 연다
for json_file, png_file in zip(json_file_list, png_file_list):

    # 1. JSON 파일 열기
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 이미지 파일 열기
    image_path = png_file
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 3. 마스크 생성 (흑백 이미지)
    mask = Image.new('L', image.size, 0)  # 흑백 마스크 (0=검정, 255=흰색)
    mask_draw = ImageDraw.Draw(mask)
    # print(image.size)
    # 4. 라벨링 정보에서 각 객체에 대해 segmentation을 그리기
    for annotation in data['annotations']:
        # 5. Segmentation 정보가 있는 경우 (세분화)
        if annotation['category_id'] in [370, 371, 372, 373, 374, 375]:
            segmentation = annotation['polyline']
            seg = segmentation[0]

            # [(x1, y1), (x2, y2), ...] 형식으로 변환
            points = [(seg[i], seg[i+1])
                      for i in range(0, len(seg), 2)]

            # 선 그리기 (각 점을 순차적으로 연결). width 조절해서 마스크 만들기
            draw.line(points, fill='blue', width=25)
            mask_draw.line(points, fill=255, width=25)  # 마스크에 흰색 선 그리기

    # 6. 마스크 테두리 추출
    edge_mask = mask.filter(ImageFilter.FIND_EDGES)

    # 7. 윤곽선 좌표 추출 (Pillow -> OpenCV 변환 후 처리)
    # Pillow 이미지를 OpenCV 형식으로 변환
    edge_mask_cv = np.array(edge_mask)
    contours, _ = cv2.findContours(
        edge_mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선 좌표 단순화 및 저장
    simplified_contour_points = []  # 윤곽선 좌표 저장할 공간1
    new_seg = []  # 윤곽선 좌표 저장할 공간2
    for contour in contours:  # 차선 좌표 덩어리가 하나씩 들어감
        # 좌표 단순화
        epsilon = 0.01 * cv2.arcLength(contour, True)  # 윤곽선 길이의 1%
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        simplified_contour_points.append(
            simplified_contour.squeeze(axis=1).tolist())

        # 좌표 변환을 위해 따로 윤곽선 좌표를 저장+ 리스트 평탄화.
        coordinates = sum(simplified_contour.squeeze(
            axis=1).tolist(), [])

        # x, y 분리해서 해상도따라 각각 처리
        xy = []
        for i in range(0, len(coordinates), 2):
            # 이미지 해상도 고려해서 변환
            x, y = coordinates[i] / RESOUTION[0], coordinates[i+1] / RESOUTION[1]
            # 예시도 소수점 6자리라 6자리로 맞춰줌
            x, y = round(x, 6), round(y, 6)
            # 변환값을 리스트에 추가
            xy.extend([x, y])
        line = f"{CATS_ID} " + " ".join(map(str, xy))
        new_seg.append(line)

    # 텍스트파일 제목과 경로설정
    text_file_path = text_folder + "/" + \
        os.path.splitext(os.path.basename(json_file))[0] + ".txt"

    with open(text_file_path, "w") as text_file:
        # txt 파일을 저장
        text_file.write("\n".join(new_seg))

    # 8. 결과 이미지 표시
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image with Lines")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Edge Mask")
    plt.imshow(edge_mask, cmap='gray')
    plt.axis('off')

    # 단순화된 윤곽선을 새 이미지에 그리기
    simplified_image = Image.new('RGB', image.size, (255, 255, 255))
    simplified_draw = ImageDraw.Draw(simplified_image)
    for simplified_contour in simplified_contour_points:
        # numpy 배열을 list(tuple) 형식으로 변환
        simplified_contour_tuples = [tuple(point)
                                     for point in simplified_contour]
        simplified_draw.line(simplified_contour_tuples +
                             [simplified_contour_tuples[0]],
                             fill='red', width=2)

    plt.subplot(1, 3, 3)
    plt.title("Simplified Contours")
    plt.imshow(simplified_image)
    plt.axis('off')
    plt.show()

    image_file_path = img_folder + "/" + \
        os.path.splitext(os.path.basename(png_file))[0]
    # 9. 결과 이미지 저장
    image.save(image_file_path + '_line.jpg')
    # mask.save(image_file_path + '_mask.jpg')
    # edge_mask.save(image_file_path + '_edge_mask.jpg')
    simplified_image.save(image_file_path + '_simplified_contours.jpg')
