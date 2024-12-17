"""
이 파일은 데이터셋이 들어간 폴더 경로를 지정해주면 라벨링 데이터를 자동으로 찾아서 YOLO-seg 데이터셋의 라벨링 데이터로 변환해주는 파일이다.
카테고리 아이디는 차선번호로 설정을 해 줘야 한다. 현재는 임의로 cats_id = 99 를 써서 99번 카테고리를 차선으로 설정했다.
세그맨테이션으로 차선을 인식하는 데이터셋이 있어야 동작한다.
아래의 folder 변수에 데이터셋의 절대 경로를 지정해주면 자동으로 변환해준다.
변환된 라벨링 데이터셋은 코드를 실행한 위치에 출력된다.
YOLO-seg 라벨링 데이터는 <class-index> <x1> <y1> <x2> <y2> ... <xn> <yn> 형식으로 .txt 파일로 저장된다.
현재 이 파일은 '승용 자율주행차 주간 도심도로 데이터' 만 인식한다. 다른것들도 추후에 추가할 예정이다.

'승용 자율주행차 주간 도심도로 데이터' 는 세그맨테이션으로 차선을 인식한다.

'승용 자율주행차 주간 도심도로 데이터'
1920 1080
08_102940_220615_01.jpg 인 데이터가 들어간 파일임.


"""

import os
import json
import glob


# 데이터셋이 들어간 폴더 경로. 지정해줄것.
folder = '/home/kgj0096/Sample'

# 데이터셋 폴더 안에 있는 라벨링 파일 리스트 불러오기
file_list = glob.glob(folder + "/**/*.json",
                      recursive=True)

# 변환된 값들을 저장하는 폴더 생성(폴더 존재하면 다시 안만듬)
text_folder = "./yolo_seg/labels/train/승용_자율주행차_주간_도심도로_데이터_라벨링_변환"
os.makedirs(text_folder, exist_ok=True)

CATS_ID = 99  # 차선 카테고리 아이디 기입. 그런데 그냥 9번으로 따로 저장할 필요는 없는 것 같다.
RESOUTION = 1920, 1080  # 해상도 : 1920 1080

# for문으로 파일명 파일 이름 넣기
for file in file_list:

    # 변환할 라벨링 파일
    json_file_path = file

    # YOLO에 쓸 라벨링 파일 형식(변환결과물 이름)
    # splitext 함수는 파일 절대 경로를 받으면 (파일경로+파일이름, 파일 확장자)를 출력하는 함수다.
    # basename 함수는 파일 이름(확장자포함)만 출력하는 함수다.
    text_file_path = text_folder + "/" + \
        os.path.splitext(os.path.basename(json_file_path))[0] + ".txt"

    # 라벨링 데이터 읽기
    with open(json_file_path) as json_file:
        json_data = json.load(json_file)

    # '승용 자율주행차 주간 도심도로 데이터' 변환 코드

    # 주석 찾기. 세그맨테이션이랑 카테고리 아이디 찾기용.
    annotations = json_data['annotations']

    # 텍스트 파일 열어서 쓰기
    with open(text_file_path, "w") as text_file:
        for annotation in annotations:

            # 카테고리 아이디가 차선인지 확인
            if annotation['class'] in ['whiteLane', 'yellowLane']:

                # 세그맨테이션 항목을 따로 저장.
                seg = annotation['polygon']

                # 세그맨테이션 항목을 이미지 해상도를 이용해 0~1 사이 값들로 변환한다.
                # x1 y1 x2 y2 x3 y3.. 이런 식이다.
                new_seg = []  # 변환값 넣을 리스트

                # x, y 분리해서 해상도따라 각각 처리
                for i in range(0, len(seg), 2):
                    # 이미지 해상도 고려해서 변환
                    x, y = seg[i] / RESOUTION[0], seg[i+1] / RESOUTION[1]
                    # 예시도 소수점 6자리라 6자리로 맞춰줌
                    x, y = round(x, 6), round(y, 6)
                    # 변환값을 리스트에 추가
                    new_seg.extend([x, y])

                seg_data = []  # YOLO-seg 라벨링 데이터 한 줄에 쓸 리스트 선언
                seg_data.append(CATS_ID)  # 차선 카테고리 아이디 추가
                seg_data.extend(new_seg)  # 변환된 세그맨테이션 값 추가

                # txt 파일을 저장
                text_file.write(" ".join(map(str, seg_data)) + "\n")
