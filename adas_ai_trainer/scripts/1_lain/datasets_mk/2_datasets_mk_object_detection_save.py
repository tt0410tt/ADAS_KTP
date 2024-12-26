import os
from ultralytics import YOLO
import glob

# 모델 로드
model = YOLO("yolo11x-seg.pt")

# 입력 이미지 폴더 경로 및 결과 저장 디렉토리 설정
input_image_folder = "/data_KTP/승용_자율주행차_데이터셋/train/"  # 이미지 폴더 경로
save_dir = "/data_KTP/승용_자율주행차_데이터셋/results"  # 결과를 저장할 디렉토리
os.makedirs(save_dir, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_file_list = glob.glob(os.path.join(input_image_folder, "*.jpg"))

# 이미지 파일이 없을 경우 에러 처리
if not image_file_list:
    raise FileNotFoundError(f"폴더에 이미지 파일이 없습니다: {input_image_folder}")

# 단일 결과 폴더에 저장 설정
run_name = "run_combined"

# 이미지 파일들에 대해 모델 추론 수행
for image_path in image_file_list:
    # 모델 추론 및 결과 저장
    results = model.predict(source=image_path, save=True, save_txt=True, project=save_dir, name=run_name)

# 결과 시각화 완료 메시지
print(f"폴더 내 모든 이미지에 대해 추론이 완료되었습니다. 결과는 {save_dir}/{run_name} 디렉토리에 저장되었습니다.")
