import os
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import torch

# GPU 설정
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 폴더 설정
input_folder = "/data_KTP/승용_자율주행차_데이터셋/train"
output_folder = "/data_KTP/승용_자율주행차_데이터셋/output_groups"
os.makedirs(output_folder, exist_ok=True)

# 유사도 기준
similarity_threshold = 0.35
batch_size = 10  # 한 번에 처리할 이미지 개수

# 함수 정의
def read_images_in_batches(folder_path, batch_size):
    filenames = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    for i in range(0, len(filenames), batch_size):
        batch_filenames = filenames[i:i + batch_size]
        images = []
        for filename in batch_filenames:
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
        yield images, batch_filenames, len(filenames)

def compute_ssim(img1, img2):
    score, _ = ssim(img1, img2, full=True)
    return score

def group_images_by_similarity_in_batches(folder_path, batch_size, threshold):
    groups = []
    group_filenames = []
    processed_count = 0

    for images, filenames, total_count in read_images_in_batches(folder_path, batch_size):
        for i, img in enumerate(images):
            processed_count += 1
            added = False
            recent_groups = list(zip(groups, group_filenames))[-5:]  # 최근 5개 그룹만 확인
            for group, group_files in recent_groups:
                main_img = group[0]  # 그룹의 메인 이미지
                if compute_ssim(main_img, img) >= threshold:
                    group.append(img)
                    group_files.append(filenames[i])
                    added = True
                    break
            if not added:
                groups.append([img])
                group_filenames.append([filenames[i]])
            
            # 프롬프트에 진행 상황 출력
            print(f"Processed {processed_count}/{total_count} images. Total groups so far: {len(groups)}")

    return group_filenames

# 그룹화 수행
grouped_filenames = group_images_by_similarity_in_batches(input_folder, batch_size, similarity_threshold)

# 결과 저장 및 출력
for idx, group in enumerate(grouped_filenames):
    group_folder = os.path.join(output_folder, f"group_{idx + 1}")
    os.makedirs(group_folder, exist_ok=True)
    for file in group:
        filename = os.path.basename(file)
        dest_path = os.path.join(group_folder, filename)
        cv2.imwrite(dest_path, cv2.imread(file))

print(f"Grouping complete. Total groups: {len(grouped_filenames)}")
