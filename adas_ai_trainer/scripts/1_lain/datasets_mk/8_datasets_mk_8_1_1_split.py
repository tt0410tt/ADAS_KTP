import os
import shutil
from sklearn.model_selection import train_test_split

def split_and_copy(a_folder, b_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 비율 검증
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")
    
    # A 폴더 하위 경로 설정
    images_folder = os.path.join(a_folder, "images")
    labels_folder = os.path.join(a_folder, "labels")
    
    # B 폴더 하위 경로 설정
    b_images_folder = os.path.join(b_folder, "images")
    b_labels_folder = os.path.join(b_folder, "labels")
    
    # B 폴더 train, val, test 디렉토리 생성
    for subdir in ["train", "val", "test"]:
        os.makedirs(os.path.join(b_images_folder, subdir), exist_ok=True)
        os.makedirs(os.path.join(b_labels_folder, subdir), exist_ok=True)

    # A 폴더의 모든 이미지 파일 가져오기
    images = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    # train/val+test 분할
    train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)

    # val/test 분할
    val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

    # 데이터 분할 정보
    datasets = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    # 파일 복사
    for dataset, image_files in datasets.items():
        for image_file in image_files:
            # 이미지 파일 복사
            src_image_path = os.path.join(images_folder, image_file)
            dest_image_path = os.path.join(b_images_folder, dataset, image_file)
            shutil.copy(src_image_path, dest_image_path)

            # 레이블 파일 복사 (이름이 같은 .txt 파일)
            label_file = os.path.splitext(image_file)[0] + ".txt"
            src_label_path = os.path.join(labels_folder, label_file)
            dest_label_path = os.path.join(b_labels_folder, dataset, label_file)
            
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dest_label_path)

    print("Dataset split and copy completed.")
    print(f"Train: {len(train_images)} images")
    print(f"Validation: {len(val_images)} images")
    print(f"Test: {len(test_images)} images")

# 사용 예시
if __name__ == "__main__":
    a_folder = "/data_KTP/승용_자율주행차_데이터셋/datasets_ver2/"  # 원본 데이터셋 폴더
    b_folder = "/data_KTP/승용_자율주행차_데이터셋/datasets_ver2_ttv/"  # 분리된 데이터셋 폴더
    split_and_copy(a_folder, b_folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
