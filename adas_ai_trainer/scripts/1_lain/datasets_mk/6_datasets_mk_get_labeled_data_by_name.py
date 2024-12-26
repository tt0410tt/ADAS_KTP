import os
import shutil
import glob

def process_folders(folder1, folder2, folder3, folder4):
    # 1. 폴더1의 모든 사진 파일 읽기
    image_files = glob.glob(os.path.join(folder1, '*.*'))  # 이미지 파일들 (*.*로 모든 파일 읽기)
    
    # 순번 초기화
    sequence_number = 0

    for image_file in image_files:
        # 2. 파일 이름 변경: o_순번.jpg
        new_image_name = f"o_{sequence_number}.jpg"
        new_image_path = os.path.join(folder3, new_image_name)
        
        # 3. 이미지 파일을 폴더3으로 복사
        shutil.copy(image_file, new_image_path)
        print(f"Copied and renamed: {image_file} -> {new_image_path}")

        # 4. 이미지 파일의 이름과 같은 txt 파일을 폴더2에서 찾기
        file_name = os.path.splitext(os.path.basename(image_file))[0]
        txt_file = os.path.join(folder2, f"{file_name}.txt")
        
        if os.path.exists(txt_file):
            # 5. txt 파일 이름 변경: o_순번.txt
            new_txt_name = f"o_{sequence_number}.txt"
            new_txt_path = os.path.join(folder4, new_txt_name)

            # 6. 폴더4로 복사
            shutil.copy(txt_file, new_txt_path)
            print(f"Copied and renamed: {txt_file} -> {new_txt_path}")

        # 순번 증가
        sequence_number += 1

# 사용 예시
if __name__ == "__main__":
    # 각 폴더 경로 설정
    folder1 = "/data_KTP/승용_자율주행차_데이터셋/output_groups_select"
    folder2 = "/data_KTP/승용_자율주행차_데이터셋/label_get/"
    folder3 = "/data_KTP/승용_자율주행차_데이터셋/datasets/images/"
    folder4 = "/data_KTP/승용_자율주행차_데이터셋/datasets/labels/"

    # 폴더 생성 (존재하지 않을 경우 대비)
    for folder in [folder1, folder2, folder3, folder4]:
        os.makedirs(folder, exist_ok=True)

    # 폴더 처리 실행
    process_folders(folder1, folder2, folder3, folder4)

    print("작업이 완료되었습니다.")
