import os

# 폴더 경로 설정
folder_a = "/data_KTP/승용_자율주행차_데이터셋/label_detecting"  # A 폴더 경로
folder_b = "/data_KTP/승용_자율주행차_데이터셋/label_to_text"  # B 폴더 경로
folder_c = "/data_KTP/승용_자율주행차_데이터셋/label_get"  # C 폴더 경로 (병합된 파일 저장)

# C 폴더가 없으면 생성
if not os.path.exists(folder_c):
    os.makedirs(folder_c)

# A 폴더의 파일 목록 가져오기
a_files = [f for f in os.listdir(folder_a) if f.endswith(".txt")]
b_files = [f for f in os.listdir(folder_b) if f.endswith(".txt")]

# 파일 병합
all_files = set(a_files + b_files)

for file_name in all_files:
    # A 폴더 파일 경로
    file_a_path = os.path.join(folder_a, file_name)

    # B 폴더 파일 경로
    file_b_path = os.path.join(folder_b, file_name)

    # C 폴더에 병합 파일 경로 생성
    merged_file_path = os.path.join(folder_c, file_name)

    # 파일 병합 또는 단독 복사
    with open(merged_file_path, "w") as merged_file:
        # A 파일 내용 추가
        if os.path.exists(file_a_path):
            with open(file_a_path, "r") as file_a:
                merged_file.write(file_a.read())

        # B 파일 내용 추가
        if os.path.exists(file_b_path):
            if os.path.exists(file_a_path):
                merged_file.write("\n")  # 구분을 위한 줄바꿈 추가
            with open(file_b_path, "r") as file_b:
                merged_file.write(file_b.read())

print("모든 파일 병합이 완료되었습니다.")
