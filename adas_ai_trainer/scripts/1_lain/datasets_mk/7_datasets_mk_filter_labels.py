import os

def clean_txt_files_in_folder(folder_path):
    # 폴더 내의 모든 파일을 확인
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # txt 파일만 처리
            file_path = os.path.join(folder_path, filename)
            
            # 파일을 읽어서 라인을 처리
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 필터링: 공백으로 나누었을 때 길이가 5 미만인 라인 제거
            cleaned_lines = [line for line in lines if len(line.split()) >= 5]

            # 필터링된 결과를 다시 파일에 기록
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(cleaned_lines)

# 사용 예시: "your_folder_path"에 원하는 폴더 경로를 입력
folder_path = "/data_KTP/승용_자율주행차_데이터셋/datasets_ver2/labels"
clean_txt_files_in_folder(folder_path)
