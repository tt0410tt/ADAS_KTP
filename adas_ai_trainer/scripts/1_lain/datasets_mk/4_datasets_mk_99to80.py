import os

def process_files_in_folder(folder_path):
    # 폴더에서 파일 목록 읽기
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # 텍스트 파일만 처리
            file_path = os.path.join(folder_path, filename)
            
            # 파일 읽기
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            processed_lines = []

            for line in lines:
                # 빈 줄 제거
                if line.strip():  # 줄이 비어 있지 않을 경우만 처리
                    parts = line.strip().split()
                    # 첫 부분이 '99'면 '80'으로 변경
                    if parts[0] == '99':
                        parts[0] = '80'
                    # 처리된 라인을 추가
                    processed_lines.append(' '.join(parts) + '\n')

            # 파일 쓰기
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(processed_lines)

# 사용 예시
folder_path = '/data_KTP/승용_자율주행차_데이터셋/datasets_ver2/labels'  # 여기에 실제 폴더 경로를 입력
process_files_in_folder(folder_path)
