# 베이스 이미지로 Python 3.9 사용
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 종속성 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . /app/

# 실행 파일 설정
CMD ["python", "prd/run.py"]
