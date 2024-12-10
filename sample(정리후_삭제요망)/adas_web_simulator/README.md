## 실행 방법

### 1.가상 환경 설정 (권장):
```bash
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
```

### 2.Flask 설치:
```bash
pip install Flask
```

### 3.애플리케이션 실행:
```bash
python app.py
```

### 4.웹 브라우저에서 확인:
- 주소창에 http://localhost:8080 입력 후 접속
- 라우트 확인:
  - 메인 페이지: http://localhost:8080/
  - Camera View: http://localhost:8080/menu1/camera-view
  - Settings: http://localhost:8080/menu2/setting
  - Board: http://localhost:8080/menu3/sub1/board
  - 예제 calculator.py는 Camera View에서 확인
