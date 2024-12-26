import cv2
from ultralytics import YOLO

# YOLO 모델 로드
model = YOLO("/data_KTP/SO/runs/segment/train11/weights/222.pt")

# 비디오 파일 입력 및 출력 설정
input_video_path = "/data_KTP/SO/분당판교 드라이브_1.mp4"  # 입력 영상 경로
output_video_path = "/data_KTP/SO/분당판교 드라이브_1_변환.mp4"  # 출력 영상 경로

cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # 초당 프레임 수
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 코덱
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# 검출된 시간 구간 저장 리스트
detection_times = []

# 현재 프레임 추적
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # YOLO 모델로 프레임 처리
    results = model.predict(source=frame, conf=0.25)

    # 검출된 객체 가져오기
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding Box 좌표
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence Scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class IDs

    # 객체가 검출된 경우
    if len(detections) > 0:
        # 현재 시간 계산
        current_time = frame_number / fps  # 초 단위 시간 계산
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        detection_times.append((minutes, seconds))

        # 검출 결과를 프레임에 그리기
        for box, score, label in zip(detections, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            confidence = f"{score:.2f}"
            color = (0, 255, 0)  # 바운딩 박스 색상 (녹색)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"Class {int(label)} {confidence}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 변환된 프레임 저장
    out.write(frame)

    # 실시간 보기 (필요하면 주석 해제)
    # cv2.imshow("YOLO Detection", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
