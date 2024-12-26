from ultralytics import YOLO
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1번만 사용
def get_gpu_info():
    # 총 GPU 개수
    gpu_count = torch.cuda.device_count()
    print(f"Total GPUs available: {gpu_count}")

    # GPU가 사용 가능한지 확인
    if gpu_count == 0 or not torch.cuda.is_available():
        print("No GPUs are available on this system.")
        return []

    available_gpus = []

    for i in range(gpu_count):
        # GPU 이름 가져오기
        gpu_name = torch.cuda.get_device_name(i)
        # GPU 메모리 사용량 확인
        gpu_stats = torch.cuda.memory_stats(i)
        used_memory = gpu_stats.get('active_bytes.all.current', 0) / (1024 ** 2)  # MB 단위
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)  # MB 단위

        # GPU 사용 가능 상태 출력
        print(f"GPU {i}: {gpu_name}")
        print(f"  Total Memory: {total_memory:.2f} MB")
        print(f"  Used Memory: {used_memory:.2f} MB")

        if used_memory < total_memory * 0.9:  # 사용 가능한 메모리가 90% 이상일 때 사용 가능으로 표시
            available_gpus.append(i)
            print("  Status: Available")
        else:
            print("  Status: In Use")

    return available_gpus

if __name__ == "__main__":
    available_gpus = get_gpu_info()

    if available_gpus:
        print(f"Available GPUs: {available_gpus}")
    else:
        print("No GPUs are currently available for use.")

    # 선택한 디바이스 설정
    device = f"cuda:{available_gpus[0]}" if available_gpus else "cpu"
    print(f"Selected device: {device}")

    # YOLO Segmentation 모델 로드
    model = YOLO("/data_KTP/SO/yolo11x-seg.pt")  # 모델 파일 경로

    # 모델 학습 실행
    model.train(
        data="/data_KTP/SO/data.yaml",  # 데이터 YAML 경로
        epochs=100,                   # 데이터 확인을 위한 1 epoch 실행
        imgsz=640,                  # 이미지 크기
        batch=8,                    # 배치 크기
        device=device               # 선택된 디바이스
    )
