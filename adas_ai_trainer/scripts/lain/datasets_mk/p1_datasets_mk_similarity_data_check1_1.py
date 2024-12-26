from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from numba import jit
"""
다양한 방법으로 이미지 비교하는 프로그램
- sift : 유사특징점 / 전체 특징점. 유사특징점의 숫자가 높을수록 유사도 높음
- sam : 0~1 사이의 유사도. 숫자가 높을수록 유사도 높음
- mse : 숫자가 낮을수록 유사도가 높음
"""
def sift(image_path_1,image_path_2):
    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    image_a = cv2.imread(image_path_1, cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(image_path_2, cv2.IMREAD_GRAYSCALE)
    # 특징점 및 디스크립터 추출
    keypoints1, descriptors1 = sift.detectAndCompute(image_a, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image_b, None)

    # 매칭을 위한 BFMatcher 생성
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 특징점 매칭
    matches = bf.match(descriptors1, descriptors2)

    # 매칭 결과 정렬
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭 결과 시각화
    print(f"sift : {len(matches)}/{len(descriptors1)}")

def calculate_ssim(image_a_path, image_b_path):
    """
    Calculate the SSIM (Structural Similarity Index) between two images.

    Args:
        image_a_path (str): Path to the first image file.
        image_b_path (str): Path to the second image file.

    Returns:
        float: SSIM value between the two images.
    """
    # Load images in grayscale
    image_a = cv2.imread(image_a_path, cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(image_b_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded correctly
    if image_a is None or image_b is None:
        raise ValueError("One or both image paths are invalid or the files cannot be read.")

    # Ensure the images have the same dimensions
    if image_a.shape != image_b.shape:
        raise ValueError("The two images must have the same dimensions.")

    # Compute SSIM
    ssim_value = ssim(image_a, image_b)
    return ssim_value

def mse(imageA, imageB):
    # 두 이미지의 차이를 계산
    im1 = cv2.imread(imageA)
    im2 = cv2.imread(imageB)
    err = np.sum((im1.astype("float") - im2.astype("float")) ** 2)
    err /= float(im1.shape[0] * im1.shape[1])
    return err


# Example usage
if __name__ == "__main__":
    image_a_path = "/data_KTP/승용_자율주행차_데이터셋/train/08_085504_221103_27.jpg"  # Replace with the path to your first image
    image_b_path = "/data_KTP/승용_자율주행차_데이터셋/train/08_085504_221103_28.jpg"  # Replace with the path to your second image
    output_path="/data_KTP/SO/test1.jpg"
    ratio = mse(image_a_path,image_b_path)
    print(f"mse : {ratio}")
    try:
        similarity = calculate_ssim(image_a_path, image_b_path)
        print(f"SSIM : {similarity:.4f}")
    except ValueError as e:
        print(e)
    
    sift(image_a_path,image_b_path)
