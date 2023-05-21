# 이미지 분석, RGB값 비교후 가장근접한 이미지 출력

import cv2
import numpy as np
import os

# 이미지 크기
IMAGE_SIZE = (500, 500)

# 기준이 될 이미지 경로
base_img_path = "C:\\Users\\000\\Desktop\\Example3.JPG"

# 비교할 이미지들이 있는 디렉토리
compare_dir = "C:\\Users\\000\\Desktop\\Rosies"

# 이미지 읽기
base_img = cv2.imread(base_img_path)
base_img = cv2.resize(base_img, IMAGE_SIZE)

# 기준 이미지와 가장 유사한 이미지와 그 유사도를 저장할 변수
most_similar_img_path = None
lowest_mse = float('inf')

# 디렉토리 안의 모든 이미지에 대해
for compare_img_name in os.listdir(compare_dir):
    compare_img_path = os.path.join(compare_dir, compare_img_name)

    # 이미지 읽기
    compare_img = cv2.imread(compare_img_path)

    # 이미지가 제대로 읽혔는지 확인
    if compare_img is None:
        print(f"Cannot read image: {compare_img_path}")
        continue

    compare_img = cv2.resize(compare_img, IMAGE_SIZE)

    # 나머지 코드...

    # MSE 계산
    mse = np.sum((base_img.astype("float") - compare_img.astype("float")) ** 2)
    mse /= float(base_img.shape[0] * base_img.shape[1])

    # 이 이미지의 MSE가 지금까지 찾은 이미지의 MSE보다 작으면 이 이미지를 가장 유사한 이미지로 갱신
    if mse < lowest_mse:
        most_similar_img_path = compare_img_path
        lowest_mse = mse

# MSE 값을 0~1 사이의 유사도로 변환 (MSE가 0이면 유사도 1, MSE가 10000이면 유사도 0)
similarity = 1 - (lowest_mse / 195075)

# 유사도를 퍼센트로 변환
similarity_percent = similarity * 100

print(f"Most similar image is {most_similar_img_path}, Similarity: {similarity_percent}%")

# 가장 유사한 이미지 읽기
most_similar_img = cv2.imread(most_similar_img_path)
most_similar_img = cv2.resize(most_similar_img, IMAGE_SIZE)

# 이미지 표시
cv2.imshow('Base Image', base_img)
cv2.imshow('Most Similar Image', most_similar_img)

# 키 입력을 기다리고 모든 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()