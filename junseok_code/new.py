# RGB값 비교

import cv2
import numpy as np

# 이미지 읽기
img1 = cv2.imread("C:\\Camera\\IMG_2094.JPG")
img2 = cv2.imread("C:\\Camera\\IMG_2096.JPG")

# 이미지 크기 조절
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# MSE 계산
mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
mse /= float(img1.shape[0] * img1.shape[1])

print(f"MSE: {mse}")
