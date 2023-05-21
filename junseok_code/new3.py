# 타겟 이미지: RGB 추출, 군중 이미지: YOLO로 탐지된 모든 객체에 바운딩 박스 그림
# 모든 박스의 RGB 추출후 타겟 이미지의 RGB와 가장 유사한 바운딩 박스 출력

import cv2
import numpy as np

# load YOLO
net = cv2.dnn.readNet("C:\\Users\\000\\Desktop\\Capston\\yolo\\yolov4.weights", "C:\\Users\\000\\Desktop\\Capston\\yolo\\yolov4.cfg.txt")

# Load names of classes and get the output layer names
with open("C:\\Users\\000\\Desktop\\Capston\\yolo\\coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()

# load the image
img = cv2.imread("C:\\Users\\000\\Desktop\\Capston\\video,image\\c4.png")
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Load target image
target_img = cv2.imread("C:\\Users\\000\\Desktop\\Capston\\video,image\\p4_2.png")

# Calculate average RGB value of target image
avg_target_color_per_row = np.average(target_img, axis=0)
avg_target_color = np.average(avg_target_color_per_row, axis=0)

print(f"Average RGB value of target image: {avg_target_color}")

# create bounding box
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            if classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

# apply non-max suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# draw bounding boxes and label for each object
min_mse = float('inf')
min_mse_box = None
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        # calculate average color inside bounding box
        avg_color_per_row = np.average(img[y:y+int(h/2), x:x+w], axis=0)  # calculate for top 50% of bounding box
        avg_color = np.average(avg_color_per_row, axis=0)
        # compare with target color
        mse = np.sum((avg_target_color - avg_color) ** 2)
        if mse < min_mse:
            min_mse = mse
            min_mse_box = boxes[i]


x, y, w, h = min_mse_box
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the target image and the result image
cv2.imshow('Target Image', target_img)
cv2.imshow('Result Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
