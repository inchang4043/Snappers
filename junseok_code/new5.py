# RGB값 비교후 YOLO로 탐지, 추적

import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("C:\\Users\\000\\Desktop\\Capston\\yolo\\yolov4-tiny.weights", "C:\\Users\\000\\Desktop\\Capston\\yolo\\yolov4-tiny.cfg.txt")

# Load names of classes and get the output layer names
with open("C:\\Users\\000\\Desktop\\Capston\\yolo\\coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()

# Load target image and calculate its average color
target_img = cv2.imread("C:\\Users\\000\\Desktop\\Capston\\video,image\\p5.jpg")
avg_target_color = cv2.mean(target_img)[:3]

# Open video file
video = cv2.VideoCapture("C:\\Users\\000\\Desktop\\Capston\\video,image\\v2.mp4")
fps = video.get(cv2.CAP_PROP_FPS) # get fps of the video
detect_limit = int(fps * 10) # detection limit to first 0.3 seconds
frame_count = 0

while True:
    ret, img = video.read()
    if not ret:
        break

    height, width, channels = img.shape

    # Detecting objects only for the first 'detect_limit' frames
    if frame_count < detect_limit:
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Create bounding box
        class_ids = []
        confidences = []
        boxes = []
        colors = []
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

                        # Crop the bounding box from img and get mean color
                        roi = img[y:y + h, x:x + w]
                        mean_color = cv2.mean(roi)[:3]  # we only need B,G,R values, not the alpha value
                        colors.append(mean_color)

        # Apply non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Find the bounding box that has the most similar color to the target image
        min_mse = float('inf')
        min_mse_index = -1
        for i in range(len(boxes)):
            if i in indexes:
                # Calculate the Mean Squared Error between the target color and the color of the bounding box
                mse = np.mean((np.array(avg_target_color) - np.array(colors[i]))**2)
                if mse < min_mse:
                    min_mse = mse
                    min_mse_index = i

        # Draw the bounding box that has the most similar color to the target image
        if min_mse_index != -1:
            x, y, w, h = boxes[min_mse_index]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result frame
    cv2.imshow('Result Video', img)
    cv2.imshow('Target Image', target_img)

    if cv2.waitKey(1) == 27: # exit if Escape is hit
        break

    time.sleep(1/fps) # slow down to match the actual video speed
    frame_count += 1

video.release()
cv2.destroyAllWindows()
