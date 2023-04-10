import cv2
import numpy as np

config_file = "C:\\Users\\000\\Desktop\\Capston\\yolov3.cfg.txt"
weights_file = "C:\\Users\\000\\Desktop\\Capston\\yolov3.weights"
class_names_file = "C:\\Users\\000\\Desktop\\Capston\\coco.names.txt"

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)

with open(class_names_file, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

video_file = "C:\\Users\\000\\Desktop\\Capston\\video\\video14.mp4"
cap = cv2.VideoCapture(video_file)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_delay = int(1000 / fps)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    outputs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:
                center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype('int')
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, width, height = box
            if class_ids[i] == 0:
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, f"Person: {confidences[i]:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Output", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()