import cv2


# 객체 추적할 영상 불러오기
video = cv2.VideoCapture("C:\\Users\\000\\Desktop\\Capston\\video\\video14.mp4")

# HOG 기반 보행자 검출기 초기화
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    # 영상 프레임 단위로 읽어오기
    ret, frame = video.read()

    if ret:
        # 보행자 검출 수행
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # 검출된 보행자 경계상자 그리기
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 영상 출력
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # 'q' 키를 누르면 종료
        if key == ord("q"):
            break
    else:
        break

# 메모리 해제
video.release()
cv2.destroyAllWindows()