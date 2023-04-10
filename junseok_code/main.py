import cv2

def main():
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture("C:\\Users\\000\\Desktop\\video1.mp4")  # 비디오 파일 경로를 입력하세요.

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while cap.isOpened():
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read video frame.")
            break

        # 프레임 표시
        cv2.imshow("Video Playback", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
