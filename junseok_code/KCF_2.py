import cv2

def main():
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture("C:\\Users\\000\\Desktop\\Capston\\video\\video14.mp4")

    # 캡처가 열렸는지 확인
    if not cap.isOpened():
        print("Error: Unable to open video capture.")
        return

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read video frame.")
        return

    # 추적할 객체를 선택하고 ROI 설정
    roi = cv2.selectROI("Select Object to Track", frame, False, False)
    cv2.destroyAllWindows()

    # KCF 추적기 생성 및 초기화
    tracker = cv2.TrackerKCF_create()
    ret = tracker.init(frame, roi)

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read video frame.")
            break

        # 객체 추적 및 ROI 업데이트
        ret, roi = tracker.update(frame)

        if ret:
            # 추적 성공: ROI에 직사각형 그리기
            x, y, w, h = tuple(map(int, roi))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # 추적 실패: 메시지 표시
            cv2.putText(frame, "Lost object!", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 프레임 표시
        cv2.imshow("Object Tracker", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
