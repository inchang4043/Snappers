import cv2
import time

# 객체 추적기 생성
tracker = cv2.TrackerCSRT_create()

# 비디오 파일 열기
cap = cv2.VideoCapture(0)


# 추적할 객체 선택
ret, frame = cap.read()
bbox = cv2.selectROI(frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()
    if ret:
        # 객체 추적 수행
        ok, bbox = tracker.update(frame)

        # 추적 결과 시각화
        if ok:
            # 추적 성공
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2, 1)
        else:
            # 추적 실패
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # 결과 영상 출력
        cv2.imshow('Object Tracking', frame)

        # ESC 키를 누르면 프로그램 종료
        if cv2.waitKey(1) == 27:
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
