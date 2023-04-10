import cv2
import time

# 객체 추적할 영상 불러오기
video = cv2.VideoCapture("C:\\Users\\000\\Desktop\\Capston\\video\\video14.mp4")

# 초기 추적 경계 상자 설정
initBB = None

# KCF 추적기 초기화
tracker = cv2.TrackerKCF_create()

# 이전 박스의 중심좌표와 현재 박스의 중심좌표 비교를 위한 변수
prev_center = None

# 경로 출력 시간 간격 설정
path_interval = 1.0
last_path_time = time.time()

while True:
    # 영상 프레임 단위로 읽어오기
    ret, frame = video.read()

    if ret:
        # 추적 시작
        if initBB is None:
            # 추적할 객체 선택
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

            # 추적 시작
            tracker.init(frame, initBB)

        # 추적 중
        else:
            # 추적 경계 상자 업데이트
            ok, bbox = tracker.update(frame)

            # 추적 성공
            if ok:
                # 추적 경계 상자 그리기
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 현재 박스의 중심좌표 계산
                center = (int(x + w/2), int(y + h/2))

                # 이전 박스의 중심좌표와 현재 박스의 중심좌표 비교
                if prev_center is not None:
                    # 이동 방향 계산
                    dx = center[0] - prev_center[0]
                    dy = center[1] - prev_center[1]

                    # 경로 출력 시간 간격 체크
                    current_time = time.time()
                    if current_time - last_path_time > path_interval:
                        # 이동 방향에 따라 문구 출력
                        if dx > 0:
                            print("오른쪽으로 이동 중")
                        elif dx < 0:
                            print("왼쪽으로 이동 중")
                        if dy > 0:
                            print("아래쪽으로 이동 중")
                        elif dy < 0:
                            print("위쪽으로 이동 중")
                        last_path_time = current_time

                # 이전 박스의 중심좌표 갱신
                prev_center = center

            # 추적 실패
            else:
                initBB = None
                prev_center = None

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