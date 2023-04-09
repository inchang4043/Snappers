import cv2
import time

last_path_time = time.time()
# 객체 추적기 생성
tracker = cv2.TrackerCSRT_create()

# 비디오 파일 열기
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 'frame' 창 생성
cv2.namedWindow('frame')


# bbox 초기화 함수
def init_bbox(x, y):
    global bbox_tl, bbox_br
    bbox_tl = (x, y)
    bbox_br = (x, y)


# bbox 업데이트 함수
def update_bbox(x, y):
    global bbox_tl, bbox_br
    bbox_br = (x, y)


# bbox 추출 함수
def get_bbox():
    x = min(bbox_tl[0], bbox_br[0])
    y = min(bbox_tl[1], bbox_br[1])
    w = abs(bbox_tl[0] - bbox_br[0])
    h = abs(bbox_tl[1] - bbox_br[1])
    return (x, y, w, h)


# 마우스 이벤트 처리를 위한 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global bbox_tl, bbox_br, tracking, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        init_bbox(x, y)
        tracking = False
    elif event == cv2.EVENT_LBUTTONUP:
        update_bbox(x, y)
        bbox = get_bbox()
        if bbox[:2] != (0, 0):
            tracker.init(frame, bbox)
            tracking = True
            cv2.rectangle(frame, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
        else:
            bbox_tl = (0, 0)
            bbox_br = (0, 0)
    elif event == cv2.EVENT_MOUSEMOVE and not tracking:
        update_bbox(x, y)
        bbox = get_bbox()
        if bbox[:2] != (0, 0):
            cv2.rectangle(frame, bbox[:2], (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
    elif event == cv2.EVENT_RBUTTONDOWN:
        bbox_tl = (0, 0)
        bbox_br = (0, 0)
        tracking = False

# 마우스 이벤트 콜백 함수 등록
cv2.setMouseCallback('frame', mouse_callback)

bbox_tl = (0, 0)
bbox_br = (0, 0)
tracking = False

# 이전 박스의 중심좌표 변수 초기화
prev_center = None

while ret:

    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    ret, frame = cap.read()

    # ESC 키를 누르면 프로그램 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # 객체 추적 수행
    if tracking:
        ok, bbox = tracker.update(frame)

        # 추적 결과 시각화
        if ok:
            # 추적 성공
            x, y, w, h = [int(i) for i in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)

            # 현재 박스의 중심좌표 계산
            center = (int(x + w/2), int(y + h/2))

            # 이전 박스의 크기와 현재 박스의 크기 비교
            if prev_center is not None:
                prev_size = abs(prev_bbox[2] * prev_bbox[3])
                curr_size = abs(w * h)
                size_ratio = curr_size / prev_size
                if size_ratio > 1.01:
                    print("가까워지는 중")
                elif size_ratio < 0.09:
                    print("멀어지는 중")

                # 이전 박스의 중심좌표와 현재 박스의 중심좌표 비교
                dx =  (center[0] - prev_center[0])
                dy =  (center[1] - prev_center[1])

                # 경로 출력 시간 간격 체크
                current_time = time.time()
                path_interval = 1.0  # 경로 출력 시간 간격
                if current_time - last_path_time > path_interval:
                    # 이동 방향에 따라 문구 출력
                    if dx > 2:
                        print("오른쪽으로 이동 중")
                    elif dx < -2:
                        print("왼쪽으로 이동 중")
                    if dy > 2:
                        print("아래쪽으로 이동 중")
                    elif dy < -2:
                        print("위쪽으로 이동 중")
                    last_path_time = current_time

            # 이전 박스의 정보 갱신
            prev_center = center
            prev_bbox = bbox

        else:
            # 추적 실패
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 결과 영상 출력
        cv2.imshow('frame', frame)


cap.release()
cv2.destroyAllWindows()