## 바운딩 박스 화면상 좌표 출력

import cv2
import time

# 객체 추적기 생성
tracker = cv2.TrackerCSRT_create()

# 비디오 파일 열기
cap = cv2.VideoCapture("C:/Users/LJS/Desktop/Capston/video/video3.mp4")
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
            # 중심좌표 계산
            cx = int(x + w/2) - int(frame.shape[1]/2)
            cy = int(y + h/2) - int(frame.shape[0]/2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
            cv2.putText(frame, f"({cx}, {cy})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 추적 실패
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 결과 영상 출력
        cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
