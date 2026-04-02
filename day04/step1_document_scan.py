import cv2 as cv
import numpy as np

# ============================================================
# 전역 변수
# ============================================================
win_name = "Document Scanning"
img = None
draw = None
rows, cols = 0, 0
pts_cnt = 0
pts = np.zeros((4, 2), dtype=np.float32)

# ============================================================
# 마우스 콜백 함수
# ============================================================
def onMouse(event, x, y, flags, param):
    global pts_cnt, draw, pts, img

    if event == cv.EVENT_LBUTTONDOWN:
        # 1️⃣ 클릭 위치 표시
        cv.circle(draw, (x, y), 5, (0, 255, 0), -1)
        cv.imshow(win_name, draw)

        # 2️⃣ 좌표 저장
        pts[pts_cnt] = [x, y]
        pts_cnt += 1

        # 3️⃣ 4개 점 모이면 실행
        if pts_cnt == 4:
            # 좌표 정렬
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            topLeft = pts[np.argmin(s)]
            bottomRight = pts[np.argmax(s)]
            topRight = pts[np.argmin(diff)]
            bottomLeft = pts[np.argmax(diff)]

            pts1 = np.float32([topLeft, topRight, bottomRight, bottomLeft])

            # 변환 후 크기 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])
            w2 = abs(topRight[0] - topLeft[0])
            width = int(max(w1, w2))

            h1 = abs(topRight[1] - bottomRight[1])
            h2 = abs(topLeft[1] - bottomLeft[1])
            height = int(max(h1, h2))

            # 변환 후 좌표
            pts2 = np.float32([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ])

            # 원근 변환
            mtrx = cv.getPerspectiveTransform(pts1, pts2)
            result = cv.warpPerspective(img, mtrx, (width, height))

            cv.imshow("Scanned Document", result)

            # 초기화
            pts_cnt = 0
            draw = img.copy()

# ============================================================
# 실행 모드 선택
# ============================================================
mode = input("모드 선택 (1: 이미지 / 2: 웹캠) → ")

# ============================================================
# 1️⃣ 이미지 모드
# ============================================================
if mode == '1':
    img = cv.imread('paper.jpg')  # 같은 폴더에 이미지 넣기

    if img is None:
        print("❌ 이미지를 불러올 수 없습니다.")
        exit()

    rows, cols = img.shape[:2]
    draw = img.copy()

    cv.imshow(win_name, draw)
    cv.setMouseCallback(win_name, onMouse)

    print("📝 사용법:")
    print("1. 문서의 4개 모서리를 클릭하세요 (순서 상관 없음)")
    print("2. 자동으로 스캔됩니다.")

    cv.waitKey(0)
    cv.destroyAllWindows()

# ============================================================
# 2️⃣ 웹캠 모드
# ============================================================
elif mode == '2':
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다.")
        exit()

    print("📝 사용법:")
    print("1. 화면에서 문서의 4개 점 클릭")
    print("2. 'q' 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.resize(frame, (800, 600))

        img = frame.copy()
        draw = frame.copy()

        cv.imshow(win_name, draw)
        cv.setMouseCallback(win_name, onMouse)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

else:
    print("❌ 잘못된 입력입니다.")