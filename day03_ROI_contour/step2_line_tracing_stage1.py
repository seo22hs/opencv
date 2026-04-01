import cv2 as cv
import numpy as np

# 웹캠 연결
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다")
    exit()

cv.namedWindow('Line Tracing Stage 1', cv.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 그레이스케일
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 2. 이진화 (자동 threshold)
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # 3. 컨투어 찾기
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 디버그 1: 검출된 전체 contour 개수
    print(f"\n[프레임 분석] 검출된 contour 개수: {len(contours)}")

    # 4. 가장 큰 컨투어 찾기
    largest_cnt = None
    max_area = 0

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            largest_cnt = cnt

    # 5. 중심 계산
    if largest_cnt is not None and max_area > 100:
        M = cv.moments(largest_cnt)

        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 컨투어
            cv.drawContours(frame, [largest_cnt], 0, (0, 255, 0), 2)

            # 중심점 (빨강)
            cv.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

            # 좌표 표시
            cv.putText(frame, f'Center: ({cx}, {cy})', (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 6. 화면 표시
    binary_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    result = np.hstack([binary_color, frame])

    cv.imshow('Line Tracing Stage 1', result)

    # 종료
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()