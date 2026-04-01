import numpy as np
import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('Trackbars')

cv.createTrackbar('H_min', 'Trackbars', 35, 179, nothing)
cv.createTrackbar('H_max', 'Trackbars', 85, 179, nothing)
cv.createTrackbar('S_min', 'Trackbars', 50, 255, nothing)
cv.createTrackbar('S_max', 'Trackbars', 255, 255, nothing)
cv.createTrackbar('V_min', 'Trackbars', 50, 255, nothing)
cv.createTrackbar('V_max', 'Trackbars', 255, 255, nothing)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    h_min = cv.getTrackbarPos('H_min', 'Trackbars')
    h_max = cv.getTrackbarPos('H_max', 'Trackbars')
    s_min = cv.getTrackbarPos('S_min', 'Trackbars')
    s_max = cv.getTrackbarPos('S_max', 'Trackbars')
    v_min = cv.getTrackbarPos('V_min', 'Trackbars')
    v_max = cv.getTrackbarPos('V_max', 'Trackbars')

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # 디버깅 출력
    print(f"H:{h_min}-{h_max} S:{s_min}-{s_max} V:{v_min}-{v_max}")

    # 한 화면 출력
    combined = np.hstack((frame, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), result))
    cv.imshow('Result', combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()