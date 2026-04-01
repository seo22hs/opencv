import cv2 as cv
import numpy as np

img = cv.imread('img/coins_spread1.jpg')

if img is None:
    print("이미지 못 읽음")
    exit()

# 1. HSV 변환
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# 2. 은색 범위 설정
lower = np.array([0, 0, 120])
upper = np.array([180, 50, 255])

mask = cv.inRange(hsv, lower, upper)

# 3. 노이즈 제거 (선택)
kernel = np.ones((3,3), np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

# 4. 컨투어 찾기
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

print("은색 컨투어:", len(contours))

# 5. 결과 표시
result = img.copy()
cv.drawContours(result, contours, -1, (0,255,0), 2)

cv.imshow("mask", mask)
cv.imshow("result", result)
cv.waitKey(0)
cv.destroyAllWindows()