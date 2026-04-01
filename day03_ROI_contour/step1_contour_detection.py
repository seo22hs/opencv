import cv2 as cv
import numpy as np

# 1. 이미지 읽기 (그레이스케일)
img = cv.imread('img/shapes.png', cv.IMREAD_GRAYSCALE)

if img is None:
    print("이미지 못 읽음")
    exit()

# 2. 이진화
ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

# 3. 컨투어 찾기
contours, hierarchy = cv.findContours(
    binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

print("전체 컨투어 수:", len(contours))

# 4. 컬러 이미지로 변환 (그리기용)
img_color = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

# 5. 모든 컨투어 (초록색)
cv.drawContours(img_color, contours, -1, (0,255,0), 2)

# 6. 면적 필터링 (파란색)
count = 0 

for cnt in contours:
    area = cv.contourArea(cnt)

    if 100 < area < 5000:
        count += 1
        cv.drawContours(img_color, [cnt], 0, (255,0,0), 2)

print("필터링된 컨투어:", count)

# 7. 출력
cv.imshow('Filtered Contours', img_color)
cv.waitKey(0)
cv.destroyAllWindows()