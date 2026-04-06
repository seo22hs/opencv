# 1

# import cv2 as cv
# import numpy as np

# # 1단계: 이미지 로드
# img = cv.imread('./img/road.png')
# #print(f"원본 이미지 크기 :{img.shape}") # 실제 크기 확인 

# scale = 0.5
# img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
# print(f"축소된 이미지 크기 :{img_resized.shape}") # 실제 크기 확인 
# gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# # 2단계: Canny 에지 검출 (day04.md 참고)
# edges = cv.Canny(gray, 100, 200, apertureSize=3)

# # 3단계: 허프 직선 변환
# #lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
# lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=150, maxLineGap=5)

# # 4단계: 검출된 직선을 원본 이미지에 그리기
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv.imshow('Original', gray)
# cv.imshow('Edges', edges)
# cv.imshow('Hough Lines', img_resized)
# cv.waitKey(0)
# cv.destroyAllWindows()



# 2

import cv2 as cv
import numpy as np

# 1단계: 이미지 로드
img = cv.imread('./img/road_ex.png')

scale = 0.5
img_resized = cv.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)

# 2단계: Canny
edges = cv.Canny(gray, 100, 200, apertureSize=3)

# 🔥 3단계: ROI 적용 (하단만)
height, width = edges.shape
roi = edges[int(height*0.5):, :]

# 4단계: 허프 변환 (ROI에만 적용)
lines = cv.HoughLinesP(roi, 1, np.pi/180,
                       threshold=50,
                       minLineLength=150,
                       maxLineGap=5)

# 5단계: 직선 그리기 (좌표 보정 필수)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 🔥 ROI 기준 좌표 → 원본 좌표로 복구
        y1 += int(height*0.5)
        y2 += int(height*0.5)

        cv.line(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv.imshow('Original', gray)
cv.imshow('Edges', edges)
cv.imshow('ROI', roi)
cv.imshow('Hough Lines', img_resized)

cv.waitKey(0)
cv.destroyAllWindows()