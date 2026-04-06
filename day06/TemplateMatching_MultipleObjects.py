# 1

# import cv2 as cv
# import numpy as np

# 이미지 읽기
# img = cv.imread('./img/mario.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# template = cv.imread('./img/coin.png', cv.IMREAD_GRAYSCALE)

# if img is None or template is None:
#     print("이미지 로드 실패")
#     exit()

# w, h = template.shape[::-1]

# # 템플릿 매칭
# res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# # 🔥 핵심: threshold
# threshold = 0.8
# loc = np.where(res >= threshold)

# # 🔥 여러 개 좌표 반복
# for pt in zip(*loc[::-1]):
#     cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# # 결과 출력
# cv.imshow('result', img)
# cv.waitKey(0)
# cv.destroyAllWindows()


#2

import cv2 as cv
import numpy as np

# 이미지 읽기
img = cv.imread('./img/mario.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

template = cv.imread('./img/coin.png', cv.IMREAD_GRAYSCALE)

if img is None or template is None:
    print("이미지 로드 실패")
    exit()

w, h = template.shape[::-1]

# 템플릿 매칭
res = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)

# 🔥 핵심: threshold
threshold = 0.8
loc = np.where(res >= threshold)

# 🔥 여러 개 좌표 반복
for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# 결과 출력
cv.imshow('result', img)
cv.waitKey(0)
cv.destroyAllWindows()