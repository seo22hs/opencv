import cv2 as cv
import numpy as np

# ============================================================
# 이미지 로드
# ============================================================
img = cv.imread('img/moon_gray.jpg')

if img is None:
    print("❌ 이미지를 불러올 수 없습니다.")
    exit()

# ============================================================
# 0. 그레이스케일 변환
# ============================================================
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ============================================================
# 1. Canny 에지 검출
# ============================================================
threshold1 = 50
threshold2 = 150

edges = cv.Canny(gray, threshold1, threshold2)

# ============================================================
# 2. 모폴로지 연산 — Opening + Closing
# ============================================================
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# 노이즈 제거
edges_cleaned = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)

# 끊긴 에지 연결
edges_closed = cv.morphologyEx(edges_cleaned, cv.MORPH_CLOSE, kernel)

# ============================================================
# 3. 결과 시각화
# ============================================================
# 흑백 → 컬러 변환 (붙이기 위해)
canny_color = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
cleaned_color = cv.cvtColor(edges_cleaned, cv.COLOR_GRAY2BGR)
closed_color = cv.cvtColor(edges_closed, cv.COLOR_GRAY2BGR)

# 위쪽: 원본 + Canny
top_row = np.hstack([img, canny_color])

# 아래쪽: Opening + Closing
bottom_row = np.hstack([cleaned_color, closed_color])

# 전체 합치기
result = np.vstack([top_row, bottom_row])

# ============================================================
# 출력
# ============================================================
cv.imshow('Edge Detection + Morphology', result)
cv.waitKey(0)
cv.destroyAllWindows()