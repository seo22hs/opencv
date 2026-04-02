import cv2 as cv
import numpy as np

def find_license_plate(img):
    height, width = img.shape[:2]

    # ============================================================
    # 1️⃣ 전처리
    # ============================================================
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)

    # ============================================================
    # 2️⃣ 에지 검출 + 모폴로지
    # ============================================================
    edges = cv.Canny(gray, 50, 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    # ============================================================
    # 3️⃣ 컨투어 검출 + 필터링
    # ============================================================
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    plate_candidates = []

    for cnt in contours:
        area = cv.contourArea(cnt)

        if area > 1000:  # 너무 작은 영역 제거
            x, y, w, h = cv.boundingRect(cnt)
            aspect_ratio = w / h

            # 번호판 비율 조건
            if 3 < aspect_ratio < 6:
                plate_candidates.append((x, y, w, h, area))

    # ============================================================
    # 4️⃣ 번호판 선택 + 원근 변환
    # ============================================================
    if plate_candidates:
        plate_candidates.sort(key=lambda x: x[4], reverse=True)
        x, y, w, h, _ = plate_candidates[0]

        pts = np.float32([
            [x, y],
            [x + w, y],
            [x, y + h],
            [x + w, y + h]
        ])

        dst_pts = np.float32([
            [0, 0],
            [w, 0],
            [0, h],
            [w, h]
        ])

        M = cv.getPerspectiveTransform(pts, dst_pts)
        plate = cv.warpPerspective(img, M, (w, h))

        return plate, (x, y, w, h)

    return None, None


# ============================================================
# 메인 실행
# ============================================================

img = cv.imread('img/num_plate.jpg')

if img is None:
    print("❌ 이미지를 불러올 수 없습니다. 경로 확인하세요.")
    exit()

plate, rect = find_license_plate(img)

if plate is not None:
    x, y, w, h = rect

    result = img.copy()
    cv.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv.putText(result, 'License Plate', (x, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plate_resized = cv.resize(plate, (200, 100))
    result_resized = cv.resize(result, (800, 600))

    cv.imshow('Original with Detection', result_resized)
    cv.imshow('Extracted Plate', plate_resized)

    cv.imwrite('license_plate_extracted.png', plate)

    print("✅ 번호판 추출 완료")

    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    print("❌ 번호판을 찾을 수 없습니다.")