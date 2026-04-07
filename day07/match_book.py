import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# ========== Step 1: 이미지 로드 ==========
img1 = cv.imread(('./img/book1.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(('./img/book2.jpeg'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

# ========== Step 2: 특징점 검출기 초기화 ==========
# SIFT 사용
sift = cv.SIFT_create()
detector_type = "SIFT"

# ========== Step 3: 키포인트와 디스크립터 추출 ==========
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========
if detector_type == "SIFT":
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

elif detector_type == "ORB":
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=12,
        key_size=20,
        multi_probe_level=2
    )
    search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========
matches = flann.knnMatch(des1, des2, k=2)

print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 ==========
good_matches = []

for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

print(f"Good matches after Lowe's ratio test: {len(good_matches)}")

# ========== Step 7: 시각화 ==========
def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    result = cv.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12,6))
    plt.title(title)
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.show()

if len(good_matches) >= 10:
    draw_matches(img1, kp1, img2, kp2, good_matches,
                 title=f"Good Matches ({len(good_matches)})")
else:
    print("Not enough good matches!")


# ========== 실습 1 코드 이후에 계속 ==========

MIN_MATCH_COUNT = 10

if len(good_matches) >= MIN_MATCH_COUNT:

    # ========== Step 1: 좌표 추출 ==========
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # ========== Step 2: 호모그래피 ==========
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if M is not None:

        # ========== Step 3: 원본 이미지 사각형 ==========
        h, w = img1.shape
        pts = np.float32([
            [0, 0],
            [0, h-1],
            [w-1, h-1],
            [w-1, 0]
        ]).reshape(-1, 1, 2)

        # ========== Step 4: 변환 ==========
        dst = cv.perspectiveTransform(pts, M)

        # ========== Step 5: 결과 시각화 ==========
        img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        result_img = img2.copy()

        # 🔥 핵심: 변환된 사각형 그리기
        result_img = cv.polylines(
            result_img,
            [np.int32(dst)],
            True,
            (255, 0, 0),  # 파란색
            3
        )

        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
        plt.title('Detected Object with Homography')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        # ========== Step 6: Inlier 매칭 시각화 ==========
        matchesMask = mask.ravel().tolist()

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        matched_img = cv.drawMatches(
            img1, kp1,
            img2, kp2,
            good_matches,
            None,
            **draw_params
        )

        plt.figure(figsize=(12, 6))
        plt.imshow(cv.cvtColor(matched_img, cv.COLOR_BGR2RGB))
        plt.title('Inlier Matches')
        plt.axis('off')
        plt.show()

        inlier_count = sum(matchesMask)
        outlier_count = len(matchesMask) - inlier_count

        print(f"Inliers: {inlier_count}, Outliers: {outlier_count}")

    else:
        print("Failed to compute homography")

else:
    print(f"Not enough matches ({len(good_matches)}/{MIN_MATCH_COUNT})")