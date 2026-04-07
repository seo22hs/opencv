import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# ========== Step 1: 이미지 로드 ==========
img1 = cv.imread(get_sample('box.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(get_sample('box_in_scene.png'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

# ========== Step 2: 특징점 검출기 초기화 ==========
# ORB 사용
orb = cv.ORB_create()
detector_type = "ORB"

# ========== Step 3: 키포인트와 디스크립터 추출 ==========
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========
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

# ========== Step 6: Lowe's ratio ==========
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