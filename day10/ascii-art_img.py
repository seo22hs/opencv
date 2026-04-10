import cv2

# ASCII 문자 팔레트 — 왼쪽이 밝음, 오른쪽이 어두움
chars = ' .,-~:;=!*#$@'
#        밝음 ←————————→ 어두움

# 이미지 로드 → 그레이스케일
img = cv2.imread('peter.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 터미널 너비 100 문자 기준으로 리사이즈
# 터미널 글자는 세로가 더 길어서 가로를 2배로 설정
h, w = gray.shape
new_w = 100
new_h = int(h / w * new_w * 0.55)  # 0.55: 터미널 글자 종횡비 보정
resized = cv2.resize(gray, (new_w * 2, new_h))

# 핵심: 픽셀 밝기 → 문자 인덱스 변환
result = ''
for row in resized:
    for pixel in row:
        # 픽셀값 0~255 → chars 인덱스 0~12 변환
        idx = min(int(pixel / 256 * len(chars)), len(chars) - 1)
        result += chars[idx]
    result += '\n'

print(result)