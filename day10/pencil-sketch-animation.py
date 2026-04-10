import cv2
import time

# ── 설정 ──────────────────────────────────────────────────

# pencilSketch 파라미터 — 여기를 바꿔보세요
SIGMA_S      = 60     # 공간 범위: 20 / 60 / 150 으로 바꿔보기
SIGMA_R      = 0.05   # 색상 범위: 0.01 / 0.05 / 0.3 으로 바꿔보기
SHADE_FACTOR = 0.015  # 음영 강도: 0.005 / 0.015 / 0.05 으로 바꿔보기

# ── 실행 함수 ────────────────────────────────────────────
def run_webcam():
    # 웹캠에서 한 프레임 캡처
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[오류] 웹캠 프레임을 읽을 수 없습니다.")
        return

    print("웹캠에서 한 프레임 캡처 후 연필 스케치 변환...")

    # 잡티 제거용 블러
    frame = cv2.GaussianBlur(frame, (9, 9), 0)

    # 연필 스케치 변환
    gray_sketch, color_sketch = cv2.pencilSketch(
        frame,
        sigma_s=SIGMA_S,
        sigma_r=SIGMA_R,
        shade_factor=SHADE_FACTOR
    )

    # 그레이스케일 저장
    fname_gray = f'sketch_gray_{int(time.time())}.png'
    cv2.imwrite(fname_gray, gray_sketch)
    print(f"[저장] {fname_gray}")

    # 컬러 저장
    fname_color = f'sketch_color_{int(time.time())}.png'
    cv2.imwrite(fname_color, color_sketch)
    print(f"[저장] {fname_color}")


# ── 실행 ──────────────────────────────────────────────────
if __name__ == '__main__':
    run_webcam()