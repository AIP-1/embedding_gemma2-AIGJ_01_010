# 4단계 테스트 영상 녹화 (안내문의 ffmpeg 명령 대체)
# 보드의 ffmpeg는 축소 빌드(--disable-everything)라 v4l2 입력과 인코더가 없어서 OpenCV로 녹화한다.
# 10초 영상 media/clip.mp4와 2초마다 사진 media/frame_NN.jpg를 저장한다.
import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
assert cap.isOpened(), "카메라를 열 수 없어요"
out = cv2.VideoWriter("media/clip.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
print("녹화 시작! 10초 동안 여러 장면을 비춰 주세요")
t0, last, n = time.time(), -2, 1
while time.time() - t0 < 10:
    ok, frame = cap.read()
    if not ok: continue
    frame = cv2.resize(frame, (1280, 720))
    out.write(frame)
    if time.time() - last >= 2:
        cv2.imwrite(f"media/frame_{n:02d}.jpg", frame); n += 1; last = time.time()
cap.release(); out.release()
print("완료:", n - 1, "장 저장")
