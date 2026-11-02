# 4단계 동영상 임베딩 (안내문의 {"video": "media/clip.mp4"} 대체)
# 보드의 torchcodec이 FFmpeg 라이브러리(libavutil 등)를 찾지 못해 파일 디코딩이 실패한다.
# OpenCV로 초당 1프레임을 읽어 numpy 배열로 쌓아 모델에 직접 넘긴다.
import time
import cv2, numpy as np
from eg2 import load

model = load(vision=True)

cap = cv2.VideoCapture("media/clip.mp4")
step = int(round(cap.get(cv2.CAP_PROP_FPS) or 15))   # 초당 1프레임 (모델 기본 샘플링과 같게)
frames, i = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i % step == 0:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))   # OpenCV는 BGR이라 RGB로 변환
    i += 1
cap.release()
video = np.stack(frames)
print("프레임:", video.shape)

t = time.perf_counter()
vid_emb = model.encode({"video": video}, normalize_embeddings=True)
print(f"동영상 1개: {(time.perf_counter()-t)*1000:.0f} ms")

queries = ["사람이 없는 빈 강의실", "학생들이 앉아 수업을 듣는 모습", "화이트보드에 쓴 글씨"]
q = model.encode(queries, prompt_name="SearchQuery", normalize_embeddings=True)
print("질의-동영상 유사도:", (q @ vid_emb).round(3))
