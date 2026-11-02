# EmbeddingGemma 2 Jetson 실습 1~10단계 (AIGJ_01_010 박수연)

Jetson Orin Nano Super 8GB에서 멀티모달 임베딩 모델 EmbeddingGemma 2로 텍스트·이미지·영상을 한 벡터 공간에서 검색하고(1~7단계), USB 웹캠으로 물체 검색·동작 인식·카메라 2대 근접 알람을 만든 실습입니다(8~10단계).

![Jetson Orin Nano](https://img.shields.io/badge/Jetson-Orin_Nano_Super_8GB-green)
![JetPack 7](https://img.shields.io/badge/JetPack-7_%28L4T_R39%29-green)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcu132-orange)

- 카메라 응용 과제 제출본(과제 1~6 형식, 기능 시험 기록)은 [practice_task_for_jetson_camera_applications-AIGJ_01_010](https://github.com/AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_010) 저장소에 있습니다.
- 실습 안내문은 [docs/guide_README.md](docs/guide_README.md)에 있습니다.
- 공유 보드 A·B에서 팀원과 순서를 맞춰 진행했습니다. 3단계는 보드 A, 4·6단계는 보드 B에서 측정했습니다.
- 8~10단계는 보드 B에서 팀원(AIGJ_01_004)과 함께 진행했습니다. 코드와 측정 기록(`app.py`, `pose_*.py`, `train_pose.py`, `data/`, `prox/`, `results/`, `reports/`)은 같은 실험에서 나온 것으로, 팀 저장소 [practice_task_for_jetson_camera_applications-AIGJ_01_004](https://github.com/AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_004)와 같습니다.


---

## 결과 한눈에

| 단계 | 내용 | 핵심 결과 | 상태 |
| :---: | --- | --- | :---: |
| 1 | 환경 준비 | JetPack 7 계열, MAXN_SUPER, torch 2.14.1+cu132 (GPU) | ✅ |
| 2 | 패키지·모델 설치 | sentence-transformers 6.2.0, 공통 로더 `eg2.py` | ✅ |
| 3 | 한국어 텍스트 검색 | 768·256·128차원 모두 정답 문서 1위, 162~182 ms | ✅ |
| 4 | 글로 사진·영상 찾기 | 이미지 5장 2714 ms, 동영상 10초 4416 ms (OpenCV로 녹화·디코딩) | ✅ |
| 5 | 음성 임베딩 | 선택 단계라 생략 | ⏭️ 생략 |
| 6 | 성능 측정 | 텍스트 전용 +1385 MB, 비전 포함 +1727 MB | ✅ |
| 7 | 도메인 미세 조정 | 카메라 응용 갈래로 진행해 생략 | ⏭️ 생략 |
| 8 | 웹캠 물체 검색 | YOLOE + 임베딩 재라벨 조합, 13~17 fps | ✅ |
| 9 | 동작·행동 인식 | 동작 GRU 검증 0.989, 행동 패턴 12/12 | ✅ |
| 10 | 카메라 2대 근접 알람 | 앉은 사람 거리 오차 0.64 → 0.14 m, 알람 시나리오 5개 모두 3/3 | ✅ |

---

## 8~10단계 구조

```
USB 웹캠 → Jetson
  ├ 캡처 스레드: 프레임 → YOLO 계열 검출·추적 → 크롭 / 관절
  ├ 임베딩 스레드: EmbeddingGemma 2 → 메모리 인덱스 (512차원, 최대 1000개)
  └ FastAPI 웹 서버 (:8000) → 브라우저 (실시간 영상, 텍스트 검색, 알람 기록)
```

영상과 임베딩을 다른 스레드로 나누고 사이에 큐를 둬서, 임베딩이 느려도 영상은 15 fps로 계속 흐르게 했습니다. 사람 이미지는 크롭·저장하지 않고 좌표와 알람 기록만 남깁니다.

---

## 환경

| 항목 | 값 |
| --- | --- |
| 보드 | Jetson Orin Nano 8GB, L4T R39 (JetPack 7 계열), Ubuntu 24.04 |
| 전원·메모리 | MAXN_SUPER (이 보드는 nvpmodel 2번), 스왑 8GB |
| 파이썬 | Python 3.12, venv `~/eg2` |
| PyTorch | 2.14.1+cu132 (공식 cu132 인덱스), GPU (8, 7), bf16 지원 |
| 라이브러리 | sentence-transformers 6.2.0, opencv-python-headless, ultralytics 8.4.184 (8~10단계) |
| 모델 | google/embeddinggemma-2, bfloat16 / yolo11n, yoloe-11s-seg, yolo11n-pose (8~10단계) |
| 카메라 | Logitech C920 2대 (/dev/video0, /dev/video2), 자동초점 끔 |

---

## 파일 구성


```
check.py         설치 확인 (torch가 CUDA 버전인지)
eg2.py           공통 로더 load(vision=False, audio=False): 안 쓰는 인코더는 만들지 않음
test_text.py     3단계 텍스트 검색, 차원별 비교
record.py        4단계 테스트 영상 녹화 (ffmpeg 대신 OpenCV)
test_vision.py   4단계 이미지 검색 (동영상 줄은 test_video.py로 옮김)
test_video.py    4단계 동영상 임베딩 (파일 대신 프레임 배열로 입력)
app.py           8단계 물체 추출 + 웹 검색 서버 (--mode yolo|relabel|yoloe)
pose_app.py      9단계 동작·행동 인식 (--raw, --record, --behavior, --model)
pose_model.py    9단계 관절 특징(53개)과 GRU 모델
train_pose.py    9단계 학습 스크립트
data/action/     9단계 동작 학습 데이터 (관절 좌표 csv)
data/behavior/   9단계 행동 학습 데이터 (관절 좌표 csv)
prox/            10단계 두 카메라 근접 알람
  calib_intrinsics.py   렌즈 보정 (체커보드)
  calib_extrinsics.py   카메라 위치 보정 (기준점 클릭, solvePnP)
  stereo.py             삼각측량, 한 카메라 추정, 칼만 필터
  proximity_app.py      근접·이탈·충돌 알람 앱
  calib/                기준점 좌표, 클릭 좌표, 측정 기록
  check/                카메라 위치 변화 확인 이미지
results/         8, 9, 10단계 측정 결과
reports/         PDF 보고서 (9단계, 10단계, 1~10단계 전체)
docs/            실습 안내문
```

---

## 실행

```bash
source ~/eg2/bin/activate && mkdir -p ~/eg2-lab/media && cd ~/eg2-lab
python check.py
python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"   # (768,)
python test_text.py
python record.py && python test_vision.py && python test_video.py

python app.py --mode relabel                      # 8단계
python pose_app.py --behavior                     # 9단계 (규칙)
python pose_app.py --model models/action.pt       # 9단계 (학습 모델)
cd prox && python proximity_app.py --cams 0 2 --target "brown cardboard box"            # 10단계
python proximity_app.py --cams 0 2 --target "brown cardboard box" --moving --ttc 3.0     # 10-8 이동 타겟
# 브라우저: http://<Jetson IP>:8000
```


---

## 주요 결과

### 3단계 텍스트 검색 (보드 A)

| 차원 | 지연 | "졸리고 머리가 맑지 않아요" → CO2 문서 | "냉방 몇 도?" → 에어컨 문서 |
| --- | --- | --- | --- |
| 768 | 182 ms | 0.717 | 0.725 |
| 256 | 162 ms | 0.734 | 0.754 |
| 128 | 180 ms | 0.785 | 0.810 |

- 세 차원 모두 두 질의의 정답 문서를 1위로 찾았다.
- 지연이 거의 같은 이유는 시간 대부분이 모델이 문장을 읽는 데 쓰이기 때문이다. 차원 축소의 이득은 저장·검색할 벡터가 많을 때 나온다.

### 4단계 이미지·동영상 검색 (보드 B)

| 입력 | 지연 | 결과 |
| --- | --- | --- |
| 이미지 5장 | 2714 ms | 처음엔 벽 위주 장면이라 질의별 점수 차이 0.01~0.03 → 서로 다른 장면으로 다시 촬영 |
| 동영상 10초 (20프레임) | 4416 ms | 질의 유사도: 빈 강의실 0.642, 수업 0.593, 화이트보드 0.618 |

### 6단계 성능 측정 (보드 B)

RAM은 `free -m`의 used 값. 모델을 올리기 전 1868 MB, 보드 전체 7485 MB.

| 구성 | 로드 크기 | 로드 후 RAM (MB) | 입력 | 지연 (ms) | GPU 사용률 (%) |
| --- | --- | --- | --- | --- | --- |
| 텍스트 전용 | 270M | 3253 (모델 +1385) | 질의 2개 | 182 | 순간 최대 11 |
| +비전 | 440M | 3595 (모델 +1727) | 이미지 5장 | 2714 | - |
| +비전 | 440M | 3595 (모델 +1727) | 동영상 10초 | 4416 | - |
| +오디오 | 570M | 미측정 (5단계 생략) | - | - | - |
| 전체 | 740M | 미측정 (5단계 생략) | - | - | - |

- 비전 인코더를 켜면 342 MB가 더 든다. 그래서 필요한 인코더만 켜서 로드했다.
- 텍스트 전용 가중치는 bfloat16 기준 약 0.5 GB인데 실제로 약 1.4 GB를 쓴다. 나머지는 CUDA·PyTorch가 처음 켜질 때 잡는 기본 메모리다.
- GPU 사용률이 낮게 보이는 이유는 검색이 0.2초 안에 끝나서 1초 간격의 `tegrastats`에 거의 잡히지 않기 때문이다.

### 8단계 물체 추출: 세 방식 비교 (보드 B)

| 물체 | yolo | relabel | yoloe |
| --- | --- | --- | --- |
| 무선 이어폰 | remote / mouse ✗ | 무선 이어폰 ✓ | mouse ✗ |
| 드럼 세탁기 | microwave ✗ | 드럼 세탁기 ✓ | drum washing machine ✓ |
| 선풍기 | clock ✗ | 천장 에어컨 ✗ (목록 추가 전) | electric fan ✓ |
| 시계 | clock ✓ | 휴대폰 ✗ (목록 추가 전) | wall clock ✓ |
| **검출 fps** | 15 | 15~17 | 13 |

- YOLO(COCO 80종)는 모르는 물체도 80종 중 하나로 부른다(닫힌 어휘).
- 임베딩 재라벨은 학습 없이 목록만으로 바로잡지만, 목록 밖 물체는 오히려 틀린다.
- YOLOE는 글로 클래스를 정해 COCO 밖 물체도 찾지만, 작고 비슷한 물체는 헷갈린다.
- **선택: YOLOE로 후보를 찾고 헷갈리는 물체는 임베딩으로 확인하는 조합.** 10단계 타겟 찾기가 이 방식이다.

### 9단계 동작·행동 인식 (보드 B)

- 프레임 하나로는 손 흔들기를 판정할 수 없어, 0.6초 다수결과 1.5초 방향 전환 판정으로 해결했다.
- 동작 GRU(window 30, idle·wave·clap·arms_crossed) 검증 정확도 0.989.
- 실시간에서 "가만히 있어도 wave" 문제가 있었다. 얼굴을 빼고 녹화해 얼굴 특징이 항상 0이었고, 표준화에서 표준편차가 0에 가까워 값이 폭주한 것이 원인이었다. 표준편차 최소값을 0.03으로 두고 박수 데이터를 더 녹화해 4개 동작 모두 실시간 인식.
- 행동 패턴(지나감·힐끗 봄·관심 있음·머무름) 규칙은 4개 시나리오 × 3회 모두 기대한 결과. 행동 모델은 한 번 지나가는 시간이 짧아 window 15로 학습했다.

### 10단계 카메라 2대 근접 알람 (보드 B)

- 렌즈 보정 RMS 0.372 / 0.355 px. 위치 보정 재투영 오차 3.78 / 3.22 px, 검증점 오차 3.1 / 3.0 cm.

| 실제 거리·자세 | ankle1 (카메라 1대, 발목) | stereo (2대, 어깨 삼각측량) | fused |
| --- | --- | --- | --- |
| 0.5 m 서 있음 | 0.56 | 0.61 | 0.63 |
| 1.0 m 서 있음 | 1.18 | 1.06 | 1.21 |
| 2.0 m 서 있음 | 2.05 | 2.14 | 2.11 |
| 0.81 m 책상에 앉음 | 1.45 (ID 끊김) | **0.67** | 0.91 |
| fps | 26.7 | 23.1 | 23 |

- 앉으면 책상에 발목이 가려져 카메라 1대 방식이 실패했고, 책상 위로 보이는 어깨를 두 카메라로 삼각측량해 해결했다.
- 알람은 히스테리시스(0.2 m)로 경계에서 깜빡이지 않게 했다. 타겟 자동 찾기 후 알람 시나리오 5개 모두 3/3 성공.
- 이동 타겟은 칼만 필터로 속도를 추정하고 TTC(충돌 예상 시간)로 알람을 냈다. 충돌 위험 알람이 근접 알람보다 천천히 걸을 때 약 3초, 보통 속도일 때 약 1초 먼저 울렸다.

8~10단계의 자세한 측정값은 [results/](results/)와 [reports/](reports/)에 있습니다.

---

## 트러블슈팅
<details>
<summary>문제 · 원인 · 해결 18건 펼쳐 보기</summary>


| 단계 | 문제 | 원인 | 해결 |
| --- | --- | --- | --- |
| 1 | torch 설치 `from versions: none` | jp6 인덱스는 Python 3.10용, 보드는 3.12 (JetPack 7 계열) | JetPack 7용 공식 cu132 인덱스 사용 |
| 1 | `nvpmodel -m 0`이 최대 성능이 아님 | 이 보드는 0번이 15W, 2번이 MAXN_SUPER | `-q`로 확인 후 `-m 2` (재부팅되므로 팀원에게 먼저 알림) |
| 1 | 스왑 추가 오류, fstab 중복 줄 | 스왑 8GB가 이미 있었음 | 중복 줄 삭제 |
| 2 | 패키지 설치 중 GPU torch가 덮어써질 위험 | 의존성으로 PyPI torch가 설치됨 | `torch-pin.txt`를 만들고 모든 설치에 `pip install -c` |
| 2 | torchaudio CUDA 버전 충돌로 sentence-transformers가 안 불러와짐 | torch는 cu132, torchaudio는 PyPI cu130. transformers가 자동 import | `pip uninstall -y torchaudio` |
| 4 | ffmpeg로 웹캠 녹화 실패 | 보드 ffmpeg가 `--disable-everything` 축소 빌드 (v4l2 입력, x264·mpeg4 인코더 없음) | OpenCV로 녹화 (`record.py`). 공유 보드라 시스템 ffmpeg는 바꾸지 않음 |
| 4 | `{"video": 파일}` 디코딩 실패 | torchcodec이 필요한 FFmpeg 라이브러리 없음 | OpenCV로 초당 1프레임을 읽어 numpy 배열로 전달 (`test_video.py`) |
| 4 | 이미지 검색 점수가 다 비슷함 (~0.58) | 벽 위주로 찍어 장면 차이가 없음 | 서로 다른 물체·장면으로 다시 촬영 |
| 6 | GPU 사용률이 0%처럼 보임 | 작업이 측정 간격(1초)보다 빨리 끝남 | 순간 최대값 11% 확인 |
| 8 | 프레임마다 경고 | ultralytics 8.4.184에서 `half=True` 미지원 | `app.py`, `pose_app.py`, `proximity_app.py`에서 제거 |
| 8 | 이어폰 → remote, 에어컨 → microwave | COCO 80종 밖 물체 | 임베딩 재라벨, YOLOE |
| 9 | 만세가 한팔들기로 깜빡임 | 한쪽 손목이 화면 밖 | 카메라에서 물러나 두 손이 다 보이게 |
| 9 | 검증 98.9%인데 가만히 있어도 wave | 얼굴 특징이 학습 때 항상 0 → 표준편차 ≈ 0 → 정규화 폭주 | `std.clamp(min=0.03)` (0.1은 박수도 둔해짐) + 박수 추가 녹화 |
| 9 | 행동 모델 학습 윈도 0개 | 지나가는 데 1.5~2초, 창은 80프레임 | `--window 15` |
| 10 | 렌즈 보정 RMS가 1.4~3.5 px로 흔들림 | C920 자동초점 | `v4l2-ctl`로 자동초점 끔 (USB 다시 꽂으면 초기화) |
| 10 | 위치 보정 첫 계산 172 px | 실측표와 클릭에서 기준점 번호가 바뀜 | json에서 좌표 교환 후 해당 점만 재클릭 |
| 10 | 이동 타겟이 10~14 m 밖으로 튐 | 상자를 카메라 높이에 가깝게 들어 시선이 수평 | 상자를 무릎~허리 높이로 |
| 10 | 타겟이 바닥 아래·8 m 밖 | 보정 뒤 cam0이 약 6° 돌아감 | 보정 때 화면과 비교해 손으로 다시 맞춤 |

</details>

---

## 안내문과 다르게 한 점

- torch는 안내문 기본(JetPack 6, cu126) 대신 JetPack 7용 cu132 인덱스로 설치하고 torchaudio는 제거했다.
- 전원 모드는 `nvpmodel -m 2` (이 보드의 MAXN_SUPER 번호).
- 4단계 녹화와 동영상 입력은 ffmpeg·torchcodec 대신 OpenCV (`record.py`, `test_video.py`).
- 8~10단계: `half=True` 제거, 9단계 `train_pose.py` 표준편차 하한 `std.clamp(min=0.03)`.
- 10단계 카메라는 `--cams 0 2`, 렌즈 보정은 태블릿 체커보드(칸 22 mm, `--square 0.022`).
- 10단계 시나리오는 공간이 좁아 4 m 대신 3 m 자리에서 진행, 이동 타겟은 카트 대신 사람이 든 갈색 상자.

---

## 한계와 다음 개선 (8~10단계)

- 카메라 높이 1.35~1.41 m, 두 시선 각도 23°로 권장(2.2~2.5 m, 60~90°)보다 낮고 좁아 한 카메라 추정 오차가 컸다.
- 몸을 돌리거나 앉고 일어설 때 0~1초짜리 가짜 ID가 생겨 불필요한 근접 알람이 났다. 짧게 사는 ID를 알람에서 빼는 것이 다음 개선이다.
- 행동 모델은 검증 데이터가 적어 점수로 판단하기 어렵다.

테스트 영상·이미지(media/), 학습된 모델(.pt), 카메라 보정 결과(.npz)는 저장소에 넣지 않았다.
