# EmbeddingGemma 2 Jetson 실습 1~7단계 (AIGJ_01_010 박수연)

Jetson Orin Nano Super 8GB에서 Google의 멀티모달 임베딩 모델 EmbeddingGemma 2를 설치하고, 텍스트·이미지·영상을 하나의 벡터 공간에서 검색한 실습입니다.

- 8~10단계(웹캠 물체 검색, 동작 인식, 카메라 2대 근접 알람)는 [practice_task_for_jetson_camera_applications-AIGJ_01_010](https://github.com/AIP-1/practice_task_for_jetson_camera_applications-AIGJ_01_010) 저장소에 있습니다.
- 실습 안내문은 [docs/guide_README.md](docs/guide_README.md)에 있습니다.
- 공유 보드 A·B에서 팀원과 순서를 맞춰 진행했습니다. 3단계는 보드 A, 4·6단계는 보드 B에서 측정했습니다.

## 환경

| 항목 | 값 |
| --- | --- |
| 보드 | Jetson Orin Nano 8GB, L4T R39 (JetPack 7 계열), Ubuntu 24.04 |
| 전원·메모리 | MAXN_SUPER (이 보드는 nvpmodel 2번), 스왑 8GB |
| 파이썬 | Python 3.12, venv `~/eg2` |
| PyTorch | 2.14.1+cu132 (공식 cu132 인덱스), GPU (8, 7), bf16 지원 |
| 라이브러리 | sentence-transformers 6.2.0, opencv-python-headless |
| 모델 | google/embeddinggemma-2, bfloat16 |

## 진행 단계

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 1 | 환경 준비 (전원 모드, 스왑, venv, GPU용 PyTorch) | 완료 |
| 2 | 패키지·모델 설치, 공통 로더 `eg2.py` | 완료 |
| 3 | 한국어 텍스트 검색, 차원(768/256/128) 비교 | 완료 |
| 4 | 이미지·동영상 임베딩과 텍스트로 검색 | 완료 (녹화·디코딩은 OpenCV로 대체) |
| 5 | 음성 임베딩 | 생략 (안내문상 다른 단계의 전제가 아닌 선택 단계) |
| 6 | 구성별 RAM·지연·GPU 측정 | 완료 (오디오·전체 구성은 5단계 생략으로 미측정) |
| 7 | 도메인 미세 조정 | 생략 (텍스트 전용 갈래, 카메라 응용 갈래로 진행) |

## 파일 구성

```
check.py         설치 확인 (torch가 CUDA 버전인지)
eg2.py           공통 로더 load(vision=False, audio=False): 안 쓰는 인코더는 만들지 않음
test_text.py     3단계 텍스트 검색, 차원별 비교
record.py        4단계 테스트 영상 녹화 (ffmpeg 대신 OpenCV)
test_vision.py   4단계 이미지 검색 (동영상 줄은 test_video.py로 옮김)
test_video.py    4단계 동영상 임베딩 (파일 대신 프레임 배열로 입력)
docs/            실습 안내문
```

## 실행

```bash
source ~/eg2/bin/activate && mkdir -p ~/eg2-lab/media && cd ~/eg2-lab
python check.py
python -c "from eg2 import load; m = load(); print(m.encode('test').shape)"   # (768,)
python test_text.py
python record.py && python test_vision.py && python test_video.py
```

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

## 트러블슈팅

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

## 안내문과 다르게 한 점

- torch는 안내문 기본(JetPack 6, cu126) 대신 JetPack 7용 cu132 인덱스로 설치하고 torchaudio는 제거했다.
- 전원 모드는 `nvpmodel -m 2` (이 보드의 MAXN_SUPER 번호).
- 4단계 녹화와 동영상 입력은 ffmpeg·torchcodec 대신 OpenCV (`record.py`, `test_video.py`).
- 테스트 영상·이미지(media/)와 모델 파일은 저장소에 넣지 않았다.
