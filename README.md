# lip-sync-score

Audio-driven lip-sync 생성 모델(ADLip2)의 립싱크 동기화 평가를 위한 커스텀 SyncNet 연구 프로젝트입니다.

기존 SyncNet 계열 모델은 얼굴 전체 단위로 동기화를 판단해 립싱크 품질을 정밀하게 반영하지 못합니다. 이 프로젝트는 입술 영역에 특화된 시간적(temporal) 동기화 모델 **SyncNetTemporal (SyncLT)** 을 개발하고, 이를 [ADLip2](https://github.com/weonjungin/ADLip2)의 립싱크 손실 함수 설계에 활용했습니다.

## 배경

이 프로젝트는 [avsync_project](https://github.com/weonjungin/avsync_project)에서 시작되었습니다. 초기에는 AVSpeech 데이터셋으로 SyncNet을 학습했으나, 유튜브 다운로드 정책 변경으로 데이터 수급이 중단되어 GRID / HDTF 데이터셋 기반으로 전환했습니다.

## 핵심 실험 결과

세 가지 조합으로 SyncNet을 학습하고 비교했습니다.

| 실험 | 데이터셋 | Loss | 결과 |
|---|---|---|---|
| `train_grid_marginranking` | GRID | Margin Ranking | GRID 자체 평가는 양호했으나, HDTF로 교차 평가 시 성능 급락 → **과적합 확인** |
| `train_grid_contrastive` | GRID | Contrastive | 위와 동일하게 과적합 확인 |
| `train_hdtf_contrastive` | HDTF | Contrastive | **최종 채택**. ADLip2의 SyncLT 손실 함수로 사용됨 |

GRID는 화자 수가 적고 발화 스크립트가 제한적인 데이터셋이라, 단일 데이터셋 학습만으로는 일반화 성능이 떨어진다는 것을 위 실험으로 확인했습니다. `train_grid_*` 결과 폴더의 `eval_curve.csv`, `eval_bad_utts.csv`는 GRID로 학습한 모델을 HDTF 발화로 교차 평가한 결과이며, 파일명의 `hdtf`는 평가 대상 데이터셋을 의미합니다. 이후 HDTF 데이터셋 하나로 학습 및 평가를 통일한 `train_hdtf_contrastive`가 최종 모델입니다.

## 레포 구조

    lip-sync-score/
    ├── configs/
    │   ├── prepare_grid_lss.yaml          # GRID 전처리 설정
    │   ├── prepare_hdtf.yaml              # HDTF 전처리 설정
    │   ├── train_grid_marginranking.yaml  # GRID + Margin Ranking (과적합 실험)
    │   ├── train_grid_contrastive.yaml    # GRID + Contrastive (과적합 실험)
    │   └── train_hdtf_contrastive.yaml    # HDTF + Contrastive (최종 모델)
    │
    ├── scripts/
    │   ├── prepare_grid.py       # GRID 원본 → 얼굴/입술 crop, mel/mfcc 추출
    │   ├── prepare_hdtf.py       # HDTF 원본 → 얼굴/입술 crop, mel/mfcc 추출
    │   ├── preprocess_latent.py  # HDTF 프레임 → VAE latent 변환 (ADLip2 연동용)
    │   ├── train_grid.py         # GRID 학습 (SyncNetLike baseline / SyncNetTemporal)
    │   ├── train_hdtf.py         # HDTF 학습
    │   ├── eval_grid.py          # GRID 학습 모델을 HDTF로 교차 평가
    │   └── eval_hdtf.py          # HDTF 학습 모델 평가
    │
    ├── src/lipsyncscore/
    │   ├── models/
    │   │   ├── syncnet_like.py           # 원본 SyncNet 구조 재현 (baseline)
    │   │   ├── audio_encoder.py, lip_encoder.py
    │   │   ├── baselines/
    │   │   │   └── syncnet_python_wrapper.py  # 공개 SyncNet 구현체 래퍼 (baseline 비교용)
    │   │   └── modified/
    │   │       ├── syncnet_temporal.py   # 핵심 모델: SyncNetTemporal (SyncLT)
    │   │       ├── temporal.py           # GRU/TCN 기반 시간적 모듈
    │   │       ├── pooling.py            # Attention pooling
    │   │       ├── lip_encoder_tokens.py, audio_encoder_tokens.py
    │   │
    │   ├── data/                 # GRID / HDTF 데이터로더
    │   └── loss/                 # Contrastive / Margin Ranking / InfoNCE loss
    │
    ├── results/
    │   ├── train_grid_marginranking/   # config, 체크포인트, 학습 로그, 평가 결과
    │   ├── train_grid_contrastive/
    │   └── train_hdtf_contrastive/
    │
    └── data/
        ├── GRID/    # 전처리 결과 샘플 (5개 화자)
        └── HDTF/    # 전처리 결과 샘플 (5개 클립)

## 환경 설정

    conda env create -f environment.yml
    conda activate lip-sync-score
    pip install -e .

`pip install -e .`로 설치하면 `PYTHONPATH` 설정 없이 `lipsyncscore` 패키지를 바로 import할 수 있습니다.

## 실행 방법

### 1. 데이터 전처리

    python scripts/prepare_grid.py --config configs/prepare_grid_lss.yaml
    python scripts/prepare_hdtf.py --config configs/prepare_hdtf.yaml
    python scripts/preprocess_latent.py --stage stage1 --data_root <HDTF_processed_경로> --out_root <출력_경로>

### 2. 학습

    python scripts/train_grid.py --config configs/train_grid_marginranking.yaml
    python scripts/train_grid.py --config configs/train_grid_contrastive.yaml
    python scripts/train_hdtf.py --config configs/train_hdtf_contrastive.yaml

### 3. 평가

    python scripts/eval_grid.py --config configs/train_grid_marginranking.yaml
    python scripts/eval_hdtf.py --config configs/train_hdtf_contrastive.yaml

## ADLip2 연동

`train_hdtf_contrastive`로 학습된 체크포인트(`results/train_hdtf_contrastive/best.pth`)는 [ADLip2](https://github.com/weonjungin/ADLip2)에서 LatentSync loss와 결합된 립 특화 contrastive loss(SyncLT)로 사용되었습니다. 두 loss를 결합했을 때 42개 테스트 비디오 기준 LSE-C 5.696, LSE-D 8.853을 달성했습니다.

## 참고

- 데이터셋: [GRID](http://spandh.dcs.shef.ac.uk/gridcorpus/), [HDTF](https://github.com/MRzzm/HDTF) (라이선스 상 원본 데이터는 포함하지 않으며, 전처리 결과 샘플만 포함)
- Baseline: SyncNet ([Chung & Zisserman, 2016](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/))
