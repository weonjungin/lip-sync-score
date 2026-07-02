"""
extract_face_and_lip_meta.py

원본 비디오 → 3단계 curriculum crop 추출

Stage 1: lip only
    landmark: LIP_LANDMARKS 20개, bbox + PAD(10)
    저장: frames/ (기존, 건드리지 않음) ✅

Stage 2: 입 + 턱
    landmark: 136(왼턱), 365(오른턱), 152(턱끝) 포함 + 입술 중심
    cx = mean(lip_xs), cy = mean(lip_ys)
    size = max(cx-136.x, 365.x-cx, 152.y-cy)
    저장: lower_frames/ (새로 추출)

Stage 3: 하관 전체
    landmark: 234(왼귀), 454(오른귀), 152(턱끝) 포함 + 입술 중심
    cx = mean(lip_xs), cy = mean(lip_ys)
    size = max(cx-234.x, 454.x-cx, 152.y-cy)
    저장: lower_frames_s3/ (새로 추출)

실행:
    # Stage 2, 3 모두 추출
    python scripts/extract_face_and_lip_meta.py --lower_only

    # 병렬 처리
    python scripts/extract_face_and_lip_meta.py --lower_only --num_workers 8

    # 테스트 (1개만)
    python scripts/extract_face_and_lip_meta.py --lower_only --max_videos 1

    # 덮어쓰기
    python scripts/extract_face_and_lip_meta.py --lower_only --overwrite
"""

import argparse
import cv2
import math
import numpy as np
import mediapipe as mp
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

FACE_SIZE  = 256
LOWER_SIZE = 96
PAD        = 10

LIP_LANDMARKS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146
]
LEFT_EYE  = 33
RIGHT_EYE = 263

# Stage 2 landmarks
JAW_LEFT_S2  = 136
JAW_RIGHT_S2 = 365
CHIN         = 152

# Stage 3 landmarks
JAW_LEFT_S3  = 234
JAW_RIGHT_S3 = 454


def setup_logger(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _align_and_get_landmarks(frame, h_orig, w_orig, face_mesh):
    """눈 기준 affine 정렬 후 landmark 반환. 실패시 None."""
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return None, None

    lm = result.multi_face_landmarks[0].landmark
    lx = int(lm[LEFT_EYE].x * w_orig)
    ly = int(lm[LEFT_EYE].y * h_orig)
    rx = int(lm[RIGHT_EYE].x * w_orig)
    ry = int(lm[RIGHT_EYE].y * h_orig)
    angle = math.degrees(math.atan2(ry - ly, rx - lx))
    M = cv2.getRotationMatrix2D((w_orig // 2, h_orig // 2), -angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR)

    result2 = face_mesh.process(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    if not result2.multi_face_landmarks:
        return None, None

    lm2 = result2.multi_face_landmarks[0].landmark
    return aligned, lm2


def _square_crop(frame, cx, cy, size, h_orig, w_orig):
    """정사각형 crop → LOWER_SIZE x LOWER_SIZE resize"""
    x1 = max(0, cx - size)
    x2 = min(w_orig, cx + size)
    y1 = max(0, cy - size)
    y2 = min(h_orig, cy + size)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (LOWER_SIZE, LOWER_SIZE))


def _compute_stage2_crop(lm2, h_orig, w_orig, frame):
    """
    Stage 2: 136, 365, 152 포함 + 입술 중심
    cx, cy = 입술 중심
    size = max(cx-136.x, 365.x-cx, 152.y-cy)
    """
    lip_xs = [int(lm2[j].x * w_orig) for j in LIP_LANDMARKS]
    lip_ys = [int(lm2[j].y * h_orig) for j in LIP_LANDMARKS]
    cx = int(np.mean(lip_xs))
    cy = int(np.mean(lip_ys))

    x_left  = int(lm2[JAW_LEFT_S2 ].x * w_orig)
    x_right = int(lm2[JAW_RIGHT_S2].x * w_orig)
    y_chin  = int(lm2[CHIN        ].y * h_orig) + 10

    size = max(cx - x_left, x_right - cx, y_chin - cy)
    if size <= 0:
        return None

    return _square_crop(frame, cx, cy, size, h_orig, w_orig)


def _compute_stage3_crop(lm2, h_orig, w_orig, frame):
    """
    Stage 3: 234, 454, 152 포함 + 입술 중심
    cx, cy = 입술 중심
    size = max(cx-234.x, 454.x-cx, 152.y-cy)
    """
    lip_xs = [int(lm2[j].x * w_orig) for j in LIP_LANDMARKS]
    lip_ys = [int(lm2[j].y * h_orig) for j in LIP_LANDMARKS]
    cx = int(np.mean(lip_xs))
    cy = int(np.mean(lip_ys))

    x_left  = int(lm2[JAW_LEFT_S3 ].x * w_orig)
    x_right = int(lm2[JAW_RIGHT_S3].x * w_orig)
    y_chin  = int(lm2[CHIN        ].y * h_orig) + 10

    size = max(cx - x_left, x_right - cx, y_chin - cy)
    if size <= 0:
        return None

    return _square_crop(frame, cx, cy, size, h_orig, w_orig)


# ──────────────────────────────────────────
# lower_only: lower_frames/ + lower_frames_s3/ 생성
# face_frames/, lip_meta.npy 는 건드리지 않음
# ──────────────────────────────────────────
def process_video_lower_only(video_path, processed_dir, overwrite=False):
    video_name = Path(video_path).stem
    save_dir   = Path(processed_dir) / video_name
    s2_dir     = save_dir / "lower_frames"
    s3_dir     = save_dir / "lower_frames_s3"

    if not save_dir.exists():
        return 0, 0, "no_processed_dir"
    indices_path = save_dir / "frame_indices.npy"
    if not indices_path.exists():
        return 0, 0, "no_frame_indices"

    if not overwrite and s2_dir.exists() and s3_dir.exists():
        existing = len(list(s2_dir.glob("*.png")))
        if existing > 0:
            return existing, 0, "skip"

    if s2_dir.exists(): shutil.rmtree(s2_dir)
    if s3_dir.exists(): shutil.rmtree(s3_dir)
    s2_dir.mkdir(parents=True, exist_ok=True)
    s3_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = np.load(str(indices_path))
    N = len(frame_indices)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )

    cap    = cv2.VideoCapture(str(video_path))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    saved = fail = 0
    blank = np.zeros((LOWER_SIZE, LOWER_SIZE, 3), dtype=np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        def save_fail():
            cv2.imwrite(str(s2_dir / f"{i:05d}.png"), blank)
            cv2.imwrite(str(s3_dir / f"{i:05d}.png"), blank)

        if not ret:
            save_fail(); fail += 1; continue

        aligned, lm2 = _align_and_get_landmarks(frame, h_orig, w_orig, face_mesh)
        if lm2 is None:
            save_fail(); fail += 1; continue

        s2_img = _compute_stage2_crop(lm2, h_orig, w_orig, aligned)
        s3_img = _compute_stage3_crop(lm2, h_orig, w_orig, aligned)

        cv2.imwrite(str(s2_dir / f"{i:05d}.png"), s2_img if s2_img is not None else blank)
        cv2.imwrite(str(s3_dir / f"{i:05d}.png"), s3_img if s3_img is not None else blank)

        if s2_img is not None and s3_img is not None:
            saved += 1
        else:
            fail += 1

    cap.release()
    face_mesh.close()
    return saved, fail, "done"


# ──────────────────────────────────────────
# 전체 처리: face_frames + lower_frames + lower_frames_s3
# ──────────────────────────────────────────
def process_video(video_path, processed_dir, overwrite=False):
    video_name      = Path(video_path).stem
    save_dir        = Path(processed_dir) / video_name
    face_dir        = save_dir / "face_frames"
    s2_dir          = save_dir / "lower_frames"
    s3_dir          = save_dir / "lower_frames_s3"
    meta_path       = save_dir / "lip_meta.npy"

    if not save_dir.exists():
        return 0, 0, "no_processed_dir"
    indices_path = save_dir / "frame_indices.npy"
    if not indices_path.exists():
        return 0, 0, "no_frame_indices"
    if not overwrite and face_dir.exists() and meta_path.exists():
        existing = len(list(face_dir.glob("*.png")))
        if existing > 0:
            return existing, 0, "skip"

    for d in [face_dir, s2_dir, s3_dir]:
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    frame_indices = np.load(str(indices_path))
    N = len(frame_indices)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5
    )

    cap    = cv2.VideoCapture(str(video_path))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    lip_meta = np.zeros((N, 6), dtype=np.int32)
    saved = fail = 0
    blank_f = np.zeros((FACE_SIZE,  FACE_SIZE,  3), dtype=np.uint8)
    blank_l = np.zeros((LOWER_SIZE, LOWER_SIZE, 3), dtype=np.uint8)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        def save_fail():
            lip_meta[i] = [0, int(frame_idx), -1, -1, -1, -1]
            cv2.imwrite(str(face_dir / f"{i:05d}.png"), blank_f)
            cv2.imwrite(str(s2_dir   / f"{i:05d}.png"), blank_l)
            cv2.imwrite(str(s3_dir   / f"{i:05d}.png"), blank_l)

        if not ret:
            save_fail(); fail += 1; continue

        aligned, lm2 = _align_and_get_landmarks(frame, h_orig, w_orig, face_mesh)
        if lm2 is None:
            save_fail(); fail += 1; continue

        # face crop (Stage 3 저장용 - 기존 방식)
        face_xs = [int(lm2[j].x * w_orig) for j in range(len(lm2))]
        face_ys = [int(lm2[j].y * h_orig) for j in range(len(lm2))]
        fx1 = max(0, min(face_xs)); fy1 = max(0, min(face_ys))
        fx2 = min(w_orig, max(face_xs)); fy2 = min(h_orig, max(face_ys))
        face_crop = aligned[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            save_fail(); fail += 1; continue
        face_resized = cv2.resize(face_crop, (FACE_SIZE, FACE_SIZE))

        # lip bbox (lip_meta 저장용)
        sx = FACE_SIZE / (fx2 - fx1) if (fx2 - fx1) > 0 else 1
        sy = FACE_SIZE / (fy2 - fy1) if (fy2 - fy1) > 0 else 1
        lip_xs_f = [int((int(lm2[j].x * w_orig) - fx1) * sx) for j in LIP_LANDMARKS]
        lip_ys_f = [int((int(lm2[j].y * h_orig) - fy1) * sy) for j in LIP_LANDMARKS]
        x1 = max(0, min(lip_xs_f) - PAD)
        x2 = min(FACE_SIZE, max(lip_xs_f) + PAD)
        y1 = max(0, min(lip_ys_f) - PAD)
        y2 = min(FACE_SIZE, max(lip_ys_f) + PAD)

        s2_img = _compute_stage2_crop(lm2, h_orig, w_orig, aligned)
        s3_img = _compute_stage3_crop(lm2, h_orig, w_orig, aligned)

        cv2.imwrite(str(face_dir / f"{i:05d}.png"), face_resized)
        cv2.imwrite(str(s2_dir   / f"{i:05d}.png"), s2_img if s2_img is not None else blank_l)
        cv2.imwrite(str(s3_dir   / f"{i:05d}.png"), s3_img if s3_img is not None else blank_l)
        lip_meta[i] = [1, int(frame_idx), x1, y1, x2, y2]
        saved += 1

    cap.release()
    face_mesh.close()
    np.save(str(meta_path), lip_meta)
    return saved, fail, "done"


def _worker(video_path, processed_dir, overwrite, lower_only):
    try:
        if lower_only:
            saved, fail, status = process_video_lower_only(
                video_path, processed_dir, overwrite
            )
        else:
            saved, fail, status = process_video(
                video_path, processed_dir, overwrite
            )
        return str(Path(video_path).stem), saved, fail, status, None
    except Exception as e:
        return str(Path(video_path).stem), 0, 0, "error", str(e)


def main(args):
    video_dir     = Path(args.video_dir)
    processed_dir = Path(args.processed_dir)
    logger = setup_logger(args.log)
    logger.info(
        f"시작: lower_only={args.lower_only} "
        f"overwrite={args.overwrite} "
        f"workers={args.num_workers}"
    )

    videos = sorted(video_dir.glob("*.mp4"))
    if args.max_videos is not None:
        videos = videos[:args.max_videos]
    logger.info(f"총 {len(videos)}개 비디오")

    worker_fn = partial(
        _worker,
        processed_dir=processed_dir,
        overwrite=args.overwrite,
        lower_only=args.lower_only,
    )

    if args.num_workers <= 1:
        results = [worker_fn(v) for v in tqdm(videos)]
    else:
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, videos),
                total=len(videos),
            ))

    total_saved = total_fail = skip = error = 0
    for vi, (video_name, saved, fail, status, err) in enumerate(results):
        if status == "skip":
            skip += 1
        elif status == "error":
            error += 1
            logger.error(f"[{vi+1}/{len(videos)}] {video_name}: ERROR {err}")
        else:
            total_saved += saved
            total_fail  += fail
            logger.info(
                f"[{vi+1}/{len(videos)}] {video_name}: "
                f"saved={saved} fail={fail} status={status}"
            )

    logger.info(
        f"완료: saved={total_saved} fail={total_fail} "
        f"skip={skip} error={error}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",     default="/media/HDD/ihjung/HDTF/videos")
    parser.add_argument("--processed_dir", default="/home/ihjung/HDTF_ssd/processed")
    parser.add_argument("--log",           default="/home/jiweon/projects/lip-sync-score/logs/extract_lower.log")
    parser.add_argument("--lower_only",    action="store_true",
                        help="lower_frames/ + lower_frames_s3/ 만 생성 (face_frames 유지)")
    parser.add_argument("--overwrite",     action="store_true")
    parser.add_argument("--max_videos",    type=int, default=None,
                        help="처리할 최대 비디오 수 (테스트용)")
    parser.add_argument("--num_workers",   type=int, default=1,
                        help="병렬 처리 워커 수 (권장: 8)")
    args = parser.parse_args()
    main(args)