cd ~/projects/lip-sync-score
conda activate lip-sync-score


##### 2.9 #####
### exp.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/exp.yaml \
  --skip_bad_meta

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/train_syncnetlike_grid/eval_offset_curve_test_d+0f.csv \
  --out logs/train_syncnetlike_grid/exp1.png \
  --no_show

# new exp1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
> nohup python -u scripts/train.py --config configs/exp.yaml \
> > logs/train_syncnetlike_grid/exp1_ssd.log 2>&1 &




### exp2.yaml

CUDA_VISIBLE_DEVICES=00000000:2B:00.0 PYTHONPATH=src nohup python scripts/train.py \
  --config configs/exp2.yaml \
  > logs/exp2_rerun.log 2>&1 &
tail -f logs/exp2_rerun.log



PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/exp2.yaml \
  --skip_bad_meta

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/train_syncnet_temporal_grid/eval_offset_curve_test_d+0f.csv \
  --out logs/train_syncnet_temporal_grid/exp2.png \
  --no_show

# new exp2
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -u scripts/train.py --config configs/exp2.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/exp2.yaml \
  --skip_bad_meta

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/train_exp2/eval_offset_curve_test_d+0f.csv \
  --out logs/train_exp2/exp2.png \
  --no_show

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/exp2.yaml \
  --global_delay_frames -11 \
  --skip_bad_meta

### exp3.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/exp3.yaml \
  --skip_bad_meta

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/train_syncnet_temporal_grid_N11/eval_offset_curve_test_d+0f.csv \
  --out logs/train_syncnet_temporal_grid_N11/exp3.png \
  --no_show

# new exp3
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python -u scripts/train.py --config configs/exp3.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/exp3.yaml \
  --skip_bad_meta

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/train_syncnet_temporal_grid_N11/eval_offset_curve_test_d+0f.csv \
  --out logs/train_syncnet_temporal_grid_N11/exp3.png \
  --no_show


### exp4.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/exp4.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py --config configs/exp4.yaml


CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src \
nohup python scripts/train.py \
  --config configs/exp4.yaml \
  > logs/exp4_crossattn_grid/train.log 2>&1 &

tail -f logs/exp4_crossattn_grid/train.log

PYTHONPATH=src \
python scripts/eval_offset_curve.py \
  --config configs/exp4.yaml \
  --ckpt logs/exp4_crossattn_grid/checkpoints/best.pth

PYTHONPATH=scripts python scripts/plot_offset_curve_single.py \
  --csv logs/exp4_crossattn_grid/eval_offset_curve_test_d+0f.csv \
  --out logs/exp4_crossattn_grid/exp4.png \
  --no_show

#### temporal test
# ExpT0 (none)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py --config configs/expT0_temporal_none.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py --config configs/expT0_temporal_none.yaml --skip_bad_meta

PYTHONPATH=src python scripts/plot_offset_curve_single.py \
  --csv logs/expT0_temporal_none/eval_offset_curve_test_d+0f.csv \
  --out logs/expT0_temporal_none/plot_test_d+0f.png \
  --label "T0_none" \
  --no_show

# ExpT1 (gru)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/train.py --config configs/expT1_temporal_gru.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py --config configs/expT1_temporal_gru.yaml --skip_bad_meta

PYTHONPATH=src python scripts/plot_offset_curve_single.py \
  --csv logs/expT1_temporal_gru/eval_offset_curve_test_d+0f.csv \
  --out logs/expT1_temporal_gru/plot_test_d+0f.png \
  --label "T1_gru" \
  --no_show

# ExpT2
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT2_N11_gru_attn.yaml

PYTHONPATH=src \
python scripts/eval_offset_curve.py --config configs/expT2_N11_gru_attn.yaml --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT2_N11_gru_attn/eval_offset_curve_test_d+0f.csv \
  --out logs/expT2_N11_gru_attn/plot_test_d+0f.png \
  --label "T2_N11_GRU_ATTn" \
  --no_show

# ExpT3
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT3_N11_gru2_attn.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/expT3_N11_gru2_attn.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT3_N11_gru2_attn/eval_offset_curve_test_d+0f.csv \
  --out logs/expT3_N11_gru2_attn/plot_test_d+0f.png \
  --label "T3_N11_gru2_attn" \
  --no_show

# ExpT4a
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT4a_N11_gru2_attn_do01_wd1e4.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/expT4a_N11_gru2_attn_do01_wd1e4.yaml \
  --skip_bad_meta
  
PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4a_N11_gru2_attn_do01_wd1e4/eval_offset_curve_test_d+0f.csv \
  --out logs/expT4a_N11_gru2_attn_do01_wd1e4/plot_test_d+0f.png \
  --label "T4a" \
  --no_show
  
# ExpT4b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT4bA_N11_gru2_attn_hardneg_msep3.yaml

PYTHONPATH=src python scripts/eval_offset_curve.py \
  --config configs/expT4bA_N11_gru2_attn_hardneg_msep3.yaml \
  --skip_bad_meta
  
PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4bA_N11_gru2_attn_hardneg_msep3/eval_offset_curve_test_d+0f.csv \
  --out logs/expT4bA_N11_gru2_attn_hardneg_msep3/plot_test_d+0f.png \
  --label "T4bA" \
  --no_show

# ExpT4c
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT4c_N11_gru2_attn_symhard.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expT4c_N11_gru2_attn_symhard.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4c_N11_gru2_attn_symhard/eval_offset_curve_test_d+0f.csv \
  --out logs/expT4c_N11_gru2_attn_symhard/plot_test_d+0f.png \
  --label "T4c" \
  --no_show

# ExpT4bA
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expT4bA_N11_gru2_attn_center_hp07.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expT4bA_N11_gru2_attn_center_hp07.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4bA_N11_gru2_attn_center_hp07/eval_offset_curve_test_d+0f.csv \
  --out logs/expT4bA_N11_gru2_attn_center_hp07/plot_test_d+0f.png \
  --label "T4bA" \
  --no_show



############################################
#### HDTF
# 전처리 all
nohup bash -lc '
PYTHONPATH=src python scripts/prepare_hdtf.py \
  --config configs/prepare_hdtf_ours.yaml
' > logs/prepare_hdtf/prepare_full.log 2>&1 &


# expT4bA - center
PYTHONPATH=src python scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_hdtf_expT4bA.yaml \
  --skip_bad_meta

nohup bash -lc '
export PYTHONPATH=/home/jiweon/projects/lip-sync-score/src
/home/jiweon/.conda/envs/lip-sync-score/bin/python \
  /home/jiweon/projects/lip-sync-score/scripts/eval_offset_curve_hdtf.py \
  --config /home/jiweon/projects/lip-sync-score/configs/eval_hdtf_expT4bA.yaml \
  --skip_bad_meta
' > /home/jiweon/projects/lip-sync-score/logs/our_hdtf/expT4bA_hdtf/eval.log 2>&1 &

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4bA_N11_gru2_attn_center_hp07/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expT4bA_N11_gru2_attn_center_hp07/hdtf.png \
  --label "T4bA_hdtf" \
  --no_show

# 재전처리 2000_v2
# 얼굴 검출 실패 -> 직전 ROI를 그대로 복제(freeze) : prepare_hdtf.py
nohup /home/jiweon/.conda/envs/lip-sync-score/bin/python \
  scripts/prepare_hdtf.py \
  --config configs/prepare_hdtf_2000_v2.yaml \
  > logs/prepare_hdtf_2000_v2/prepare.log 2>&1 &

PYTHONPATH=src python scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_p_v2.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expT4bA_N11_gru2_attn_center_hp07/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expT4bA_N11_gru2_attn_center_hp07/hdtf_v2.png \
  --label "T4bA_hdtf_v2" \
  --no_show

# 2500을 필터링함 -> 2404개 남음
PYTHONPATH=src python scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_hdtf_expT4bA.yaml \
  --skip_bad_meta

# our VS SyncNet overlap plot
cd /home/jiweon/projects/lip-sync-score

PYTHONPATH=src python scripts/plot_offset_meanstd_overlay.py \
  --csv_a /home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/eval_summary_hdtf_syncnet_python_as_ours_hdtf_d+0f.csv \
  --label_a "SyncNet-python (ours metric)" \
  --csv_b /home/jiweon/projects/lip-sync-score/logs/expT4bA_N11_gru2_attn_center_hp07/eval_summary_hdtf_hdtf_d+0f.csv \
  --label_b "Ours (expT4bA)" \
  --out /home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/overlay_syncnetpython_vs_ours_hdtf.png \
  --title "HDTF Offset Curve Overlay (same metric)" \
  --no_show

### MEAD
# HDD로 불러오기
(syncnet) jiweon@cvlab:/media/HDD/jiweon$ mkdir -p /home/jiweon/projects/lip-sync-score/logs/MEAD
(syncnet) jiweon@cvlab:/media/HDD/jiweon$ mkdir -p /media/HDD/jiweon/MEAD nohup rclone copy \ gdrive:MEAD \ /media/HDD/jiweon/MEAD \ --progress \ --transfers 4 \ --checkers 8 \ --retries 10 \ --low-level-retries 50 \ > /home/jiweon/projects/lip-sync-score/logs/MEAD/rclone.log 2>&1 &

# front+neutral 2000개만 추출하는 스크립트

MEAD_ROOT=/media/HDD/jiweon/MEAD
OUT_ROOT=/media/HDD/jiweon/MEAD_selected
N=2000

mkdir -p "$OUT_ROOT"

TMPDIR=$(mktemp -d)
ALL_LIST="$TMPDIR/all_front_neutral_videos.tsv"
SEL_LIST="$TMPDIR/selected_${N}.tsv"
SPEAKERS="$TMPDIR/speakers.txt"

: > "$ALL_LIST"

# 1) 전체 speaker에서 front+neutral video 목록 수집 (speaker \t path)
for spkdir in "$MEAD_ROOT"/M* "$MEAD_ROOT"/W*; do
  [ -d "$spkdir" ] || continue
  spk=$(basename "$spkdir")
  vt="$spkdir/video.tar"
  [ -f "$vt" ] || continue

  tar -tf "$vt" \
    | grep '^video/front/neutral/' \
    | grep -E '\.mp4$' \
    | sed "s|^|$spk\t|" >> "$ALL_LIST"
done

echo "[INFO] candidates:" $(wc -l < "$ALL_LIST")

# 2) 2000개 샘플링
shuf "$ALL_LIST" | head -n "$N" > "$SEL_LIST"
echo "[INFO] selected:" $(wc -l < "$SEL_LIST")

cut -f1 "$SEL_LIST" | sort -u > "$SPEAKERS"

# 3) speaker별로 video/audio 같이 추출
while read -r spk; do
  spkdir="$MEAD_ROOT/$spk"
  vt="$spkdir/video.tar"
  at="$spkdir/audio.tar"
  [ -f "$vt" ] || continue

  out_spk="$OUT_ROOT/$spk"
  mkdir -p "$out_spk"

  # 선택된 video 경로들
  awk -F'\t' -v s="$spk" '$1==s{print $2}' "$SEL_LIST" > "$TMPDIR/${spk}_videos.txt"
  echo "[INFO] $spk videos:" $(wc -l < "$TMPDIR/${spk}_videos.txt")

  # video 추출 (경로 유지: out_spk/video/front/neutral/...)
  tar -xf "$vt" -C "$out_spk" -T "$TMPDIR/${spk}_videos.txt"

  # audio 경로 만들기:
  # video/front/neutral/level_1/001.mp4 -> audio/neutral/level_1/001.m4a
  sed 's|^video/[^/]\+/|audio/|; s|\.mp4$|.m4a|' "$TMPDIR/${spk}_videos.txt" > "$TMPDIR/${spk}_audios.txt"

  if [ -f "$at" ]; then
    # audio.tar에 실제 존재하는 것만 추출 (안전)
    tar -tf "$at" > "$TMPDIR/${spk}_audio_index.txt"
    grep -F -f "$TMPDIR/${spk}_audios.txt" "$TMPDIR/${spk}_audio_index.txt" > "$TMPDIR/${spk}_audios_exist.txt"
    echo "[INFO] $spk audios:" $(wc -l < "$TMPDIR/${spk}_audios_exist.txt")

    [ -s "$TMPDIR/${spk}_audios_exist.txt" ] && tar -xf "$at" -C "$out_spk" -T "$TMPDIR/${spk}_audios_exist.txt"
  else
    echo "[WARN] no audio.tar for $spk"
  fi

done < "$SPEAKERS"

echo "[DONE] OUT_ROOT=$OUT_ROOT"
echo "[TMP] $TMPDIR"

# MEAD_selected 전처리 코드


### 3.4

# expN1.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN1.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN1.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN1/eval_offset_curve_test_d+0f.csv \
  --out logs/expN1/plot_test_d+0f.png \
  --label "ExpN1" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN1_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN1/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN1/plot_hdtf.png \
  --label "ExpN1_hdtf" \
  --no_show

# expN2.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN2.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN2.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN2/eval_offset_curve_test_d+0f.csv \
  --out logs/expN2/plot_test_d+0f.png \
  --label "ExpN2" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN2_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN2/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN2/plot_hdtf.png \
  --label "ExpN2_hdtf" \
  --no_show

# expN3.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN3.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN3.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN3/eval_offset_curve_test_d+0f.csv \
  --out logs/expN3/plot_test_d+0f.png \
  --label "ExpN3" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN3_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN3/plot_hdtf.png \
  --label "ExpN3_hdtf" \
  --no_show

# expN4.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN4.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN4.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN4/eval_offset_curve_test_d+0f.csv \
  --out logs/expN4/plot_test_d+0f.png \
  --label "ExpN4" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN4_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN4/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN4/plot_hdtf.png \
  --label "ExpN4_hdtf" \
  --no_show

# expNx.yaml
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN7.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN7.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN7/more_pos/eval_offset_curve_test_d+0f.csv \
  --out logs/expN7/more_pos/plot_test_d+0f.png \
  --label "ExpN7_more_pos" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN7_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN7/more_pos/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN7/more_pos/plot_hdtf.png \
  --label "ExpN7_more_pos_hdtf" \
  --no_show

# expN8,9 여기서부터 다시 
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py --config configs/expN11.yaml

PYTHONPATH=src python -u scripts/eval_offset_curve.py \
  --config configs/expN11.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN11/eval_offset_curve_test_d+0f.csv \
  --out logs/expN11/plot_test_d+0f.png \
  --label "ExpN11" \
  --no_show

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/eval_expN11_hdtf.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN11/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expN11/plot_hdtf.png \
  --label "ExpN11_hdtf" \
  --no_show

### expN 시리즈 all overlap plot
PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expT4bA=logs/expT4bA_N11_gru2_attn_center_hp07/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN1=logs/expN1/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN2=logs/expN2/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN3=logs/expN3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN4=logs/expN4/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN5=logs/expN5/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN6=logs/expN6/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN7=logs/expN7/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN8=logs/expN8/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN9=logs/expN9/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  syncnet_python=/home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/eval_summary_hdtf_syncnet_python_as_ours_hdtf_d+0f.csv \
  --out logs/compare_all/offset_curve_hdtf_overlap_0to1.png \
  --title "HDTF Offset Curve Overlay (expT4bA ~ expN9)"

  PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expN2=logs/expN2/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN3=logs/expN3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN4=logs/expN4/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/margin_comp.png \
  --title "Margin 1.0~1,5 Overlay"

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expN5=logs/expN5/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN6=logs/expN6/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN7=logs/expN7/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  more_neg=logs/expN7/more_neg/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  more_pos=logs/expN7/more_pos/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/loss_comp.png \
  --title "loss_comp Overlay"

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expN8=logs/expN8/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN9=logs/expN9/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/change_loss.png \
  --title "change loss Overlay"

# 3.9

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expN10=logs/expN10/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  syncnet_python=/home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/eval_summary_hdtf_syncnet_python_as_ours_hdtf_d+0f.csv \
  --out logs/compare_all/10VSsyncnet.png \
  --title "10 VS syncnet"

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expN11=logs/expN11/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN11_1.0=logs/expN11/1.0/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN11_1.5=logs/expN11/1.5/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/N11.png \
  --title "N11 compare"

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expT4bA=logs/expT4bA_N11_gru2_attn_center_hp07/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN1=logs/expN1/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN2=logs/expN2/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN3=logs/expN3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN4=logs/expN4/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN5=logs/expN5/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN6=logs/expN6/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN7=logs/expN7/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN8=logs/expN8/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN9=logs/expN9/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN10=logs/expN10/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN11=logs/expN11/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN11_1.0=logs/expN11/1.0/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expN11_1.5=logs/expN11/1.5/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  syncnet_python=/home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/eval_summary_hdtf_syncnet_python_as_ours_hdtf_d+0f.csv \
  --out logs/compare_all/all.png \
  --title "HDTF Offset Curve Overlay (expT4bA ~ expN11)"

### 3.12

PYTHONPATH=src python -u scripts/eval_offset_curve_mead.py \
  --config configs/eval_expN1_mead.yaml \
  --skip_bad_meta

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expN1/eval_offset_curve_mead_mead_d+0f.csv \
  --out logs/expN1/plot_mead.png \
  --label "ExpN1_mead" \
  --no_show


#############################

for d in -5 -4 -3 -2 -1 0 1 2 3 4 5; do
  echo "===== delay $d ====="
  PYTHONPATH=/home/jiweon/projects/lip-sync-score/src CUDA_VISIBLE_DEVICES=0 \
  python /home/jiweon/projects/lip-sync-score/scripts/eval_offset_curve_mead.py \
    --config /home/jiweon/projects/lip-sync-score/configs/eval_expN1_mead.yaml \
    --global_delay_frames $d \
    --skip_bad_meta
done


###########################
######## 3.13 입 얼굴 비교

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
python scripts/train.py \
  --config configs/expE3B.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
nohup python -u scripts/train.py \
  --config configs/expE2_syncnet_mouth.yaml \
  > nohup_E2_syncnet_mouth.out 2>&1 &

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/expE3B.yaml \
  --skip_bad_meta

############
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
nohup python -u scripts/train.py \
  --config configs/expE2C.yaml \
  > nohup_E2C_syncnet_mouth.out 2>&1 &

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
nohup python -u scripts/train.py \
  --config configs/expE3B.yaml \
  > nohup_E3B_ours_mouth_hdtf.out 2>&1 &
  
## 내일 할 것
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
nohup python -u scripts/train.py \
  --config configs/expE2B.yaml \
  > nohup_E2B_syncnet_mouth_grid.out 2>&1 &


PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/expE3B.yaml \
  --skip_bad_meta

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/expE2B.yaml \
  --utts_list data/HDTF/hdtf_eval_200.txt \
  --skip_bad_meta

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/expE1.yaml \
  --utts_list data/HDTF/hdtf_eval_200.txt \
  --skip_bad_meta

# E2C, E3B

PYTHONPATH=src \
python scripts/plot_offset_curve_single.py \
  --csv logs/expE2_syncnet_mouth/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/expE2_syncnet_mouth/plot_hdtf.png \
  --label "ExpE2A" \
  --no_show

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expE1_syncnet_python=/home/jiweon/projects/syncnet_python/log/syncnet_python_as_ours_eval/eval_summary_hdtf_syncnet_python_as_ours_hdtf_d+0f.csv \
  expE2A=logs/expE2_syncnet_mouth/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE2B=logs/expE2B_syncnet_mouth_grid/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE2C=logs/expE2C_syncnet_mouth_scratch/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE3A_expN11=logs/expN11/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/expEX.png \
  --title "HDTF Offset Curve Overlay (expE1 ~ expE3)"

PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expE2A=logs/expE2_syncnet_mouth/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE2B=logs/expE2B_syncnet_mouth_grid/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE2C=logs/expE2C_syncnet_mouth_scratch/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE3A_expN11=logs/expN11/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expE2C=logs/expE2C_syncnet_mouth_scratch/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/syncnetVSours.png \
  --title "SyncNet VS Ours"

################################
### 3.18

# hdtf fix split
PYTHONPATH=src python scripts/make_hdtf_ft_splits.py

# Ax, Bx
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
nohup python -u scripts/train.py \
  --config configs/expB3.yaml \
  > nohup_B3.out 2>&1 &

PYTHONPATH=src python -u scripts/eval_offset_curve_hdtf.py \
  --config configs/expB3.yaml \
  --utts_list data/HDTF/hdtf_eval_200.txt \
  --skip_bad_meta


PYTHONPATH=src python scripts/plot_offset_curve_overlap_all.py \
  --inputs \
  expA1=logs/expA1/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expA3=logs/expA3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expB1=logs/expB1/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  expB3=logs/expB3/eval_offset_curve_hdtf_hdtf_d+0f.csv \
  --out logs/compare_all/compare.png \
  --title "compare"

# grid 전처리
PYTHONPATH=src python scripts/prepare_hdtf_fullface_from_pycrop.py \
  --config configs/prepare_grid_fullface.yaml

#################################


# 사람별 SSD 사용량
du -sh /home/* 2>/dev/null | sort -h

############################################### 5.6 재학습

cd ~/projects/lip-sync-score
conda activate lip-sync-score

nohup python /home/jiweon/projects/lip-sync-score/scripts/preprocess_latent.py \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --out_root /media/HDD/jiweon/hdtf_latents \
    --device cuda:0 \
    --batch_size 16 \
    > /media/HDD/jiweon/hdtf_latents/preprocess.log 2>&1 &

### 
cd /home/jiweon/projects/lip-sync-score

PYTHONPATH=src nohup python scripts/train.py \
    --config configs/train_hdtf_B.yaml \
    > logs/train_hdtf_B.log 2>&1 &

###
cd /home/jiweon/projects/lip-sync-score

PYTHONPATH=src python scripts/eval_hdtf_latent.py \
    --ckpt logs/syncnet_hdtf/checkpoints/best.pth \
    --latent_root /media/HDD/jiweon/hdtf_latents \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --val_split /home/jiweon/projects/ADLip2/data/splits/val.txt \
    --device cuda:0 \
    --out logs/syncnet_hdtf/eval_offset.png

### 하관, 얼굴 전체 추출
# Stage 2 (하관)
PYTHONPATH=src python preprocess_latent.py \
    --stage stage2 \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --out_root  /media/HDD/jiweon/hdtf_latents_lower \
    --device cuda:3

# Stage 3 (얼굴 전체)
PYTHONPATH=src python preprocess_latent.py \
    --stage stage3 \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --out_root  /media/HDD/jiweon/hdtf_latents_face \
    --device cuda:3

### ---

nohup env PYTHONPATH=src python scripts/extract_face_and_lip_meta.py \
    --processed_dir /home/ihjung/HDTF_ssd/processed \
    --lower_only \
    > logs/extract_lower.log 2>&1 &

echo $!


nohup env PYTHONPATH=src python scripts/extract_face_and_lip_meta.py \
    --lower_only \
    --overwrite \
    --num_workers 8 \
    --video_dir /media/HDD/ihjung/HDTF/videos \
    --processed_dir /home/ihjung/HDTF_ssd/processed \
    > logs/extract_lower_final.log 2>&1 &

echo $!

# Stage 2
nohup env PYTHONPATH=src python scripts/preprocess_latent.py \
    --stage stage2 \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --out_root /media/HDD/jiweon/hdtf_latents_s2 \
    > logs/preprocess_s2.log 2>&1 &

# Stage 3
nohup env PYTHONPATH=src python scripts/preprocess_latent.py \
    --stage stage3 \
    --data_root /home/ihjung/HDTF_ssd/processed \
    --out_root /media/HDD/jiweon/hdtf_latents_s3 \
    > logs/preprocess_s3.log 2>&1 &

### train
PYTHONPATH=src nohup python scripts/train_progressive.py \
    --config configs/expP2.yaml \
    > logs/expP2_run.log 2>&1 &

PYTHONPATH=src nohup python scripts/train_progressive.py \
    --config configs/expP3.yaml \
    > logs/expP3_run.log 2>&1 &

PYTHONPATH=src nohup python scripts/train.py \
    --config configs/expP3.yaml \
    > logs/expP3_run.log 2>&1 &

### exal P1 P2
# expP1
PYTHONPATH=src python scripts/eval_hdtf_progressive.py \
    --ckpt logs/expP3/checkpoints/best.pth \
    --fusion_mode last \
    --out logs/expP3/eval_offset.png

# expP2
PYTHONPATH=src python scripts/eval_hdtf_progressive.py \
    --ckpt logs/expP2/checkpoints/best.pth \
    --fusion_mode concat \
    --out logs/expP2/eval_offset.png

# expP3
PYTHONPATH=src python scripts/eval_hdtf_temporal.py \
    --ckpt logs/syncnet_hdtf_tcn/checkpoints/best.pth \
    --latent_root /media/HDD/jiweon/latents/hdtf_s1 \
    --data_root /media/HDD/jiweon/processed/HDTF/HDTF_processed \
    --val_split /home/jiweon/projects/ADLip2/data/splits/val.txt \
    --device cuda:0 \
    --out logs/syncnet_hdtf_tcn/eval_offset.png