#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/junxinfu/BIRL_benchmark"
PYTHON_BIN="/home/junxinfu/.conda/envs/Deeperhistreg/bin/python"
DHR_SCRIPT="${DHR_SCRIPT:-${ROOT}/bm_experiments/run_deeperhistreg_ablation.py}"
DATASET="${DATASET:-/home/junxinfu/ANHIR_dataset_medium}"
TABLE="${TABLE:-${ROOT}/data-images/pairs-imgs-lnds_anhir_training_available_5.csv}"
OUT_DIR="${OUT_DIR:-/mnt/d/home/junxinfu/BM_DeeperHistReg_ablation_ANHIR}"
NB_WORKERS="${NB_WORKERS:-1}"
RUN_SUFFIX="${RUN_SUFFIX:-ANHIR_5pairs}"
DELETE_TEMP="${DELETE_TEMP:-1}"

METHODS=(
  "SiftRansac:${ROOT}/configs/deeperhistreg_ablation_sift_ransac_birl_cpu.json"
  "SiftSuperGlue:${ROOT}/configs/deeperhistreg_ablation_sift_superglue_birl_cpu.json"
  "SuperPointRansac:${ROOT}/configs/deeperhistreg_ablation_superpoint_ransac_birl_cpu.json"
  "SuperPointSuperGlue:${ROOT}/configs/deeperhistreg_ablation_superpoint_superglue_birl_cpu.json"
)

cd "${ROOT}"

for entry in "${METHODS[@]}"; do
  name="${entry%%:*}"
  params="${entry#*:}"
  if [[ -n "${ONLY_METHODS:-}" && " ${ONLY_METHODS} " != *" ${name} "* ]]; then
    continue
  fi
  echo "Running ${name} with ${params}"
  cmd=(
    "${PYTHON_BIN}" bm_experiments/bm_DeeperHistReg.py
    -t "${TABLE}" \
    -d "${DATASET}" \
    -o "${OUT_DIR}" \
    -py "${PYTHON_BIN}" \
    -script "${DHR_SCRIPT}" \
    -params "${params}" \
    --name "Ablation_${name}_${RUN_SUFFIX}" \
    --case_name "BIRL_DeeperHistReg_${name}" \
    --unique \
    --nb_workers "${NB_WORKERS}"
  )
  if [[ "${DELETE_TEMP}" == "1" ]]; then
    cmd+=(--dtmp)
  fi
  "${cmd[@]}"
done
