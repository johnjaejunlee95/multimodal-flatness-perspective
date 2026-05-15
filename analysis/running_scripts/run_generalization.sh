#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST_DEFAULT="${ROOT_DIR}/analysis/configs/checkpoints.yaml"
DATASET="${1:-all}"
shift || true
EXTRA_ARGS=("$@")

run_one() {
  local script_path="$1"
  local dataset_name="$2"
  local profile_name="$3"
  python "${ROOT_DIR}/${script_path}" \
    --dataset "${dataset_name}" \
    --manifest "${MANIFEST_DEFAULT}" \
    --profile "${profile_name}" \
    "${EXTRA_ARGS[@]}"
}

case "${DATASET}" in
  all)
    run_one "analysis/scripts/generalization/generalization_measure_avmnist.py" "AVMNIST" "generalization_avmnist_default"
    run_one "analysis/scripts/generalization/generalization_measure_cremad.py" "CREMAD" "generalization_cremad_default"
    run_one "analysis/scripts/generalization/generalization_measure_ks.py" "kinetic" "generalization_kineticsound_default"
    run_one "analysis/scripts/generalization/generalization_measure_food101.py" "Food101" "generalization_food101_default"
    ;;
  AVMNIST|avmnist)
    run_one "analysis/scripts/generalization/generalization_measure_avmnist.py" "AVMNIST" "generalization_avmnist_default"
    ;;
  CREMAD|cremad)
    run_one "analysis/scripts/generalization/generalization_measure_cremad.py" "CREMAD" "generalization_cremad_default"
    ;;
  KineticSound|kinetic|Kinetics)
    run_one "analysis/scripts/generalization/generalization_measure_ks.py" "kinetic" "generalization_kineticsound_default"
    ;;
  Food101|food101)
    run_one "analysis/scripts/generalization/generalization_measure_food101.py" "Food101" "generalization_food101_default"
    ;;
  *)
    echo "Unknown dataset: ${DATASET}" >&2
    exit 1
    ;;
esac
