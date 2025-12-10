#!/usr/bin/env bash
export COVERAGE_RCFILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.coveragerc"
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# shellcheck source=tests/utils.sh
source "${SCRIPT_DIR}/utils.sh"

target="all"
reports_dir=""
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --unit)
      target="unit"
      shift
      ;;
    --integrated|--integration)
      target="integrated"
      shift
      ;;
    --reports-dir)
      reports_dir="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

marker_args=()
target_path="${SCRIPT_DIR}"
report_name="pytest-all.xml"

case "${target}" in
  unit)
    marker_args=(-m "unit and not gpu")
    target_path="${SCRIPT_DIR}/unit_tests"
    report_name="pytest-unit.xml"
    ;;
  integrated)
    marker_args=(-m "integrated and not gpu")
    target_path="${SCRIPT_DIR}/integrated_tests"
    report_name="pytest-integrated.xml"
    ;;
  all)
    marker_args=(-m "not gpu")
    ;; # run everything except GPU-tagged cases
  *)
    log_warn "Unknown target '${target}', defaulting to all tests."
    ;;
esac

cmd=(
  pytest 
  "${marker_args[@]}" 
  "${target_path}"
  --cov-report=html
  --cov-report=xml
  "${extra_args[@]}"
)

if [[ -n "${reports_dir}" ]]; then
  mkdir -p "${reports_dir}"
  report_file="${reports_dir}/${report_name}"
  cmd+=(--junitxml "${report_file}")
  log_info "JUnit report will be written to ${report_file}"
fi

log_info "Running tests: ${cmd[*]}"
(cd "${ROOT_DIR}" && "${cmd[@]}")