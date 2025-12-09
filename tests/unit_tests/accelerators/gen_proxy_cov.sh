#!/bin/bash
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "$WORK_DIR/../../../" && pwd)"
echo "TOP_DIR is ${TOP_DIR}"

mkdir -p proxy_report
gcovr --root ${TOP_DIR}/omni/accelerators/sched/nginx-1.28.0 --filter "${TOP_DIR}/omni/accelerators/sched/omni_proxy/.*\.c$" --html --html-details -o ./proxy_report/coverage_report.html --gcov-ignore-errors=no_working_dir_found
tar czf proxy_cov.tar.gz ./proxy_report