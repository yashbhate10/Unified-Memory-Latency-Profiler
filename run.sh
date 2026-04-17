#!/usr/bin/env bash
# =============================================================================
# run.sh — Full Pipeline Runner
# CUDA Unified Memory Latency Profiler
#
# Usage:
#   ./run.sh          — Build + run all 5 experiments + analyze
#   ./run.sh 1 3      — Run only experiments 1 and 3
#   ./run.sh --clean  — Remove all generated artifacts
# =============================================================================

set -euo pipefail

TARGET="profiler"
SRC="profiler.cu"
ARCH="sm_86"
PYTHON="${PYTHON:-python3}"

# Colors for terminal output
RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'
YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
    echo ""
    echo -e "${BOLD}${CYAN}============================================================${RESET}"
    echo -e "${BOLD}${CYAN}  CUDA Unified Memory Latency Profiler — Pipeline${RESET}"
    echo -e "${BOLD}${CYAN}============================================================${RESET}"
    echo ""
}

step() { echo -e "${YELLOW}[STEP]${RESET} $1"; }
ok()   { echo -e "${GREEN}[ OK ]${RESET} $1"; }
err()  { echo -e "${RED}[ERR ]${RESET} $1"; exit 1; }

# --clean
if [[ "${1:-}" == "--clean" ]]; then
    step "Cleaning generated files..."
    rm -f "$TARGET" *.csv report.md
    rm -rf plots/
    ok "Clean complete"
    exit 0
fi

banner

# ---- Step 1: Check prerequisites ----
step "Checking prerequisites..."
command -v nvcc   >/dev/null 2>&1 || err "nvcc not found. Install CUDA Toolkit."
command -v nvidia-smi >/dev/null 2>&1 || err "nvidia-smi not found. Check GPU drivers."
command -v $PYTHON >/dev/null 2>&1 || err "$PYTHON not found."

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo    "  GPU    : $GPU_NAME"
echo    "  NVCC   : $(nvcc --version | grep release | awk '{print $5,$6}')"
echo    "  Python : $($PYTHON --version)"
ok "Prerequisites OK"

# ---- Step 2: Compile ----
step "Compiling $SRC → $TARGET  (arch=$ARCH, O3)..."
nvcc -std=c++17 -O3 -arch=$ARCH -lineinfo --use_fast_math -lcuda \
     "$SRC" -o "$TARGET" 2>&1
ok "Compilation successful"

# ---- Step 3: Run experiments ----
EXP_ARGS="${@:-}"   # pass specific exp numbers, or empty = all
if [[ -z "$EXP_ARGS" ]]; then
    step "Running ALL 5 experiments..."
    ./"$TARGET"
else
    step "Running experiments: $EXP_ARGS"
    ./"$TARGET" $EXP_ARGS
fi
ok "Experiments complete. CSVs written."

# ---- Step 4: Install Python deps if needed ----
step "Checking Python dependencies..."
$PYTHON -c "import pandas, matplotlib, seaborn, numpy, scipy" 2>/dev/null || {
    echo "  Installing missing packages..."
    pip install -q -r requirements.txt
}
ok "Python deps ready"

# ---- Step 5: Generate analysis ----
step "Generating plots and statistical report..."
mkdir -p plots
$PYTHON analyze.py
ok "Plots saved to ./plots/"
ok "Statistical report saved to report.md"

echo ""
echo -e "${BOLD}${GREEN}[DONE]${RESET} Pipeline complete."
echo -e "  Plots  : $(ls plots/*.png 2>/dev/null | wc -l) files in ./plots/"
echo -e "  Report : report.md"
echo ""
