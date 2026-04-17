# ============================================================
# Makefile — CUDA Unified Memory Latency Profiler
# Target: NVIDIA RTX 3050 Ti (Ampere, sm_86)
# ============================================================

NVCC       := nvcc
CXX_FLAGS  := -std=c++17
NVCC_FLAGS := -O3 -arch=sm_86 -lineinfo --use_fast_math
LDFLAGS    := -lcuda

TARGET     := profiler
SRC        := profiler.cu

.PHONY: all clean run analyze report help

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXX_FLAGS) $(NVCC_FLAGS) $(LDFLAGS) $< -o $@
	@echo "[BUILD] $(TARGET) compiled successfully"

# Run all 5 experiments
run: $(TARGET)
	./$(TARGET)

# Run a specific experiment (usage: make exp EXP=1)
exp: $(TARGET)
	./$(TARGET) $(EXP)

# Generate plots and statistical report
analyze:
	python analyze.py

# Full pipeline: build → run → analyze
report: $(TARGET)
	./$(TARGET)
	python analyze.py

clean:
	rm -f $(TARGET) *.csv report.md
	rm -rf plots/

help:
	@echo ""
	@echo "  make           — Build the profiler binary"
	@echo "  make run       — Run all 5 experiments (generates CSVs)"
	@echo "  make exp EXP=N — Run experiment N only (1–5)"
	@echo "  make analyze   — Generate plots from existing CSVs"
	@echo "  make report    — Full pipeline: build + run + analyze"
	@echo "  make clean     — Remove build artifacts and outputs"
	@echo ""
