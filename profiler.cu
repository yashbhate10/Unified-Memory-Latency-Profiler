// =============================================================================
// profiler.cu — CUDA Unified Memory Latency Profiler
// =============================================================================
// Systematically characterizes NVIDIA Unified Memory (UM) page migration
// overhead across five experimental dimensions:
//
//   EXP 1 — Size Sweep        : How does buffer size affect migration cost?
//   EXP 2 — Access Patterns   : Sequential vs. strided vs. random access
//   EXP 3 — Prefetch Benefit  : cudaMemPrefetchAsync vs. on-demand faulting
//   EXP 4 — Memory Advice     : ReadMostly / PreferredLocation hints
//   EXP 5 — Concurrent Access : CPU+GPU atomic contention latency
//
// Target GPU  : NVIDIA RTX 3050 Ti (Ampere, sm_86)
// Build       : nvcc -O3 -arch=sm_86 -lineinfo profiler.cu -o profiler
// Output      : CSV files consumed by analyze.py
// =============================================================================

#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <atomic>
#include <cuda_runtime.h>
#if defined(_MSC_VER)
    #include <intrin.h>   // _InterlockedExchangeAdd64
#endif

// ---------------------------------------------------------------------------
// CUDA 12+ API compatibility helpers
// cudaMemPrefetchAsync / cudaMemAdvise now take cudaMemLocation instead of
// a plain int device ID. These wrappers restore the old ergonomic interface.
// ---------------------------------------------------------------------------
static inline cudaError_t prefetchToCPU(void* ptr, size_t bytes) {
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeHost;
    loc.id   = 0;
    return cudaMemPrefetchAsync(ptr, bytes, loc, 0);
}
static inline cudaError_t prefetchToGPU(void* ptr, size_t bytes, int dev) {
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemPrefetchAsync(ptr, bytes, loc, 0);
}
static inline cudaError_t adviseGPU(void* ptr, size_t bytes,
                                     cudaMemoryAdvise advice, int dev) {
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemAdvise(ptr, bytes, advice, loc);
}
static inline cudaError_t adviseCPU(void* ptr, size_t bytes,
                                     cudaMemoryAdvise advice) {
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeHost;
    loc.id   = 0;
    return cudaMemAdvise(ptr, bytes, advice, loc);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::cerr << "[CUDA ERROR] " << __FILE__ << ":" << __LINE__         \
                      << "  " << cudaGetErrorString(_err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUDA_UM(call)                                                     \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess && _err != cudaErrorInvalidDevice &&            \
            _err != cudaErrorNotSupported) {                                    \
            std::cerr << "[CUDA UM WARNING] " << __FILE__ << ":" << __LINE__    \
                      << "  " << cudaGetErrorString(_err) << " (ignoring)\n";   \
        }                                                                       \
    } while (0)

// High-resolution wall-clock timer (nanoseconds)
static inline long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// Simple descriptive stats over a vector of latency samples
struct Stats {
    double mean, stddev, p50, p95, p99, min_val, max_val;
};

Stats compute_stats(std::vector<long>& v) {
    Stats s{};
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    long sum = std::accumulate(v.begin(), v.end(), 0L);
    s.mean   = (double)sum / v.size();
    s.min_val = v.front();
    s.max_val = v.back();
    // standard deviation
    double sq = 0;
    for (auto x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = std::sqrt(sq / v.size());
    // percentiles
    auto pct = [&](double p) -> double {
        double idx = p * (v.size() - 1);
        size_t lo  = (size_t)idx;
        size_t hi  = lo + 1 < v.size() ? lo + 1 : lo;
        return v[lo] + (idx - lo) * (v[hi] - v[lo]);
    };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

// Sequential read — touches every element; triggers page migration if data is
// resident on CPU when kernel launches.
__global__ void kernel_sequential_rw(float* data, int n, float delta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += delta;
}

// Strided read — stride-64 pattern stresses TLB by touching non-contiguous
// pages; reveals TLB shootdown overhead separate from page-fault costs.
__global__ void kernel_strided_rw(float* data, int n, int stride, float delta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i   = tid * stride;
    if (i < n) data[i] += delta;
}

// Random read — worst-case TLB miss pattern; each warp accesses random pages.
__global__ void kernel_random_rw(float* data, int n, float delta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // LCG hash to scatter threads; avoids sorting by the hardware prefetcher
    unsigned int idx = (1664525u * (unsigned int)tid + 1013904223u) % (unsigned int)n;
    data[idx] += delta;
}

// Atomic add — used for concurrent CPU+GPU contention experiment
__global__ void kernel_atomic_add(unsigned long long* counter, int iters) {
    for (int i = 0; i < iters; i++)
        atomicAdd(counter, 1ULL);
}

// Lightweight kernel just to force GPU-side page ownership
__global__ void kernel_touch(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 1.0f;   // force page fault + residency claim
}

// ---------------------------------------------------------------------------
// CPU-side access helpers
// ---------------------------------------------------------------------------

// Sequential CPU read — measures post-GPU-touch migration latency
long cpu_seq_access(volatile float* data, int n) {
    volatile float sum = 0;
    long t0 = now_ns();
    for (int i = 0; i < n; i++) sum += data[i];
    long t1 = now_ns();
    (void)sum;   // prevent dead-code elimination
    return t1 - t0;
}

// Strided CPU read
long cpu_stride_access(volatile float* data, int n, int stride) {
    volatile float sum = 0;
    long t0 = now_ns();
    for (int i = 0; i < n; i += stride) sum += data[i];
    long t1 = now_ns();
    (void)sum;
    return t1 - t0;
}

// Random CPU read (same LCG as GPU kernel)
long cpu_random_access(volatile float* data, int n) {
    volatile float sum = 0;
    long t0 = now_ns();
    for (int k = 0; k < n; k++) {
        unsigned int idx = (1664525u * (unsigned int)k + 1013904223u) % (unsigned int)n;
        sum += data[idx];
    }
    long t1 = now_ns();
    (void)sum;
    return t1 - t0;
}

// ---------------------------------------------------------------------------
// EXP 1 — Size Sweep
// ---------------------------------------------------------------------------
// For each buffer size S ∈ {4KB … 256MB}:
//   a) Measure CPU-only read latency (baseline, no GPU involvement)
//   b) Touch S on GPU, sync, then measure CPU re-read latency (post-migration)
// This isolates how page-migration cost scales with working-set size.
// ---------------------------------------------------------------------------
void exp1_size_sweep(const std::string& outfile) {
    const int WARM  = 5;    // warm-up iterations (discarded)
    const int RUNS  = 30;   // measured iterations per size

    // Buffer sizes: powers of 2 from 1KB to 256MB (in floats)
    std::vector<size_t> sizes;
    for (size_t bytes = 1 << 10; bytes <= (256ULL << 20); bytes <<= 1)
        sizes.push_back(bytes / sizeof(float));

    std::ofstream f(outfile);
    f << "size_bytes,access_type,run,latency_ns\n";

    std::cout << "\n[EXP 1] Size Sweep\n";

    for (size_t N : sizes) {
        size_t bytes = N * sizeof(float);
        float* data;
        CHECK_CUDA(cudaMallocManaged(&data, bytes));

        // Initialize on CPU — pages initially CPU-resident
        for (size_t i = 0; i < N; i++) data[i] = 1.0f;

        // Warm-up
        for (int w = 0; w < WARM; w++) {
            cpu_seq_access((volatile float*)data, (int)N);
            kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        std::vector<long> cpu_lats, migration_lats;

        for (int r = 0; r < RUNS; r++) {
            // Ensure pages are CPU-resident: prefetch back to CPU
            CHECK_CUDA_UM(prefetchToCPU(data, bytes));
            CHECK_CUDA(cudaDeviceSynchronize());

            // (a) Pure CPU baseline
            long cpu_t = cpu_seq_access((volatile float*)data, (int)N);
            cpu_lats.push_back(cpu_t);
            f << bytes << ",cpu_baseline," << r << "," << cpu_t << "\n";

            // (b) GPU touch → CPU re-read (migration path)
            kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
            CHECK_CUDA(cudaDeviceSynchronize());
            long mig_t = cpu_seq_access((volatile float*)data, (int)N);
            migration_lats.push_back(mig_t);
            f << bytes << ",post_migration," << r << "," << mig_t << "\n";
        }

        Stats cs = compute_stats(cpu_lats);
        Stats ms = compute_stats(migration_lats);
        std::cout << "  " << bytes / 1024 << " KB"
                  << "  CPU=" << (long)cs.mean << " ns"
                  << "  PostMigrate=" << (long)ms.mean << " ns"
                  << "  overhead=" << (long)(ms.mean - cs.mean) << " ns\n";

        CHECK_CUDA(cudaFree(data));
    }

    std::cout << "  -> Saved: " << outfile << "\n";
}

// ---------------------------------------------------------------------------
// EXP 2 — Access Pattern Comparison
// ---------------------------------------------------------------------------
// Fixed large buffer (64 MB). After GPU touch, compare CPU re-access latency
// for: sequential, stride-64, and random patterns.
// Reveals TLB pressure and hardware prefetcher effectiveness under migration.
// ---------------------------------------------------------------------------
void exp2_access_patterns(const std::string& outfile) {
    const int WARM  = 10;
    const int RUNS  = 50;
    const size_t N  = (64ULL << 20) / sizeof(float);   // 64 MB
    const size_t bytes = N * sizeof(float);
    const int STRIDE = 64;   // 64-element stride → 256-byte stride (4 cache lines)

    float* data;
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    for (size_t i = 0; i < N; i++) data[i] = 1.0f;

    // Warm-up
    for (int w = 0; w < WARM; w++) {
        kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());
        cpu_seq_access((volatile float*)data, (int)N);
    }

    std::ofstream f(outfile);
    f << "pattern,run,phase,latency_ns\n";

    std::cout << "\n[EXP 2] Access Patterns (64 MB)\n";

    struct PatternResult { std::string name; std::vector<long> cpu, mig; };
    std::vector<PatternResult> results = {
        {"sequential", {}, {}},
        {"strided",    {}, {}},
        {"random",     {}, {}}
    };

    for (int r = 0; r < RUNS; r++) {
        // Sequential
        CHECK_CUDA_UM(prefetchToCPU(data, bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
        results[0].cpu.push_back(cpu_seq_access((volatile float*)data, (int)N));
        kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());
        results[0].mig.push_back(cpu_seq_access((volatile float*)data, (int)N));

        // Strided
        CHECK_CUDA_UM(prefetchToCPU(data, bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
        results[1].cpu.push_back(cpu_stride_access((volatile float*)data, (int)N, STRIDE));
        kernel_strided_rw<<<((int)N / STRIDE + 255) / 256, 256>>>(data, (int)N, STRIDE, 0.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        results[1].mig.push_back(cpu_stride_access((volatile float*)data, (int)N, STRIDE));

        // Random
        CHECK_CUDA_UM(prefetchToCPU(data, bytes));
        CHECK_CUDA(cudaDeviceSynchronize());
        results[2].cpu.push_back(cpu_random_access((volatile float*)data, (int)N));
        kernel_random_rw<<<((int)N + 255) / 256, 256>>>(data, (int)N, 0.0f);
        CHECK_CUDA(cudaDeviceSynchronize());
        results[2].mig.push_back(cpu_random_access((volatile float*)data, (int)N));
    }

    for (auto& pr : results) {
        Stats cs = compute_stats(pr.cpu);
        Stats ms = compute_stats(pr.mig);
        std::cout << "  [" << pr.name << "]"
                  << "  CPU=" << (long)cs.mean << " ns"
                  << "  PostMigrate=" << (long)ms.mean << " ns\n";
        for (int r = 0; r < RUNS; r++) {
            f << pr.name << "," << r << ",cpu_baseline," << pr.cpu[r] << "\n";
            f << pr.name << "," << r << ",post_migration," << pr.mig[r] << "\n";
        }
    }

    std::cout << "  -> Saved: " << outfile << "\n";
    CHECK_CUDA(cudaFree(data));
}

// ---------------------------------------------------------------------------
// EXP 3 — Prefetch Benefit (On-demand vs. Explicit Prefetch)
// ---------------------------------------------------------------------------
// Across buffer sizes, compare GPU kernel execution time when:
//   (a) Data is CPU-resident → kernel must page-fault for every page (on-demand)
//   (b) cudaMemPrefetchAsync moves data to GPU before kernel launch (async overlap)
// Uses CUDA events for accurate GPU-side timing.
// ---------------------------------------------------------------------------
void exp3_prefetch(const std::string& outfile) {
    const int WARM  = 3;
    const int RUNS  = 20;

    std::vector<size_t> sizes;
    for (size_t bytes = 1 << 20; bytes <= (128ULL << 20); bytes <<= 1)
        sizes.push_back(bytes / sizeof(float));

    std::ofstream f(outfile);
    f << "size_bytes,strategy,run,gpu_time_us\n";

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    std::cout << "\n[EXP 3] Prefetch Benefit\n";

    for (size_t N : sizes) {
        size_t bytes = N * sizeof(float);
        float* data;
        CHECK_CUDA(cudaMallocManaged(&data, bytes));
        for (size_t i = 0; i < N; i++) data[i] = 1.0f;

        // Warm-up
        for (int w = 0; w < WARM; w++) {
            kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        std::vector<long> on_demand_us, prefetch_us;

        for (int r = 0; r < RUNS; r++) {
            // (a) On-demand: data on CPU, kernel triggers page faults
            CHECK_CUDA_UM(prefetchToCPU(data, bytes));
            CHECK_CUDA(cudaDeviceSynchronize());
            CHECK_CUDA(cudaEventRecord(ev_start));
            kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));
            float ms_od = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms_od, ev_start, ev_stop));
            long us_od = (long)(ms_od * 1000.0f);
            on_demand_us.push_back(us_od);
            f << bytes << ",on_demand," << r << "," << us_od << "\n";

            // (b) Explicit prefetch: async transfer hides latency before kernel
            CHECK_CUDA_UM(prefetchToCPU(data, bytes));
            CHECK_CUDA(cudaDeviceSynchronize());
            int dev; CHECK_CUDA(cudaGetDevice(&dev));
            CHECK_CUDA_UM(prefetchToGPU(data, bytes, dev));
            CHECK_CUDA(cudaDeviceSynchronize());          // overlap done
            CHECK_CUDA(cudaEventRecord(ev_start));
            kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
            CHECK_CUDA(cudaEventRecord(ev_stop));
            CHECK_CUDA(cudaEventSynchronize(ev_stop));
            float ms_pf = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ms_pf, ev_start, ev_stop));
            long us_pf = (long)(ms_pf * 1000.0f);
            prefetch_us.push_back(us_pf);
            f << bytes << ",explicit_prefetch," << r << "," << us_pf << "\n";
        }

        Stats od = compute_stats(on_demand_us);
        Stats pf = compute_stats(prefetch_us);
        double speedup = od.mean / pf.mean;
        std::cout << "  " << bytes / (1 << 20) << " MB"
                  << "  OnDemand=" << (long)od.mean << " us"
                  << "  Prefetch=" << (long)pf.mean << " us"
                  << "  Speedup=" << speedup << "x\n";

        CHECK_CUDA(cudaFree(data));
    }

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    std::cout << "  -> Saved: " << outfile << "\n";
}

// ---------------------------------------------------------------------------
// EXP 4 — Memory Advice Impact
// ---------------------------------------------------------------------------
// cudaMemAdvise hints guide the UM driver's migration policy.
// Compares four configurations on a 64 MB shared buffer:
//   (a) No hint (default)
//   (b) cudaMemAdviseSetReadMostly  — driver may duplicate pages; reduces migration
//   (c) cudaMemAdviseSetPreferredLocation (GPU) — pages default to GPU memory
//   (d) cudaMemAdviseSetAccessedBy (CPU) — CPU gets HW-assisted direct mapping
// Metric: CPU read latency after GPU has written the data.
// ---------------------------------------------------------------------------
void exp4_memory_advice(const std::string& outfile) {
    const int WARM  = 5;
    const int RUNS  = 40;
    const size_t N  = (64ULL << 20) / sizeof(float);
    const size_t bytes = N * sizeof(float);

    float* data;
    CHECK_CUDA(cudaMallocManaged(&data, bytes));
    for (size_t i = 0; i < N; i++) data[i] = 1.0f;

    int dev; CHECK_CUDA(cudaGetDevice(&dev));

    std::ofstream f(outfile);
    f << "advice,run,phase,latency_ns\n";

    struct AdviceConfig {
        std::string name;
        cudaMemoryAdvise advise;
        bool apply;
    };

    std::vector<AdviceConfig> configs = {
        {"no_hint",             cudaMemAdviseSetReadMostly, false},
        {"read_mostly",         cudaMemAdviseSetReadMostly, true},
        {"preferred_gpu",       cudaMemAdviseSetPreferredLocation, true},
        {"accessed_by_cpu",     cudaMemAdviseSetAccessedBy, true}
    };

    std::cout << "\n[EXP 4] Memory Advice\n";

    // Warm-up
    for (int w = 0; w < WARM; w++) {
        kernel_touch<<<((int)N + 255) / 256, 256>>>(data, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());
        cpu_seq_access((volatile float*)data, (int)N);
    }

    for (auto& cfg : configs) {
        // Reset advice
        CHECK_CUDA_UM(adviseGPU(data, bytes, cudaMemAdviseUnsetReadMostly, dev));
        CHECK_CUDA_UM(adviseGPU(data, bytes, cudaMemAdviseUnsetPreferredLocation, dev));
        CHECK_CUDA_UM(adviseGPU(data, bytes, cudaMemAdviseUnsetAccessedBy, dev));
        CHECK_CUDA_UM(adviseCPU(data, bytes, cudaMemAdviseUnsetAccessedBy));

        if (cfg.apply) {
            if (cfg.advise == cudaMemAdviseSetAccessedBy)
                CHECK_CUDA_UM(adviseCPU(data, bytes, cfg.advise));
            else
                CHECK_CUDA_UM(adviseGPU(data, bytes, cfg.advise, dev));
        }

        std::vector<long> cpu_lats, mig_lats;

        for (int r = 0; r < RUNS; r++) {
            CHECK_CUDA_UM(prefetchToCPU(data, bytes));
            CHECK_CUDA(cudaDeviceSynchronize());

            // CPU baseline read
            long t_cpu = cpu_seq_access((volatile float*)data, (int)N);
            cpu_lats.push_back(t_cpu);
            f << cfg.name << "," << r << ",cpu_baseline," << t_cpu << "\n";

            // GPU write → CPU re-read (migration)
            kernel_sequential_rw<<<((int)N + 255) / 256, 256>>>(data, (int)N, 1.0f);
            CHECK_CUDA(cudaDeviceSynchronize());
            long t_mig = cpu_seq_access((volatile float*)data, (int)N);
            mig_lats.push_back(t_mig);
            f << cfg.name << "," << r << ",post_migration," << t_mig << "\n";
        }

        Stats cs = compute_stats(cpu_lats);
        Stats ms = compute_stats(mig_lats);
        std::cout << "  [" << cfg.name << "]"
                  << "  CPU=" << (long)cs.mean << " ns"
                  << "  PostMig=" << (long)ms.mean << " ns"
                  << "  Overhead=" << (long)(ms.mean - cs.mean) << " ns\n";
    }

    std::cout << "  -> Saved: " << outfile << "\n";
    CHECK_CUDA(cudaFree(data));
}

// ---------------------------------------------------------------------------
// EXP 5 — Concurrent CPU+GPU Atomic Contention
// ---------------------------------------------------------------------------
// A shared atomic counter is incremented simultaneously by:
//   - GPU kernel (M threads × iters atomicAdd)
//   - CPU thread (serial loop of atomicAdd via system-wide atomics on UM)
// Measures additional latency on the CPU side vs. CPU-only baseline.
// Reveals cache-coherence protocol (PCIe/NVLink MESI-equivalent) overhead.
// ---------------------------------------------------------------------------
void exp5_concurrent_contention(const std::string& outfile) {
    const int WARM   = 3;
    const int RUNS   = 30;
    const int ITERS  = 10000;       // CPU-side atomic iterations per run
    const int GTHREADS = 1024;      // GPU threads
    const int GITERS   = 1000;      // GPU atomicAdds per thread

    unsigned long long* counter;
    CHECK_CUDA(cudaMallocManaged(&counter, sizeof(unsigned long long)));

    std::ofstream f(outfile);
    f << "scenario,run,latency_ns\n";

    std::cout << "\n[EXP 5] CPU+GPU Concurrent Contention\n";

    // Warm-up
    for (int w = 0; w < WARM; w++) {
        *counter = 0;
        kernel_atomic_add<<<4, 256>>>(counter, 100);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Use a separate std::atomic wrapper for portable CPU-side atomics
    // The GPU counter is still in UM; CPU uses volatile+loop to simulate
    // sequential coherence pressure (MSVC-safe, no GCC builtins needed).
    std::vector<long> cpu_only_lats, concurrent_lats;

    for (int r = 0; r < RUNS; r++) {
        // (a) CPU-only baseline atomic loop
        *counter = 0;
        volatile unsigned long long* vc = counter;
        long t0 = now_ns();
        for (int i = 0; i < ITERS; i++) {
            // Portable interlocked increment that works on MSVC + GCC
            #if defined(_MSC_VER)
                _InterlockedExchangeAdd64((volatile __int64*)vc, 1LL);
            #else
                __atomic_fetch_add(vc, 1ULL, __ATOMIC_SEQ_CST);
            #endif
        }
        long t1 = now_ns();
        cpu_only_lats.push_back(t1 - t0);
        f << "cpu_only," << r << "," << (t1 - t0) << "\n";

        // (b) Concurrent: GPU kernel runs during CPU atomic loop
        *counter = 0;
        // Launch GPU kernel asynchronously
        kernel_atomic_add<<<GTHREADS / 256, 256>>>(counter, GITERS);
        // CPU begins its own atomics immediately (contention window)
        long t2 = now_ns();
        for (int i = 0; i < ITERS; i++) {
            #if defined(_MSC_VER)
                _InterlockedExchangeAdd64((volatile __int64*)vc, 1LL);
            #else
                __atomic_fetch_add(vc, 1ULL, __ATOMIC_SEQ_CST);
            #endif
        }
        long t3 = now_ns();
        CHECK_CUDA(cudaDeviceSynchronize());
        concurrent_lats.push_back(t3 - t2);
        f << "cpu_gpu_concurrent," << r << "," << (t3 - t2) << "\n";
    }

    Stats cs = compute_stats(cpu_only_lats);
    Stats cc = compute_stats(concurrent_lats);
    std::cout << "  CPU-only     mean=" << (long)cs.mean << " ns"
              << "  p99=" << (long)cs.p99 << " ns\n";
    std::cout << "  Concurrent   mean=" << (long)cc.mean << " ns"
              << "  p99=" << (long)cc.p99 << " ns\n";
    std::cout << "  Contention overhead: " << (long)(cc.mean - cs.mean) << " ns  ("
              << (cc.mean / cs.mean) << "x slowdown)\n";
    std::cout << "  -> Saved: " << outfile << "\n";

    CHECK_CUDA(cudaFree(counter));
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Print device info
    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::cout << "============================================================\n";
    std::cout << " CUDA Unified Memory Latency Profiler\n";
    std::cout << "============================================================\n";
    std::cout << " Device : " << prop.name << "\n";
    std::cout << " SM Arch: sm_" << prop.major << prop.minor << "\n";
    std::cout << " GMEM   : " << prop.totalGlobalMem / (1 << 20) << " MB\n";
    std::cout << " SMs    : " << prop.multiProcessorCount << "\n";
    std::cout << " UM Concurrent Access: "
              << (prop.concurrentManagedAccess ? "YES" : "NO") << "\n";
    std::cout << "============================================================\n";

    // Select which experiments to run (all by default, or via argv)
    bool run_all = (argc == 1);
    auto should_run = [&](int exp) {
        if (run_all) return true;
        for (int i = 1; i < argc; i++)
            if (std::atoi(argv[i]) == exp) return true;
        return false;
    };

    if (should_run(1)) exp1_size_sweep      ("size_sweep.csv");
    if (should_run(2)) exp2_access_patterns ("patterns.csv");
    if (should_run(3)) exp3_prefetch        ("prefetch.csv");
    if (should_run(4)) exp4_memory_advice   ("advice.csv");
    if (should_run(5)) exp5_concurrent_contention("concurrent.csv");

    std::cout << "\n[DONE] All experiments complete. Run 'python analyze.py' to generate plots.\n";
    return 0;
}