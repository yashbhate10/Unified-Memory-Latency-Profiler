// profiler.cu

#include <iostream>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(call) << std::endl; \
        exit(1); \
    }

__global__ void touchKernel(float *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        data[i] += 1.0f;
    }
}

long measure_cpu_access(float *data, int N) {
    volatile float sum = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        sum += data[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main() {
    const int N = 1 << 22; // ~4M elements (large enough for migration)
    const int RUNS = 50;

    float *data;
    CHECK_CUDA(cudaMallocManaged(&data, N * sizeof(float)));

    // Initialize
    for (int i = 0; i < N; i++) data[i] = 1.0f;

    // Warm-up (important for stable results)
    touchKernel<<<(N+255)/256, 256>>>(data, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::ofstream file("results.csv");
    file << "iteration,latency_ns,type\n";

    for (int i = 0; i < RUNS; i++) {

        // ---- CASE 1: CPU-only access (baseline) ----
        long cpu_latency = measure_cpu_access(data, N);
        std::cout << "[CPU] Iter " << i << ": " << cpu_latency << " ns\n";
        file << i << "," << cpu_latency << ",cpu\n";

        // ---- CASE 2: GPU then CPU (migration expected) ----
        touchKernel<<<(N+255)/256, 256>>>(data, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        long gpu_latency = measure_cpu_access(data, N);
        std::cout << "[GPU->CPU] Iter " << i << ": " << gpu_latency << " ns\n";
        file << i << "," << gpu_latency << ",gpu\n";
    }

    file.close();
    CHECK_CUDA(cudaFree(data));

    std::cout << "\n[INFO] Results saved to results.csv\n";
    return 0;
}