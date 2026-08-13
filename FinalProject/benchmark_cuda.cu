#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

// ==========================================
// IMPLEMENTATION 1: Baseline (Global Atomics)
// ==========================================
// In this baseline kernel, each block corresponds to a single network node 'i'.
// Threads within the block process jump events using a block-stride loop and 
// write their computed partial results directly to global memory using atomic additions.
__global__ void hessian_kernel_baseline(
    const double* coeffs,
    const double* g_flat,
    const size_t* jump_offsets,
    size_t n_nodes,
    size_t n_alpha_i,
    double* out
) {
    // Each block processes a specific node index 'i'
    size_t i = blockIdx.x;
    if (i >= n_nodes) return;

    // Retrieve jump boundaries for node 'i' from the offset array
    size_t jump_start_idx = jump_offsets[i];
    size_t n_jumps = jump_offsets[i + 1] - jump_offsets[i];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const double mu_i = coeffs[i];
    const size_t start_mu_line = i * (n_alpha_i + 1);
    const size_t block_start = (n_nodes + i * n_alpha_i) * (n_alpha_i + 1);

    // Block-strided loop: threads cooperate to process all jumps for node 'i'
    for (size_t k = tid; k < n_jumps; k += stride) {
        size_t g_base = (jump_start_idx + k) * n_alpha_i;

        // Compute intensity scaling factor 's' for the current jump
        double s = mu_i;
        for (size_t j = 0; j < n_alpha_i; ++j) {
            s += coeffs[n_nodes + i * n_alpha_i + j] * g_flat[g_base + j];
        }

        const double s_2 = s * s;
        const double inv_s2 = 1.0 / s_2;

        // Directly update global memory using atomic additions to avoid race conditions
        atomicAdd(&out[start_mu_line], inv_s2);

        for (size_t j = 0; j < n_alpha_i; ++j) {
            atomicAdd(&out[start_mu_line + j + 1], g_flat[g_base + j] * inv_s2);
        }

        for (size_t l = 0; l < n_alpha_i; ++l) {
            const size_t start_alpha_line = block_start + l * (n_alpha_i + 1);

            atomicAdd(&out[start_alpha_line], g_flat[g_base + l] * inv_s2);

            for (size_t m = 0; m < n_alpha_i; ++m) {
                atomicAdd(&out[start_alpha_line + m + 1], (g_flat[g_base + l] * g_flat[g_base + m]) * inv_s2);
            }
        }
    }
}

// ==========================================
// IMPLEMENTATION 2: Shared Memory Accumulation
// ==========================================
// This approach reduces global memory traffic by using fast, on-chip shared memory 
// to accumulate partial matrix elements locally before writing them out once per block.
__global__ void hessian_kernel_shared(
    const double* coeffs,
    const double* g_flat,
    const size_t* jump_offsets,
    size_t n_nodes,
    size_t n_alpha_i,
    double* out
) {
    size_t i = blockIdx.x;
    if (i >= n_nodes) return;

    size_t jump_start_idx = jump_offsets[i];
    size_t n_jumps = jump_offsets[i + 1] - jump_offsets[i];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const double mu_i = coeffs[i];
    const size_t mat_dim = n_alpha_i + 1;
    const size_t block_out_size = mat_dim * mat_dim;

    // Dynamically allocated shared memory buffer for the Hessian block matrix
    extern __shared__ double s_mem[];

    // Initialize shared memory accumulator elements to zero
    for (size_t idx = tid; idx < block_out_size; idx += stride) {
        s_mem[idx] = 0.0;
    }
    __syncthreads(); // Ensure all shared memory is initialized before threads proceed

    const size_t local_mu_offset = 0;
    const size_t local_alpha_offset = mat_dim;

    // Accumulate jump calculations into fast shared memory using atomics
    for (size_t k = tid; k < n_jumps; k += stride) {
        size_t g_base = (jump_start_idx + k) * n_alpha_i;

        double s = mu_i;
        for (size_t j = 0; j < n_alpha_i; ++j) {
            s += coeffs[n_nodes + i * n_alpha_i + j] * g_flat[g_base + j];
        }

        const double s_2 = s * s;
        const double inv_s2 = 1.0 / s_2;

        atomicAdd(&s_mem[local_mu_offset], inv_s2);

        for (size_t j = 0; j < n_alpha_i; ++j) {
            atomicAdd(&s_mem[local_mu_offset + j + 1], g_flat[g_base + j] * inv_s2);
        }

        for (size_t l = 0; l < n_alpha_i; ++l) {
            const size_t row_base = local_alpha_offset + l * mat_dim;

            atomicAdd(&s_mem[row_base], g_flat[g_base + l] * inv_s2);

            for (size_t m = 0; m < n_alpha_i; ++m) {
                atomicAdd(&s_mem[row_base + m + 1], (g_flat[g_base + l] * g_flat[g_base + m]) * inv_s2);
            }
        }
    }
    __syncthreads(); // Synchronize before writing shared memory results back to global memory

    size_t start_mu_line = i * mat_dim;
    size_t block_start = (n_nodes + i * n_alpha_i) * mat_dim;

    // Transfer aggregated results from shared memory back to global output VRAM
    for (size_t idx = tid; idx < mat_dim; idx += stride) {
        atomicAdd(&out[start_mu_line + idx], s_mem[local_mu_offset + idx]);
    }

    size_t alpha_total_elements = n_alpha_i * mat_dim;
    for (size_t idx = tid; idx < alpha_total_elements; idx += stride) {
        atomicAdd(&out[block_start + idx], s_mem[local_alpha_offset + idx]);
    }
}

// ==========================================
// IMPLEMENTATION 3: Register-Cached Inner Loop Variant
// ==========================================
// Enhances the shared memory kernel by adding compiler directives like loop unrolling
// to optimize instruction pipeline efficiency and encourage register caching.
__global__ void hessian_kernel_optimized_registers(
    const double* coeffs,
    const double* g_flat,
    const size_t* jump_offsets,
    size_t n_nodes,
    size_t n_alpha_i,
    double* out
) {
    size_t i = blockIdx.x;
    if (i >= n_nodes) return;

    size_t jump_start_idx = jump_offsets[i];
    size_t n_jumps = jump_offsets[i + 1] - jump_offsets[i];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    const double mu_i = coeffs[i];
    const size_t mat_dim = n_alpha_i + 1;
    const size_t block_out_size = mat_dim * mat_dim;

    extern __shared__ double s_mem[];

    for (size_t idx = tid; idx < block_out_size; idx += stride) {
        s_mem[idx] = 0.0;
    }
    __syncthreads();

    const size_t local_mu_offset = 0;
    const size_t local_alpha_offset = mat_dim;

    for (size_t k = tid; k < n_jumps; k += stride) {
        size_t g_base = (jump_start_idx + k) * n_alpha_i;

        double s = mu_i;
        // #pragma unroll 4 instructs the compiler to unroll this inner loop by a factor of 4
        // to reduce branching overhead and expose instruction-level parallelism.
        #pragma unroll 4
        for (size_t j = 0; j < n_alpha_i; ++j) {
            s += coeffs[n_nodes + i * n_alpha_i + j] * g_flat[g_base + j];
        }

        const double inv_s2 = 1.0 / (s * s);

        atomicAdd(&s_mem[local_mu_offset], inv_s2);

        for (size_t j = 0; j < n_alpha_i; ++j) {
            atomicAdd(&s_mem[local_mu_offset + j + 1], g_flat[g_base + j] * inv_s2);
        }

        for (size_t l = 0; l < n_alpha_i; ++l) {
            const size_t row_base = local_alpha_offset + l * mat_dim;
            // Cache gradient values in local registers to avoid redundant global reads
            double g_l = g_flat[g_base + l];

            atomicAdd(&s_mem[row_base], g_l * inv_s2);

            #pragma unroll 4
            for (size_t m = 0; m < n_alpha_i; ++m) {
                atomicAdd(&s_mem[row_base + m + 1], (g_l * g_flat[g_base + m]) * inv_s2);
            }
        }
    }
    __syncthreads();

    size_t start_mu_line = i * mat_dim;
    size_t block_start = (n_nodes + i * n_alpha_i) * mat_dim;

    for (size_t idx = tid; idx < mat_dim; idx += stride) {
        atomicAdd(&out[start_mu_line + idx], s_mem[local_mu_offset + idx]);
    }

    size_t alpha_total_elements = n_alpha_i * mat_dim;
    for (size_t idx = tid; idx < alpha_total_elements; idx += stride) {
        atomicAdd(&out[block_start + idx], s_mem[local_alpha_offset + idx]);
    }
}

// ==========================================
// IMPLEMENTATION 4: Warp-Level Shuffle Reduction
// ==========================================

// 1. Define the Warp Reduction Primitive
__inline__ __device__ double warpReduceSum(double val) {
    // A warp consists of 32 threads. We fold data in half across lanes (16 -> 8 -> 4 -> 2 -> 1).
    // __shfl_down_sync pulls a register value from a sibling thread 'offset' lanes away in the warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 2. The Optimized Kernel
__global__ void hessian_kernel_warp_optimized(
    const double* coeffs,
    const double* g_flat,
    const size_t* jump_offsets,
    size_t n_nodes,
    size_t n_alpha_i,
    double* out
) {
    size_t i = blockIdx.x;
    if (i >= n_nodes) return;

    size_t jump_start_idx = jump_offsets[i];
    size_t n_jumps = jump_offsets[i + 1] - jump_offsets[i];

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    
    // Identify the specific thread lane (0 to 31) within its 32-thread warp
    unsigned int lane_id = tid % 32;

    const double mu_i = coeffs[i];
    const size_t mat_dim = n_alpha_i + 1;
    const size_t start_mu_line = i * mat_dim;
    const size_t block_start = (n_nodes + i * n_alpha_i) * mat_dim;

    for (size_t k_base = 0; k_base < n_jumps; k_base += stride) {
        size_t k = k_base + tid;
        
        // CRITICAL: Avoid early return branches. All 32 warp threads must reach 
        // the warp shuffle instruction together to prevent deadlocks (SIMT execution constraint).
        bool valid_jump = (k < n_jumps);
        
        double inv_s2 = 0.0;
        size_t g_base = 0;

        if (valid_jump) {
            g_base = (jump_start_idx + k) * n_alpha_i;
            double s = mu_i;
            for (size_t j = 0; j < n_alpha_i; ++j) {
                s += coeffs[n_nodes + i * n_alpha_i + j] * g_flat[g_base + j];
            }
            inv_s2 = 1.0 / (s * s);
        }

        // --- 1. Warp-reduce the mu_i term across the 32 threads ---
        double val_mu = valid_jump ? inv_s2 : 0.0;
        double sum_mu = warpReduceSum(val_mu);
        // Only lane 0 performs the atomic add to global memory for the entire warp
        if (lane_id == 0) atomicAdd(&out[start_mu_line], sum_mu);

        // --- 2. Warp-reduce the alpha vector components ---
        for (size_t j = 0; j < n_alpha_i; ++j) {
            double val_alpha = valid_jump ? (g_flat[g_base + j] * inv_s2) : 0.0;
            double sum_alpha = warpReduceSum(val_alpha);
            if (lane_id == 0) atomicAdd(&out[start_mu_line + j + 1], sum_alpha);
        }

        // --- 3. Warp-reduce the outer product matrix (A x A) ---
        for (size_t l = 0; l < n_alpha_i; ++l) {
            double g_l = valid_jump ? g_flat[g_base + l] : 0.0;
            
            double val_row_head = valid_jump ? (g_l * inv_s2) : 0.0;
            double sum_row_head = warpReduceSum(val_row_head);
            const size_t start_alpha_line = block_start + l * mat_dim;
            
            if (lane_id == 0) atomicAdd(&out[start_alpha_line], sum_row_head);

            for (size_t m = 0; m < n_alpha_i; ++m) {
                double val_matrix = valid_jump ? ((g_l * g_flat[g_base + m]) * inv_s2) : 0.0;
                double sum_matrix = warpReduceSum(val_matrix);
                
                if (lane_id == 0) {
                    atomicAdd(&out[start_alpha_line + m + 1], sum_matrix);
                }
            }
        }
    }
}

// Helper utility to initialize vectors with pseudo-random numbers
void fill_random(std::vector<double>& vec, double min = 0.1, double max = 1.0) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(min, max);
    for (auto& val : vec) val = dist(rng);
}

int main() {
    std::cout << "======================================================================================" << std::endl;
    std::cout << "                 MULTI-IMPLEMENTATION CUDA HESSIAN BENCHMARK SUITE                    " << std::endl;
    std::cout << "======================================================================================" << std::endl;
    std::cout << std::left << std::setw(12) << "Jumps/Node"
              << std::setw(10) << "Alpha (A)"
              << std::setw(18) << "Baseline (ms)"
              << std::setw(18) << "Shared Mem (ms)"
              << std::setw(18) << "Reg-Unrolled (ms)"
              << std::setw(18) << "Warp-Reduced (ms)" << std::endl;
    std::cout << "--------------------------------------------------------------------------------------" << std::endl;

    std::vector<size_t> jump_scales = {1000, 5000, 10000, 50000};
    std::vector<size_t> alpha_scales = {10, 50, 100};
    const size_t n_nodes = 10;
    const size_t threads_per_block = 256;

    // Set maximum dynamic shared memory attributes for kernels that need large allocations
    cudaFuncSetAttribute(hessian_kernel_shared, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
    cudaFuncSetAttribute(hessian_kernel_optimized_registers, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    for (size_t n_jumps : jump_scales) {
        for (size_t n_alpha_i : alpha_scales) {

            // 1. Host Data Setup & Initialization
            std::vector<double> h_coeffs(n_nodes + (n_nodes * n_alpha_i));
            fill_random(h_coeffs);

            std::vector<size_t> h_jump_offsets(n_nodes + 1, 0);
            for (size_t n = 0; n < n_nodes; ++n) {
                h_jump_offsets[n + 1] = h_jump_offsets[n] + n_jumps;
            }

            size_t total_jumps = h_jump_offsets.back();
            std::vector<double> h_g_flat(total_jumps * n_alpha_i);
            fill_random(h_g_flat);

            size_t out_size = (n_nodes + n_nodes * n_alpha_i) * (n_alpha_i + 1);
            std::vector<double> h_out_baseline(out_size, 0.0);
            std::vector<double> h_out_shared(out_size, 0.0);
            std::vector<double> h_out_optimized(out_size, 0.0);
            std::vector<double> h_out_warp(out_size, 0.0);

            // 2. Device Memory Allocation on the GPU
            double *d_coeffs, *d_g_flat, *d_out_b, *d_out_s, *d_out_o, *d_out_w;
            size_t *d_jump_offsets;

            cudaMalloc(&d_coeffs, h_coeffs.size() * sizeof(double));
            cudaMalloc(&d_g_flat, h_g_flat.size() * sizeof(double));
            cudaMalloc(&d_jump_offsets, h_jump_offsets.size() * sizeof(size_t));
            cudaMalloc(&d_out_b, h_out_baseline.size() * sizeof(double));
            cudaMalloc(&d_out_s, h_out_shared.size() * sizeof(double));
            cudaMalloc(&d_out_o, h_out_optimized.size() * sizeof(double));
            cudaMalloc(&d_out_w, h_out_warp.size() * sizeof(double));

            // Copy input data from Host (CPU) to Device (GPU)
            cudaMemcpy(d_coeffs, h_coeffs.data(), h_coeffs.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_g_flat, h_g_flat.data(), h_g_flat.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_jump_offsets, h_jump_offsets.data(), h_jump_offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            // --- Run Implementation 1: Baseline Kernel ---
            cudaMemcpy(d_out_b, h_out_baseline.data(), h_out_baseline.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            hessian_kernel_baseline<<<n_nodes, threads_per_block>>>(
                d_coeffs, d_g_flat, d_jump_offsets, n_nodes, n_alpha_i, d_out_b
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_baseline = 0;
            cudaEventElapsedTime(&time_baseline, start, stop);

            // --- Run Implementation 2: Shared Memory Kernel ---
            cudaMemcpy(d_out_s, h_out_shared.data(), h_out_shared.size() * sizeof(double), cudaMemcpyHostToDevice);
            size_t shared_mem_bytes = (n_alpha_i + 1) * (n_alpha_i + 1) * sizeof(double);
            cudaEventRecord(start);
            hessian_kernel_shared<<<n_nodes, threads_per_block, shared_mem_bytes>>>(
                d_coeffs, d_g_flat, d_jump_offsets, n_nodes, n_alpha_i, d_out_s
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_shared = 0;
            cudaEventElapsedTime(&time_shared, start, stop);

            // --- Run Implementation 3: Register-Unrolled Kernel ---
            cudaMemcpy(d_out_o, h_out_optimized.data(), h_out_optimized.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            hessian_kernel_optimized_registers<<<n_nodes, threads_per_block, shared_mem_bytes>>>(
                d_coeffs, d_g_flat, d_jump_offsets, n_nodes, n_alpha_i, d_out_o
            );

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_optimized = 0;
            cudaEventElapsedTime(&time_optimized, start, stop);

            // --- Run Implementation 4: Warp-Reduced Kernel ---
            cudaMemcpy(d_out_w, h_out_warp.data(), h_out_warp.size() * sizeof(double), cudaMemcpyHostToDevice);
            cudaEventRecord(start);
            
            hessian_kernel_warp_optimized<<<n_nodes, threads_per_block>>>(
                d_coeffs, d_g_flat, d_jump_offsets, n_nodes, n_alpha_i, d_out_w
            );
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float time_warp = 0;
            cudaEventElapsedTime(&time_warp, start, stop);

            // Print benchmark results for the current scale configuration
            std::cout << std::left << std::setw(12) << n_jumps
                      << std::setw(10) << n_alpha_i
                      << std::setw(18) << std::fixed << std::setprecision(3) << time_baseline
                      << std::setw(18) << time_shared
                      << std::setw(18) << time_optimized 
                      << std::setw(18) << time_warp << std::endl;

            // Clean up device memory and CUDA events
            cudaFree(d_coeffs);
            cudaFree(d_g_flat);
            cudaFree(d_jump_offsets);
            cudaFree(d_out_b);
            cudaFree(d_out_s);
            cudaFree(d_out_o);
            cudaFree(d_out_w);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
    std::cout << "======================================================================================" << std::endl;
    return 0;
}