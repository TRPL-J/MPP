#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

using ulong = unsigned long;

// Standalone mock matching the exact mathematical workload of ModelHawkesLogLikSingle::hessian_i
void mock_tick_hessian_i(
    ulong i,
    const std::vector<double>& coeffs,
    const std::vector<std::vector<double>>& g_i,
    ulong n_jumps,
    ulong n_nodes,
    ulong n_alpha_i,
    std::vector<double>& out
) {
    auto get_alpha_i_first_index = [&](ulong node_idx) { return n_nodes + node_idx * n_alpha_i; };
    auto get_alpha_i_last_index = [&](ulong node_idx) { return n_nodes + (node_idx + 1) * n_alpha_i; };

    const double mu_i = coeffs[i];
    const ulong start_mu_line = i * (n_alpha_i + 1);
    const ulong block_start = (n_nodes + i * n_alpha_i) * (n_alpha_i + 1);
    const ulong alpha_start = get_alpha_i_first_index(i);

    for (ulong k = 0; k < n_jumps; ++k) {
        const std::vector<double>& g_i_k = g_i[k];

        double s = mu_i;
        for (ulong j = 0; j < n_alpha_i; ++j) {
            s += coeffs[alpha_start + j] * g_i_k[j];
        }

        const double s_2 = s * s;

        // 1. Fill mu mu
        out[start_mu_line] += 1.0 / s_2;
        
        // 2. Fill mu alpha
        for (ulong j = 0; j < n_alpha_i; ++j) {
            out[start_mu_line + j + 1] += g_i_k[j] / s_2;
        }

        // 3. Fill alpha mu and alpha square
        for (ulong l = 0; l < n_alpha_i; ++l) {
            const ulong start_alpha_line = block_start + l * (n_alpha_i + 1);
            out[start_alpha_line] += g_i_k[l] / s_2;
            
            for (ulong m = 0; m < n_alpha_i; ++m) {
                out[start_alpha_line + m + 1] += (g_i_k[l] * g_i_k[m]) / s_2;
            }
        }
    }
}

void fill_random(std::vector<double>& vec) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.1, 1.0);
    for (auto& val : vec) val = dist(rng);
}

int main() {
    std::cout << "--- Standalone CPU Hessian Benchmark ---" << std::endl;
    std::cout << std::left << std::setw(15) << "Jumps (K)"
              << std::setw(15) << "Alphas (A)"
              << std::setw(15) << "Time (ms)" << std::endl;

    std::vector<ulong> jump_scales = {1000, 5000, 10000, 50000};
    std::vector<ulong> alpha_scales = {10, 50, 100};

    const ulong n_nodes = 10;
    const ulong i = 0;

    for (ulong n_jumps : jump_scales) {
        for (ulong n_alpha_i : alpha_scales) {

            std::vector<double> coeffs(n_nodes + (n_nodes * n_alpha_i));
            fill_random(coeffs);

            std::vector<std::vector<double>> g_i(n_jumps, std::vector<double>(n_alpha_i));
            for (auto& jump : g_i) fill_random(jump);

            ulong out_size = (n_nodes + n_nodes * n_alpha_i) * (n_alpha_i + 1);
            std::vector<double> out(out_size, 0.0);

            auto start = std::chrono::high_resolution_clock::now();

            mock_tick_hessian_i(i, coeffs, g_i, n_jumps, n_nodes, n_alpha_i, out);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;

            std::cout << std::left << std::setw(15) << n_jumps
                      << std::setw(15) << n_alpha_i
                      << std::setw(15) << duration.count() << std::endl;
        }
    }
    return 0;
}