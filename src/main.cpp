#include "particles.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static int integer(const std::string& value) {
    std::size_t used = 0;
    const int parsed = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("invalid integer: " + value);
    return parsed;
}
int main(int argc, char** argv) {
    try {
        int count = 1024, steps = 100, seed = 42;
        std::string backend = "cpu-grid", output;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help") {
                std::cout << "particle_sim [--backend cpu-grid|cpu-naive|cuda] [--count N] [--steps N] [--seed N] [--output snapshot.csv]\n";
                return 0;
            }
            if (++i == argc) throw std::invalid_argument("missing value for " + key);
            const std::string value = argv[i];
            if (key == "--count") count = integer(value);
            else if (key == "--steps") steps = integer(value);
            else if (key == "--seed") seed = integer(value);
            else if (key == "--backend") backend = value;
            else if (key == "--output") output = value;
            else throw std::invalid_argument("unknown argument: " + key);
        }
        if (steps < 0 || steps > 100000 || seed < 0) throw std::invalid_argument("invalid steps or seed");
        particles::Parameters p;
        auto state = particles::seeded_state(count, static_cast<unsigned>(seed), p.domain);
        const auto start = std::chrono::steady_clock::now();
        double device_ms = -1, transfer_ms = -1;
        if (backend == "cpu-grid") state = particles::simulate_grid(state, p, steps);
        else if (backend == "cpu-naive") state = particles::simulate_naive(state, p, steps);
        else if (backend == "cuda") {
#ifdef HAS_CUDA
            auto result = particles::simulate_cuda(state, p, steps);
            state = std::move(result.state);
            device_ms = result.device_ms;
            transfer_ms = result.transfer_ms;
#else
            throw std::invalid_argument("CUDA backend was not compiled; configure ENABLE_CUDA=ON");
#endif
        } else throw std::invalid_argument("unknown backend: " + backend);
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        double checksum = 0;
        for (const auto& q : state) checksum += q.x + q.y + q.vx + q.vy;
        std::cout << std::setprecision(10) << "backend,count,steps,total_ms,device_ms,transfer_ms,checksum\n"
                  << backend << ',' << count << ',' << steps << ',' << total_ms << ','
                  << device_ms << ',' << transfer_ms << ',' << checksum << '\n';
        if (!output.empty()) {
            std::ofstream file(output);
            if (!file) throw std::runtime_error("cannot open output: " + output);
            file << std::setprecision(9) << "id,x,y,vx,vy\n";
            for (std::size_t i = 0; i < state.size(); ++i) {
                const auto& q = state[i];
                file << i << ',' << q.x << ',' << q.y << ',' << q.vx << ',' << q.vy << '\n';
            }
            if (!file) throw std::runtime_error("failed to write output");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
