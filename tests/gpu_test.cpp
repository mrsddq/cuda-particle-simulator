#include "particles.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void compare(const std::vector<particles::Particle>& input, particles::Parameters p, int steps) {
    const auto expected = particles::simulate_naive(input, p, steps);
    const auto actual = particles::simulate_cuda(input, p, steps).state;
    if (actual.size() != expected.size()) throw std::runtime_error("particle lost");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float a[] = {actual[i].x, actual[i].y, actual[i].vx, actual[i].vy};
        const float b[] = {expected[i].x, expected[i].y, expected[i].vx, expected[i].vy};
        for (int k = 0; k < 4; ++k)
            if (!std::isfinite(a[k]) || std::fabs(a[k] - b[k]) > 3e-4f)
                throw std::runtime_error("CUDA/reference mismatch");
    }
}
int main() {
    try {
        if (!particles::cuda_available()) { std::cout << "SKIP: no CUDA device/driver\n"; return 77; }
        particles::Parameters p;
        for (int n : {1, 17, 257, 1025}) compare(particles::seeded_state(n, 42, 1), p, 8);
        compare(std::vector<particles::Particle>(513, {0.5f, 0.5f, 0, 0}), p, 1);
        compare({{0, 0, -1, -1}, {1, 1, 1, 1}, {0.0399f, 0.5f, 0, 0},
                 {0.0401f, 0.5f, 0, 0}}, p, 3);
        compare(particles::seeded_state(17, 42, 1), p, 0);
        std::cout << "CUDA/reference tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
