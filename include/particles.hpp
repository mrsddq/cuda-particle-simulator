#pragma once

#include <cstdint>
#include <vector>

namespace particles {
struct Particle { float x, y, vx, vy; };
struct Parameters {
    float domain = 1.0f;
    float radius = 0.02f;
    float stiffness = 25.0f;
    float dt = 0.001f;
};

void validate(const std::vector<Particle>& state, const Parameters& parameters);
int grid_width(const Parameters& parameters);
std::vector<Particle> seeded_state(int count, std::uint32_t seed, float domain);
std::vector<Particle> simulate_naive(std::vector<Particle> state, const Parameters&, int steps);
std::vector<Particle> simulate_grid(std::vector<Particle> state, const Parameters&, int steps);

struct GpuResult {
    std::vector<Particle> state;
    double device_ms = 0;
    double transfer_ms = 0;
};
// Defined only in the optional CUDA target. Other CUDA errors are exceptions, not skips.
bool cuda_available();
GpuResult simulate_cuda(const std::vector<Particle>& state, const Parameters&, int steps);
}  // namespace particles
