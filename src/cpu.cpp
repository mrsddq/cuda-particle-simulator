#include "particles.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace particles {
int grid_width(const Parameters& p) {
    if (!std::isfinite(p.domain) || p.domain <= 0 || p.domain > 10000 ||
        !std::isfinite(p.radius) || p.radius <= 0 || p.radius > p.domain / 2)
        throw std::invalid_argument("domain must be in (0,10000], radius in (0,domain/2]");
    const double cells = std::ceil(static_cast<double>(p.domain) / (2.0 * p.radius));
    if (cells > 1024) throw std::invalid_argument("grid exceeds 1024 cells per axis");
    return static_cast<int>(cells);
}

void validate(const std::vector<Particle>& state, const Parameters& p) {
    grid_width(p);
    if (!std::isfinite(p.dt) || p.dt <= 0 || p.dt > 0.1f ||
        !std::isfinite(p.stiffness) || p.stiffness < 0 || p.stiffness > 1e6f)
        throw std::invalid_argument("dt must be in (0,0.1], stiffness in [0,1e6]");
    if (state.empty() || state.size() > 1000000)
        throw std::invalid_argument("particle count must be in [1,1000000]");
    for (const auto& q : state)
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.vx) ||
            !std::isfinite(q.vy) || q.x < 0 || q.x > p.domain || q.y < 0 || q.y > p.domain)
            throw std::invalid_argument("particles must be finite and inside the domain");
}

std::vector<Particle> seeded_state(int count, std::uint32_t seed, float domain) {
    if (count < 1 || count > 1000000 || !std::isfinite(domain) || domain <= 0)
        throw std::invalid_argument("invalid count or domain");
    std::mt19937 rng(seed);
    // Map the specified generator's high 24 bits directly, independent of distribution libraries.
    auto unit = [&]() { return static_cast<float>(rng() >> 8) / 16777216.0f; };
    std::vector<Particle> result;
    result.reserve(count);
    for (int i = 0; i < count; ++i)
        result.push_back({unit() * domain, unit() * domain,
                          (unit() - 0.5f) * 0.2f, (unit() - 0.5f) * 0.2f});
    return result;
}

static void force(const Particle& a, const Particle& b, int i, int j,
                  const Parameters& p, float& fx, float& fy) {
    if (i == j) return;
    float dx = a.x - b.x, dy = a.y - b.y;
    const float distance2 = dx * dx + dy * dy;
    const float cutoff = 2 * p.radius;
    if (distance2 >= cutoff * cutoff) return;
    if (distance2 == 0) {
        // Coincident pairs receive deterministic, antisymmetric x impulses.
        fx += (i < j ? -1.0f : 1.0f) * p.stiffness * cutoff;
    } else {
        const float distance = std::sqrt(distance2);
        const float scale = p.stiffness * (cutoff - distance) / distance;
        fx += scale * dx;
        fy += scale * dy;
    }
}

static void reflect(float& position, float& velocity, float domain) {
    float phase = std::fmod(position, 2 * domain);
    if (phase < 0) phase += 2 * domain;
    if (phase > domain) { position = 2 * domain - phase; velocity = -velocity; }
    else position = phase;
    if (position == 0 && velocity < 0) velocity = -velocity;
    if (position == domain && velocity > 0) velocity = -velocity;
}

static Particle advance(Particle q, float fx, float fy, const Parameters& p) {
    q.vx += fx * p.dt;
    q.vy += fy * p.dt;
    q.x += q.vx * p.dt;
    q.y += q.vy * p.dt;
    reflect(q.x, q.vx, p.domain);
    reflect(q.y, q.vy, p.domain);
    return q;
}

std::vector<Particle> simulate_naive(std::vector<Particle> state, const Parameters& p, int steps) {
    validate(state, p);
    if (steps < 0) throw std::invalid_argument("steps must be nonnegative");
    std::vector<Particle> next(state.size());
    for (int step = 0; step < steps; ++step) {
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            float fx = 0, fy = 0;
            for (int j = 0; j < static_cast<int>(state.size()); ++j) force(state[i], state[j], i, j, p, fx, fy);
            next[i] = advance(state[i], fx, fy, p);
        }
        state.swap(next);
    }
    validate(state, p);
    return state;
}

std::vector<Particle> simulate_grid(std::vector<Particle> state, const Parameters& p, int steps) {
    validate(state, p);
    if (steps < 0) throw std::invalid_argument("steps must be nonnegative");
    const int width = grid_width(p);
    const float cell_size = 2 * p.radius;
    auto cell = [&](float position) { return std::min(static_cast<int>(position / cell_size), width - 1); };
    std::vector<int> heads(width * width), links(state.size());
    std::vector<Particle> next(state.size());
    for (int step = 0; step < steps; ++step) {
        std::fill(heads.begin(), heads.end(), -1);
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            const int id = cell(state[i].y) * width + cell(state[i].x);
            links[i] = heads[id];
            heads[id] = i;
        }
        for (int i = 0; i < static_cast<int>(state.size()); ++i) {
            float fx = 0, fy = 0;
            const int x = cell(state[i].x), y = cell(state[i].y);
            for (int cy = std::max(0, y - 1); cy <= std::min(width - 1, y + 1); ++cy)
                for (int cx = std::max(0, x - 1); cx <= std::min(width - 1, x + 1); ++cx)
                    for (int j = heads[cy * width + cx]; j != -1; j = links[j])
                        force(state[i], state[j], i, j, p, fx, fy);
            next[i] = advance(state[i], fx, fy, p);
        }
        state.swap(next);
    }
    validate(state, p);
    return state;
}
}  // namespace particles
