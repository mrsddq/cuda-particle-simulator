#include "particles.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
static void close(float actual, float expected, float tolerance = 2e-5f) {
    require(std::isfinite(actual) && std::fabs(actual - expected) <= tolerance, "numeric mismatch");
}
static void compare(const std::vector<particles::Particle>& input, const particles::Parameters& p, int steps) {
    const auto expected = particles::simulate_naive(input, p, steps);
    const auto actual = particles::simulate_grid(input, p, steps);
    require(actual.size() == expected.size(), "particle lost");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        close(actual[i].x, expected[i].x); close(actual[i].y, expected[i].y);
        close(actual[i].vx, expected[i].vx); close(actual[i].vy, expected[i].vy);
    }
}
int main() {
    try {
        particles::Parameters p;
        p.radius = 0.1f;
        p.stiffness = 10;
        p.dt = 0.01f;
        const auto pair = particles::simulate_grid({{0.4f, 0.5f, 0, 0}, {0.5f, 0.5f, 0, 0}}, p, 1);
        // F = k*(2r-d) = 1, then semi-implicit Euler with dt=.01.
        close(pair[0].vx, -0.01f); close(pair[1].vx, 0.01f);
        close(pair[0].x, 0.3999f); close(pair[1].x, 0.5001f);
        close(pair[0].vx + pair[1].vx, 0);

        const auto coincident = particles::simulate_grid({{0.5f, 0.5f, 0, 0}, {0.5f, 0.5f, 0, 0}}, p, 1);
        close(coincident[0].vx, -0.02f); close(coincident[1].vx, 0.02f);
        close(coincident[0].vy, 0); close(coincident[1].vy, 0);

        particles::Parameters wall;
        wall.stiffness = 0;
        wall.dt = 0.1f;
        const auto bounced = particles::simulate_grid({{0.95f, 0.02f, 1, -1}}, wall, 1);
        close(bounced[0].x, 0.95f); close(bounced[0].vx, -1);
        close(bounced[0].y, 0.08f); close(bounced[0].vy, 1);
        const auto multi = particles::simulate_grid({{0.2f, 0.8f, 37, -37}}, wall, 1);
        close(multi[0].x, 0.1f); close(multi[0].vx, -37);
        close(multi[0].y, 0.9f); close(multi[0].vy, 37);

        for (int n : {1, 17, 257}) compare(particles::seeded_state(n, 7, 1), p, 8);
        compare({{0, 0, 0, 0}, {1, 1, 0, 0}, {0.1999f, 0.4f, 0, 0},
                 {0.2001f, 0.4f, 0, 0}, {0.3999f, 0.4f, 0, 0}}, p, 3);
        compare(std::vector<particles::Particle>(513, {0.5f, 0.5f, 0, 0}), p, 1);
        const auto initial = particles::seeded_state(17, 7, 1);
        compare(initial, p, 0);
        const auto repeated = particles::seeded_state(17, 7, 1);
        for (std::size_t i = 0; i < initial.size(); ++i) close(initial[i].x, repeated[i].x, 0);

        int rejected = 0;
        try { auto invalid = p; invalid.radius = 0; particles::simulate_grid(initial, invalid, 1); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { particles::simulate_grid({}, p, 1); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { particles::simulate_grid(initial, p, -1); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { particles::simulate_grid({{0, 0, std::numeric_limits<float>::quiet_NaN(), 0}}, p, 1); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { particles::simulate_grid({{1.1f, 0, 0, 0}}, p, 1); }
        catch (const std::invalid_argument&) { ++rejected; }
        try { auto invalid = p; invalid.radius = 1e-9f; particles::simulate_grid(initial, invalid, 1); }
        catch (const std::invalid_argument&) { ++rejected; }
        require(rejected == 6, "invalid input accepted");
        std::cout << "CPU analytic, grid/reference, dense-cell, border, seed and validation checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
