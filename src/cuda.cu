#include "particles.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace particles {
static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(std::size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Event {
    cudaEvent_t value;
    Event() { check(cudaEventCreate(&value)); }
    ~Event() { cudaEventDestroy(value); }
};

bool cuda_available() {
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) return false;
    check(status);
    return count > 0;
}

__device__ int cell(float position, float cell_size, int width) {
    return min(static_cast<int>(position / cell_size), width - 1);
}
__global__ void build_grid(const Particle* state, int count, int width, float cell_size,
                           int* heads, int* links) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int id = cell(state[i].y, cell_size, width) * width + cell(state[i].x, cell_size, width);
    links[i] = atomicExch(&heads[id], i);
}
__device__ void reflect_gpu(float& position, float& velocity, float domain) {
    float phase = fmodf(position, 2 * domain);
    if (phase < 0) phase += 2 * domain;
    if (phase > domain) { position = 2 * domain - phase; velocity = -velocity; }
    else position = phase;
    if (position == 0 && velocity < 0) velocity = -velocity;
    if (position == domain && velocity > 0) velocity = -velocity;
}
__global__ void integrate(const Particle* state, Particle* next, int count, Parameters p,
                          int width, const int* heads, const int* links) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    Particle q = state[i];
    float fx = 0, fy = 0;
    const float cutoff = 2 * p.radius;
    const int x = cell(q.x, cutoff, width), y = cell(q.y, cutoff, width);
    for (int cy = max(0, y - 1); cy <= min(width - 1, y + 1); ++cy)
        for (int cx = max(0, x - 1); cx <= min(width - 1, x + 1); ++cx)
            for (int j = heads[cy * width + cx]; j != -1; j = links[j]) {
                if (i == j) continue;
                const float dx = q.x - state[j].x, dy = q.y - state[j].y;
                const float distance2 = dx * dx + dy * dy;
                if (distance2 >= cutoff * cutoff) continue;
                if (distance2 == 0) fx += (i < j ? -1.0f : 1.0f) * p.stiffness * cutoff;
                else {
                    const float distance = sqrtf(distance2);
                    const float scale = p.stiffness * (cutoff - distance) / distance;
                    fx += scale * dx;
                    fy += scale * dy;
                }
            }
    q.vx += fx * p.dt;
    q.vy += fy * p.dt;
    q.x += q.vx * p.dt;
    q.y += q.vy * p.dt;
    reflect_gpu(q.x, q.vx, p.domain);
    reflect_gpu(q.y, q.vy, p.domain);
    next[i] = q;
}

GpuResult simulate_cuda(const std::vector<Particle>& input, const Parameters& p, int steps) {
    validate(input, p);
    if (steps < 0) throw std::invalid_argument("steps must be nonnegative");
    if (!cuda_available()) throw std::runtime_error("no CUDA device/driver available");
    const int count = static_cast<int>(input.size()), width = grid_width(p);
    const auto bytes = input.size() * sizeof(Particle);
    Buffer<Particle> first(input.size()), second(input.size());
    Buffer<int> heads(width * width), links(input.size());
    Event start, end;
    float transfer = 0, elapsed = 0, download = 0;
    check(cudaEventRecord(start.value));
    check(cudaMemcpy(first.data, input.data(), bytes, cudaMemcpyHostToDevice));
    check(cudaEventRecord(end.value));
    check(cudaEventSynchronize(end.value));
    check(cudaEventElapsedTime(&transfer, start.value, end.value));
    auto* current = first.data;
    auto* next = second.data;
    check(cudaEventRecord(start.value));
    for (int step = 0; step < steps; ++step) {
        check(cudaMemset(heads.data, 0xff, width * width * sizeof(int)));
        build_grid<<<(count + 255) / 256, 256>>>(current, count, width, 2 * p.radius, heads.data, links.data);
        check(cudaGetLastError());
        integrate<<<(count + 255) / 256, 256>>>(current, next, count, p, width, heads.data, links.data);
        check(cudaGetLastError());
        std::swap(current, next);
    }
    check(cudaEventRecord(end.value));
    check(cudaEventSynchronize(end.value));
    check(cudaEventElapsedTime(&elapsed, start.value, end.value));
    GpuResult result;
    result.state.resize(input.size());
    check(cudaEventRecord(start.value));
    check(cudaMemcpy(result.state.data(), current, bytes, cudaMemcpyDeviceToHost));
    check(cudaEventRecord(end.value));
    check(cudaEventSynchronize(end.value));
    check(cudaEventElapsedTime(&download, start.value, end.value));
    validate(result.state, p);
    result.device_ms = elapsed;
    result.transfer_ms = transfer + download;
    return result;
}
}  // namespace particles
