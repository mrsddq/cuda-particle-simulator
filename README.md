# CUDA Particle Simulator

A small C++17/CUDA project comparing an all-pairs CPU reference, a uniform-grid CPU implementation, and custom linked-cell GPU kernels for 2D soft-contact particles. There is no rendering dependency: the CLI writes CSV snapshots and timing records.

## Simulation contract

Particles have equal unit mass. For each pair at distance `d < 2*radius`, repulsion is `stiffness * (2*radius-d)` along their separation. An exactly coincident pair gets an antisymmetric x-direction force chosen by particle ID. Every step reads the old snapshot, accumulates forces, updates velocity, and then updates position (semi-implicit Euler). Walls reflect positions and velocities into `[0, domain]`, including multiple crossings in a step. There is no gravity, friction, hard-sphere impulse solver, or rigid-body rotation.

The grid's cell width is the interaction cutoff. Only nine neighboring cells need inspection. Each CUDA thread updates its own particle, so there are no racing velocity writes. A linked list has one link per particle and no fixed per-cell occupancy cap: a dense cell never silently drops particles. Dense scenes can still have quadratic work. CUDA atomic insertion changes neighbor summation order, so floating-point CPU/GPU comparisons use a tolerance rather than claiming bitwise determinism. Seeds and CPU runs are repeatable; long chaotic trajectories need not stay numerically identical across backends.

## Build and run without CUDA

Requires CMake 3.20+ and a C++17 compiler. The CPU build is the default.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/particle_sim --backend cpu-grid --count 1024 --steps 100 --seed 42 --output snapshot.csv
./build/particle_sim --backend cpu-naive --count 1024 --steps 100 --seed 42
```

On Windows with a multi-configuration generator, use `--config Release` for build/CTest and `build/Release/particle_sim.exe`.

## Build and validate CUDA

Requires NVIDIA CUDA Toolkit and a compatible NVIDIA GPU/driver. Choose the architecture for your device; `75` is the documented example, not a claim about the build machine.

```bash
cmake -S . -B build-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
./build-cuda/particle_sim --backend cuda --count 1024 --steps 100 --seed 42 --output gpu.csv
compute-sanitizer --tool memcheck ./build-cuda/gpu_test
```

The GPU test returns CTest skip code 77 if the device/driver is absent. Other CUDA errors fail. An enabled CUDA build requires a working toolkit; it never silently substitutes CPU execution. GPU tests compare irregular counts, dense/coincident particles, neighboring cell boundaries, wall cases and zero-step behavior against the all-pairs CPU reference.

## Measurements and current evidence

The CLI prints `total_ms`, `device_ms`, and `transfer_ms`. Total time surrounds simulation, including allocation, copies, validation, cleanup and first-call runtime initialization. CUDA events measure device grid-clear/build/integration work; transfer events measure host/device copies. CPU rows use `-1` for the two CUDA-only metrics. File serialization is outside simulation timing. Compare the same seed/count/steps, run warmups and repetitions, record GPU/driver/toolkit and report medians; do not infer a speedup from a CPU-only build.

CI runs CPU tests and compiles the CUDA target in the NVIDIA CUDA 12.8.1 development image. Hosted CI does not execute GPU kernels. No GPU runtime correctness, compute-sanitizer result, measured GPU speedup, or production-physics accuracy is claimed until a real GPU run is recorded. The creation environment has Intel integrated graphics only.

Local CPU validation on 15 November 2026 used WSL Ubuntu g++ 15.2.0: the analytic/reference tests passed with optimization and again with AddressSanitizer/UndefinedBehaviorSanitizer. Both CPU CLI backends produced the same checksum for 257 particles over ten steps; CSV output and invalid/uncompiled-backend errors were exercised. CMake and CUDA compilation are separate CI gates, not implied by those direct-compiler checks.

This is an AI-assisted educational implementation, not a deployed physics engine. Conceptual references: [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) and [CUDA best practices: performance measurement](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). No tutorial source is vendored.
