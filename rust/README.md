# Genetica Rust - GPU-Accelerated Verlet Integration

This is a Rust implementation demonstrating GPU-accelerated physics simulation using wgpu.

## Features

- **6000 points** simulated using Verlet integration on the GPU
- **GPU-side storage** - All point data stays on the GPU
- **GPU-side rendering** - Points are rendered directly from GPU buffers
- **Cross-platform** - Works on macOS and Windows (and Linux)

## Requirements

- Rust (latest stable version recommended)
- Cargo (comes with Rust)

## Building

```bash
cd rust
cargo build --release
```

## Running

```bash
cargo run --release
```

Or run the compiled binary:
```bash
./target/release/genetica-rust
```

## Architecture

- **winit**: Window management
- **wgpu**: GPU compute and rendering (cross-platform graphics API)
- **Compute Shader** (`compute.wgsl`): Handles Verlet integration physics on the GPU
- **Render Shader** (`render.wgsl`): Renders points directly from GPU buffers

The simulation runs entirely on the GPU - points are stored in GPU buffers, updated via compute shaders, and rendered without any CPU-GPU data transfer during the simulation loop.


