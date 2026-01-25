# Genetica Rust - A GPU accelerated evolutionary algorithm of multicellular creatures using gene regulatory networks.

This is a Rust implementation demonstrating GPU-accelerated physics simulation using wgpu.


## Requirements

- Rust (latest stable version recommended)
- Cargo (comes with Rust)

## Building

```bash
cd rust
cargo build --release
```

## Running

### Option 1: Standard Cargo Run (Development)
```bash
cargo run --release
```

### Option 2: Build macOS App Bundle (Recommended for macOS)
```bash
./build_macos_app.sh
```
This creates a proper `Genetica.app` bundle with the app icon that:
- Shows the proper icon in the dock when running
- Can be double-clicked to launch (no terminal window)
- Can be copied to `/Applications/` for permanent installation
- Includes all assets bundled inside

After building, launch with:
```bash
open Genetica.app
```

Or install to Applications:
```bash
cp -r Genetica.app /Applications/
```

## Architecture

The simulation runs entirely on the GPU - points are stored in GPU buffers, updated via compute shaders, and rendered without any CPU-GPU data transfer during the simulation loop.


