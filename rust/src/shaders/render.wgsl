// Render shader for drawing points as quads

struct Point {
    pos: vec2<f32>,
    prev_pos: vec2<f32>,
    velocity: vec2<f32>,
}

// Uniforms struct must match Rust struct layout exactly (including padding)
// In WGSL uniform buffers, vec3<f32> takes 16 bytes for alignment (padded to vec4)
struct Uniforms {
    delta_time: f32, // 4 bytes at offset 0
    _padding1: f32, // 4 bytes at offset 4 - pad to align vec2 (not needed, but for consistency)
    _padding2: f32, // 4 bytes at offset 8
    _padding3: f32, // 4 bytes at offset 12 (totals 16 bytes to align vec2)
    camera_pos: vec2<f32>, // 8 bytes at offset 16
    zoom: f32, // 4 bytes at offset 24
    point_radius: f32, // 4 bytes at offset 28
    bounds: vec4<f32>, // 16 bytes at offset 32
    view_size: vec2<f32>, // 8 bytes at offset 48
    _padding5: vec2<f32>, // 8 bytes at offset 56 to reach 64 bytes
    _final_padding: vec4<f32>, // 16 bytes at offset 64 to reach 80 bytes
}

@group(0) @binding(0)
var<storage, read> points: array<Point>;

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) point_index: f32,
    @location(1) uv: vec2<f32>,
}

// Generate a quad for each point using instanced rendering
// Using instance_id to identify which point, and vertex_index to identify quad vertex
@vertex
fn vs_main(@builtin(instance_index) instance_index: u32, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Get the point data for this instance
    let point_idx = instance_index;
    let quad_vertex = vertex_index;
    
    let point = points[point_idx];
    
    // Transform point from world space to clip space
    // Match SFML view transformation exactly:
    // - SFML view.setCenter(camera_pos) - center in world coordinates
    // - SFML view.setSize(view_size) - visible size in world units (targetSize / zoomLevel)
    // - SFML transforms: (world - center) / (view_size / 2) to clip space [-1, 1]
    
    let world_pos = point.pos;
    let relative_x = world_pos.x - uniforms.camera_pos.x;
    let relative_y = world_pos.y - uniforms.camera_pos.y;
    
    // Visible size in world units (what the camera sees)
    let view_size_x = uniforms.view_size.x / uniforms.zoom;  // World units visible in X
    let view_size_y = uniforms.view_size.y / uniforms.zoom;  // World units visible in Y
    
    // Convert to clip space [-1, 1]
    // SFML view transformation: clip = (world - center) / (view_size / 2)
    // When world = center + view_size/2, clip = 1.0 (right edge)
    // When world = center - view_size/2, clip = -1.0 (left edge)
    let clip_x = (relative_x / view_size_x) * 2.0;
    
    // Y axis: SFML screen Y increases downward, but clip space Y increases upward
    // So we flip Y: clip_y = -(relative_y / view_size_y) * 2.0
    let clip_y = -(relative_y / view_size_y) * 2.0;
    
    // Clamp to reasonable range to ensure visibility (safety check)
    // Points slightly outside [-1, 1] might still be visible due to quad size
    // but let's ensure the center is in range
    
    // Point size in world units (scales with zoom)
    // Use the same radius as physics and collisions for consistency
    let point_radius_world = uniforms.point_radius;
    
    // Convert point size from world units to clip space
    // If point_radius_world = view_size_x, then clip_size = 2.0 (spans entire screen)
    // So: point_size_clip = (point_radius_world / view_size) * 2.0
    let point_size_clip_x = (point_radius_world / view_size_x) * 2.0;
    let point_size_clip_y = (point_radius_world / view_size_y) * 2.0;
    
    // Generate quad vertices for TriangleStrip
    // TriangleStrip order: 0-1-2 (first triangle), 1-2-3 (second triangle, shares 1-2 edge)
    // Order: bottom-left (0), bottom-right (1), top-left (2), top-right (3)
    var offset: vec2<f32>;
    var uv_offset: vec2<f32>;
    
    switch quad_vertex {
        case 0u {  // Bottom-left: first triangle vertex 0
            offset = vec2<f32>(-1.0, -1.0) * vec2<f32>(point_size_clip_x, point_size_clip_y);
            uv_offset = vec2<f32>(0.0, 1.0); // Note: Y=1.0 is bottom in texture coords
        }
        case 1u {  // Bottom-right: shared edge vertex
            offset = vec2<f32>(1.0, -1.0) * vec2<f32>(point_size_clip_x, point_size_clip_y);
            uv_offset = vec2<f32>(1.0, 1.0);
        }
        case 2u {  // Top-left: shared edge vertex
            offset = vec2<f32>(-1.0, 1.0) * vec2<f32>(point_size_clip_x, point_size_clip_y);
            uv_offset = vec2<f32>(0.0, 0.0);
        }
        default {  // Top-right: second triangle vertex 3
            offset = vec2<f32>(1.0, 1.0) * vec2<f32>(point_size_clip_x, point_size_clip_y);
            uv_offset = vec2<f32>(1.0, 0.0);
        }
    }
    
    // Output clip position - ensure it's valid
    // Clip space: X and Y must be in [-1, 1] to be visible, Z can be anything, W must be 1.0
    out.clip_position = vec4<f32>(clip_x + offset.x, clip_y + offset.y, 0.0, 1.0);
    
    // Debug: Output a very visible color if point is near center (testing only)
    // This will help verify the transformation is working
    out.point_index = f32(point_idx);
    out.uv = uv_offset;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create circular shape by discarding pixels outside the circle
    // UV is now in [0, 1] range directly
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(in.uv - center);
    if dist > 0.5 {
        discard;
    }
    
    // Use a bright, visible color for testing
    // Bright cyan/white for maximum visibility
    return vec4<f32>(0.0, 1.0, 1.0, 1.0); // Bright cyan
}

