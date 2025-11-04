// UI Rectangle shader for rendering backgrounds, borders, and shadows

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct Uniforms {
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Convert pixel coordinates to normalized device coordinates (-1 to 1)
    // Y is flipped because screen coordinates have (0,0) at top-left, but NDC has (0,0) at center
    let x = (vertex.position.x / uniforms.screen_width) * 2.0 - 1.0;
    let y = 1.0 - (vertex.position.y / uniforms.screen_height) * 2.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.color = vertex.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
