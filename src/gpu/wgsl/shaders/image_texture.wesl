// Vertex shader for rendering images with tint

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) tint: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) tint: vec4<f32>,
}

struct Uniforms {
    screen_width: f32,
    screen_height: f32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Convert screen coordinates to NDC (Normalized Device Coordinates)
    // Screen space: (0,0) top-left, (width, height) bottom-right
    // NDC space: (-1,-1) bottom-left, (1,1) top-right
    let x_ndc = (in.position.x / uniforms.screen_width) * 2.0 - 1.0;
    let y_ndc = 1.0 - (in.position.y / uniforms.screen_height) * 2.0;
    
    out.clip_position = vec4<f32>(x_ndc, y_ndc, 0.0, 1.0);
    out.uv = in.uv;
    out.tint = in.tint;
    
    return out;
}

// Fragment shader

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_color = textureSample(texture, texture_sampler, in.uv);
    
    // Apply tint by multiplying texture color with tint color
    // This allows for color tinting and opacity control
    let final_color = texture_color * in.tint;
    
    return final_color;
}

