use once_cell::sync::Lazy;
use wesl::include_wesl;
use wgpu::ShaderSource;

/// Helper to convert WGSL source to a `wgpu::ShaderSource`
fn shader_source(source: String) -> ShaderSource<'static> {
    ShaderSource::Wgsl(source.into())
}

/// All shader modules, lazy-loaded once at startup.
pub static CELLS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_cells").to_string()));
pub static LINKS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_links").to_string()));
pub static NUTRIENTS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_nutrients").to_string()));
pub static SPAWN_CELLS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_spawn_cells").to_string()));
pub static POINTS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_points").to_string()));
pub static PICK_CELL_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("kernels_pick_cell").to_string()));
    
pub static CELLS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_cells").to_string()));
pub static IMAGE_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_image_texture").to_string()));
pub static LINKS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_links").to_string()));
pub static NUTRIENTS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_nutrients").to_string()));
pub static PERLIN_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_perlin").to_string()));
pub static ENV_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_env_texture").to_string()));
pub static TERRAIN_HEIGHT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_terrain_height").to_string()));
pub static TERRAIN_ROCK_NOISE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_terrain_rock_noise").to_string()));
pub static TERRAIN_SHADOW_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_terrain_shadow").to_string()));
pub static TERRAIN_COMPOSITE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_terrain_composite").to_string()));
pub static CAUSTICS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_caustics").to_string()));
pub static CAUSTICS_BLIT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_caustics_blit").to_string()));
pub static CAUSTICS_COMPOSITE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_caustics_composite").to_string()));
pub static TERRAIN_CAUSTICS_COMPOSITE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_terrain_caustics_composite").to_string()));
pub static POST_PROCESSING_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_post_processing").to_string()));
pub static TEXT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_text").to_string()));
pub static UI_RECT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_ui_rect").to_string()));
pub static PERLIN_NOISE_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_perlin_noise_texture").to_string()));
pub static POINT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wesl!("shaders_points").to_string()));
