use wgpu::ShaderSource;
use wgsl_includes::include_wgsl;
use once_cell::sync::Lazy;

/// Helper to convert WGSL source to a `wgpu::ShaderSource`
fn shader_source(source: String) -> ShaderSource<'static> {
    ShaderSource::Wgsl(source.into())
}

/// All shader modules, lazy-loaded once at startupss
pub static CELLS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/cells.wgsl").to_string()));
pub static LINKS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/links.wgsl").to_string()));
pub static NUTRIENTS_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/nutrients.wgsl").to_string()));
pub static SEQUENCE_GRN_KERNEL: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/sequence_grn.wgsl").to_string()));
// TODO: Uncomment when implementing parallel genome operations
// pub static COPY_GENOMES_KERNEL: Lazy<ShaderSource<'static>> =
//     Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/copy_genomes.wgsl").to_string()));
// pub static MUTATE_GENOMES_KERNEL: Lazy<ShaderSource<'static>> =
//     Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/kernels/mutate_genomes.wgsl").to_string()));

pub static CELLS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/cells.wgsl").to_string()));
pub static IMAGE_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/image_texture.wgsl").to_string()));
pub static LINKS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/links.wgsl").to_string()));
pub static NUTRIENTS_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/nutrients.wgsl").to_string()));
pub static PERLIN_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/perlin.wgsl").to_string()));
pub static ENV_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/env_texture.wgsl").to_string()));
pub static POST_PROCESSING_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/post_processing.wgsl").to_string()));
pub static TEXT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/text.wgsl").to_string()));
pub static UI_RECT_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/ui_rect.wgsl").to_string()));
pub static PERLIN_NOISE_TEXTURE_SHADER: Lazy<ShaderSource<'static>> =
    Lazy::new(|| shader_source(include_wgsl!("src/gpu/wgsl/shaders/perlin_noise_texture.wgsl").to_string()));
