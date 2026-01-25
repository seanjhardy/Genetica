// Planet module - manages planet background rendering with Perlin noise

use wgpu;
use crate::{gpu::wgsl::PERLIN_SHADER, utils::math::Rect};

/// Planet background configuration and rendering
pub struct Planet {
    /// Planet name
    name: String,
    
    /// Colors for the noise gradient (RGB as f32 0-1 range)
    colors: Vec<[f32; 4]>,
    
    /// Noise parameters
    noise_frequency: f32,
    noise_octaves: u32,
    noise_warp: f32,
    smooth_noise: bool,
    
    /// Random seed for noise generation
    seed: f32,
    
    /// Current bounds of the simulation
    current_bounds: Rect,
    
    /// GPU resources
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    pipeline: Option<wgpu::RenderPipeline>,
    uniform_buffer: Option<wgpu::Buffer>,
    color_buffer: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    
    /// Flag to trigger texture regeneration
    needs_update: bool,
}

impl Planet {
    /// Convert sRGB color component (0-1) to linear color space
    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }
    
    /// Convert sRGB color (0-255) to linear color space
    fn srgb_u8_to_linear(r: u8, g: u8, b: u8) -> [f32; 4] {
        [
            Self::srgb_to_linear(r as f32 / 255.0),
            Self::srgb_to_linear(g as f32 / 255.0),
            Self::srgb_to_linear(b as f32 / 255.0),
            1.0,
        ]
    }
    
    /// Create Delune planet with default configuration
    pub fn new_delune() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            name: "Delune".to_string(),
            // Colors in sRGB (0-255) converted to linear space for rendering
            colors: vec![
                Self::srgb_u8_to_linear(1, 1, 18),
                Self::srgb_u8_to_linear(7, 4, 51),
                Self::srgb_u8_to_linear(20, 11, 92),
            ],
            noise_frequency: 0.6,
            noise_octaves: 2,
            noise_warp: 0.0,
            smooth_noise: true,
            seed: rng.gen_range(0.0..10000.0),
            current_bounds: Rect::new(0.0, 0.0, 1000.0, 1000.0),
            texture: None,
            texture_view: None,
            pipeline: None,
            uniform_buffer: None,
            color_buffer: None,
            bind_group: None,
            needs_update: true,
        }
    }
    
    /// Update bounds and mark for regeneration if changed
    pub fn set_bounds(&mut self, bounds: Rect) {
        if self.current_bounds != bounds {
            self.current_bounds = bounds;
            self.needs_update = true;
        }
    }
    
    /// Initialize GPU resources (called once)
    pub fn initialize(&mut self, device: &wgpu::Device, texture_format: wgpu::TextureFormat) {
        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planet Perlin Shader"),
            source: PERLIN_SHADER.clone(),
        });
        
        // Create uniform buffer (will be updated per frame)
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Uniform Buffer"),
            size: std::mem::size_of::<PlanetUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create color buffer
        let color_data: Vec<f32> = self.colors.iter().flat_map(|c| c.iter().copied()).collect();
        let color_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Color Buffer"),
            size: (color_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planet Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planet Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: color_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planet Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planet Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: texture_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            cache: None,
            multiview: None,
        });
        
        self.pipeline = Some(pipeline);
        self.uniform_buffer = Some(uniform_buffer);
        self.color_buffer = Some(color_buffer);
        self.bind_group = Some(bind_group);
    }
    
    /// Update the planet texture if needed
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_format: wgpu::TextureFormat) {
        if !self.needs_update {
            return;
        }
        
        // Calculate texture size based on bounds
        const SCALE: f32 = 2.0; // Scale factor (MAP_SCALE from C++ version)
        let width = (self.current_bounds.width / SCALE).max(1.0).min(4096.0) as u32;
        let height = (self.current_bounds.height / SCALE).max(1.0).min(4096.0) as u32;
        
        // Create or recreate texture
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Planet Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Update uniforms
        println!("Seed: {}", self.seed);
        let uniforms = PlanetUniforms {
            seed: self.seed / 10000.0,
            noise_frequency: self.noise_frequency,
            noise_octaves: self.noise_octaves,
            noise_warp: self.noise_warp,
            num_colors: self.colors.len() as u32,
            smooth_noise: if self.smooth_noise { 1 } else { 0 },
            _padding1: 0.0,
            _padding2: 0.0,
            offset: [self.current_bounds.left, self.current_bounds.top],
            resolution: [self.current_bounds.width, self.current_bounds.height],
        };
        
        // Write uniforms to buffer
        if let Some(uniform_buffer) = &self.uniform_buffer {
            queue.write_buffer(uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }
        
        // Write colors to buffer
        if let Some(color_buffer) = &self.color_buffer {
            let color_data: Vec<f32> = self.colors.iter().flat_map(|c| c.iter().copied()).collect();
            queue.write_buffer(color_buffer, 0, bytemuck::cast_slice(&color_data));
        }
        
        // Render noise to texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planet Render Encoder"),
        });
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Planet Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });
            
            if let (Some(pipeline), Some(bind_group)) = (&self.pipeline, &self.bind_group) {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..4, 0..1); // Full-screen quad
            }
        }
        
        queue.submit(std::iter::once(encoder.finish()));
        
        self.texture = Some(texture);
        self.texture_view = Some(texture_view);
        self.needs_update = false;
    }
    
    /// Get the texture view for rendering
    pub fn texture_view(&self) -> Option<&wgpu::TextureView> {
        self.texture_view.as_ref()
    }
    
}

/// Uniforms structure matching the shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PlanetUniforms {
    seed: f32,
    noise_frequency: f32,
    noise_octaves: u32,
    noise_warp: f32,
    num_colors: u32,
    smooth_noise: u32,
    _padding1: f32,
    _padding2: f32,
    offset: [f32; 2],
    resolution: [f32; 2],
}

