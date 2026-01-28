// Planet module - manages terrain background rendering

use wgpu;
use wgpu::util::DeviceExt;
use crate::{
    gpu::wgsl::{
        CAUSTICS_SHADER,
        TERRAIN_CAUSTICS_COMPOSITE_SHADER,
        TERRAIN_COMPOSITE_SHADER,
        TERRAIN_HEIGHT_SHADER,
        TERRAIN_ROCK_NOISE_SHADER,
        TERRAIN_SHADOW_SHADER,
    },
    utils::math::Rect,
};

/// Planet background configuration and rendering
pub struct Planet {
    /// Planet name
    name: String,

    /// Random seed for noise generation
    seed: u32,

    /// Current bounds of the simulation
    current_bounds: Rect,

    /// GPU resources
    texture: Option<wgpu::Texture>,
    texture_view: Option<wgpu::TextureView>,
    height_texture: Option<wgpu::Texture>,
    height_texture_view: Option<wgpu::TextureView>,
    rock_noise_texture: Option<wgpu::Texture>,
    rock_noise_texture_view: Option<wgpu::TextureView>,
    shadow_texture: Option<wgpu::Texture>,
    shadow_texture_view: Option<wgpu::TextureView>,
    composite_temp_texture: Option<wgpu::Texture>,
    composite_temp_view: Option<wgpu::TextureView>,
    caustics_texture: Option<wgpu::Texture>,
    caustics_texture_view: Option<wgpu::TextureView>,

    sampler: Option<wgpu::Sampler>,

    height_pipeline: Option<wgpu::RenderPipeline>,
    rock_noise_pipeline: Option<wgpu::RenderPipeline>,
    shadow_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    caustics_pipeline: Option<wgpu::RenderPipeline>,
    terrain_caustics_composite_pipeline: Option<wgpu::RenderPipeline>,

    height_uniform_buffer: Option<wgpu::Buffer>,
    rock_uniform_buffer: Option<wgpu::Buffer>,
    shadow_uniform_buffer: Option<wgpu::Buffer>,
    caustics_uniform_buffer: Option<wgpu::Buffer>,

    height_bind_group: Option<wgpu::BindGroup>,
    rock_bind_group: Option<wgpu::BindGroup>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    composite_bind_group: Option<wgpu::BindGroup>,
    caustics_bind_group: Option<wgpu::BindGroup>,
    terrain_caustics_composite_bind_group: Option<wgpu::BindGroup>,

    /// Flag to trigger texture regeneration
    needs_update: bool,
    /// Monotonic counter for texture regenerations
    texture_generation: u64,
}

impl Planet {
    fn expanded_bounds(&self) -> Rect {
        const TERRAIN_EXPANSION: f32 = 1.0;
        let center = self.current_bounds.center();
        let expanded_width = self.current_bounds.width * TERRAIN_EXPANSION;
        let expanded_height = self.current_bounds.height * TERRAIN_EXPANSION;
        Rect::new(
            center.x - expanded_width * 0.5,
            center.y - expanded_height * 0.5,
            expanded_width,
            expanded_height,
        )
    }
    /// Create Delune planet with default configuration
    pub fn new_delune() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        Self {
            name: "Delune".to_string(),
            seed: rng.gen(),
            current_bounds: Rect::new(0.0, 0.0, 1000.0, 1000.0),
            texture: None,
            texture_view: None,
            height_texture: None,
            height_texture_view: None,
            rock_noise_texture: None,
            rock_noise_texture_view: None,
            shadow_texture: None,
            shadow_texture_view: None,
            composite_temp_texture: None,
            composite_temp_view: None,
            caustics_texture: None,
            caustics_texture_view: None,
            sampler: None,
            height_pipeline: None,
            rock_noise_pipeline: None,
            shadow_pipeline: None,
            composite_pipeline: None,
            caustics_pipeline: None,
            terrain_caustics_composite_pipeline: None,
            height_uniform_buffer: None,
            rock_uniform_buffer: None,
            shadow_uniform_buffer: None,
            caustics_uniform_buffer: None,
            height_bind_group: None,
            rock_bind_group: None,
            shadow_bind_group: None,
            composite_bind_group: None,
            caustics_bind_group: None,
            terrain_caustics_composite_bind_group: None,
            needs_update: true,
            texture_generation: 0,
        }
    }

    /// Update bounds and mark for regeneration if changed
    pub fn set_bounds(&mut self, bounds: Rect) {
        if self.current_bounds != bounds {
            self.current_bounds = bounds;
            self.needs_update = true;
        }
    }

    /// Reseed noise generation and mark for regeneration
    pub fn reseed(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut next_seed = rng.gen::<u32>();
        if next_seed == self.seed {
            next_seed = next_seed.wrapping_add(1);
        }
        self.seed = next_seed;
        self.needs_update = true;
    }

    /// Initialize GPU resources (called once)
    pub fn initialize(&mut self, device: &wgpu::Device, texture_format: wgpu::TextureFormat) {
        let height_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Height Shader"),
            source: TERRAIN_HEIGHT_SHADER.clone(),
        });
        let rock_noise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Rock Noise Shader"),
            source: TERRAIN_ROCK_NOISE_SHADER.clone(),
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Shadow Shader"),
            source: TERRAIN_SHADOW_SHADER.clone(),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Composite Shader"),
            source: TERRAIN_COMPOSITE_SHADER.clone(),
        });
        let caustics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Caustics Shader"),
            source: CAUSTICS_SHADER.clone(),
        });
        let terrain_caustics_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Terrain Caustics Composite Shader"),
            source: TERRAIN_CAUSTICS_COMPOSITE_SHADER.clone(),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Terrain Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let height_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Height Uniform Buffer"),
            size: std::mem::size_of::<TerrainHeightUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rock_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Rock Noise Uniform Buffer"),
            size: std::mem::size_of::<TerrainNoiseUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Terrain Shadow Uniform Buffer"),
            contents: bytemuck::bytes_of(&ShadowSunRayUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let caustics_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Terrain Caustics Uniform Buffer"),
            size: std::mem::size_of::<CausticsUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let height_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Height Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let rock_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Rock Noise Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let shadow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Shadow Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Composite Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let caustics_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Terrain Caustics Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let terrain_caustics_composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Terrain Caustics Composite Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let height_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Height Pipeline Layout"),
            bind_group_layouts: &[&height_bind_group_layout],
            push_constant_ranges: &[],
        });
        let rock_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Rock Noise Pipeline Layout"),
            bind_group_layouts: &[&rock_bind_group_layout],
            push_constant_ranges: &[],
        });
        let shadow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Shadow Pipeline Layout"),
            bind_group_layouts: &[&shadow_bind_group_layout],
            push_constant_ranges: &[],
        });
        let composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Composite Pipeline Layout"),
            bind_group_layouts: &[&composite_bind_group_layout],
            push_constant_ranges: &[],
        });
        let caustics_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Terrain Caustics Pipeline Layout"),
            bind_group_layouts: &[&caustics_bind_group_layout],
            push_constant_ranges: &[],
        });
        let terrain_caustics_composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Terrain Caustics Composite Pipeline Layout"),
                bind_group_layouts: &[&terrain_caustics_composite_bind_group_layout],
                push_constant_ranges: &[],
            });

        let height_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Height Pipeline"),
            layout: Some(&height_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &height_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &height_shader,
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

        let rock_noise_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Rock Noise Pipeline"),
            layout: Some(&rock_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &rock_noise_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &rock_noise_shader,
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

        let shadow_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Terrain Shadow Pipeline"),
            layout: Some(&shadow_pipeline_layout),
            module: &shadow_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
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
        let caustics_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Terrain Caustics Pipeline"),
            layout: Some(&caustics_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &caustics_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_shader,
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
        let terrain_caustics_composite_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Terrain Caustics Composite Pipeline"),
                layout: Some(&terrain_caustics_composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &terrain_caustics_composite_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &terrain_caustics_composite_shader,
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

        let height_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Height Bind Group"),
            layout: &height_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: height_uniform_buffer.as_entire_binding(),
            }],
        });
        let rock_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Rock Noise Bind Group"),
            layout: &rock_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: rock_uniform_buffer.as_entire_binding(),
            }],
        });

        self.sampler = Some(sampler);
        self.height_pipeline = Some(height_pipeline);
        self.rock_noise_pipeline = Some(rock_noise_pipeline);
        self.shadow_pipeline = Some(shadow_pipeline);
        self.composite_pipeline = Some(composite_pipeline);
        self.caustics_pipeline = Some(caustics_pipeline);
        self.terrain_caustics_composite_pipeline = Some(terrain_caustics_composite_pipeline);
        self.height_uniform_buffer = Some(height_uniform_buffer);
        self.rock_uniform_buffer = Some(rock_uniform_buffer);
        self.shadow_uniform_buffer = Some(shadow_uniform_buffer);
        self.caustics_uniform_buffer = Some(caustics_uniform_buffer);
        self.height_bind_group = Some(height_bind_group);
        self.rock_bind_group = Some(rock_bind_group);
    }

    /// Update the planet texture if needed
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_format: wgpu::TextureFormat) {
        if !self.needs_update {
            return;
        }

        const SCALE: f32 = 1.0;
        let expanded_bounds = self.expanded_bounds();
        let width = (expanded_bounds.width / SCALE).max(1.0).min(4096.0) as u32;
        let height = (expanded_bounds.height / SCALE).max(1.0).min(4096.0) as u32;

        let height_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Height Texture"),
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
        let height_texture_view = height_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let rock_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Rock Noise Texture"),
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
        let rock_noise_texture_view = rock_noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Shadow Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_texture_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Composite Texture"),
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
        let composite_temp_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Composite Temp Texture"),
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
        let composite_temp_view = composite_temp_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let caustics_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Terrain Caustics Texture"),
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
        let caustics_texture_view = caustics_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let height_uniforms = TerrainHeightUniforms {
            seed: self.seed,
            base_frequency: 0.001,
            octave_count: 5,
            _padding0: 0,
            frequency_falloff: 2.0,
            amplitude_falloff: 0.5,
            floor_height: 0.0,
            _padding1: 0.0,
            offset: [expanded_bounds.left, expanded_bounds.top],
            resolution: [expanded_bounds.width, expanded_bounds.height],
            warp_frequency: 0.2,
            warp_strength: 1.5,
            warp_octave_count: 3,
            _padding2: 0,
            warp_frequency_falloff: 2.0,
            warp_amplitude_falloff: 0.5,
            _padding3: 0.0,
            _padding4: 0.0,
            band_min: 0.05,
            band_max: 0.95,
            band_steps: 5,
            _padding5: 0,
            detail_frequency_mult: 4.0,
            detail_amplitude: 0.08,
            _padding6: 0.0,
            _padding7: 0.0,
        };
        let rock_uniforms = TerrainNoiseUniforms {
            seed: self.seed.wrapping_add(1337),
            base_frequency: 0.001,
            octave_count: 5,
            _padding0: 0,
            frequency_falloff: 1.5,
            amplitude_falloff: 2.0,
            _pad1: 0.0,
            _pad2: 0.0,
            offset: [expanded_bounds.left, expanded_bounds.top],
            resolution: [expanded_bounds.width, expanded_bounds.height],
        };
        // 3D ray-to-sun parameters.
        let sun_x = -0.5 * width as f32;
        let sun_y = -0.5 * height as f32;
        let height_scale = 500.0;
        // Set sun height so the elevation from the texture center is ~30 degrees,
        // but clamp to a fraction of terrain height so rays can intersect.
        let center_x = 0.5 * width as f32;
        let center_y = 0.5 * height as f32;
        let dx = center_x - sun_x;
        let dy = center_y - sun_y;
        let sun_dist = (dx * dx + dy * dy).sqrt();
        let sun_z_raw = 0.57735026 * sun_dist; // tan(30°)
        let sun_z = sun_z_raw.min(height_scale * 0.8);
        let shadow_uniforms = ShadowSunRayUniforms {
            // width, height, max_steps, ray_count
            dims: [width, height, 4096, 20],
            // height_scale, bias, strength, step_size_texels
            p0: [height_scale, 0.02, 1.2, 0.5],
            // unused, unused, unused, distance_falloff
            p1: [sun_x, sun_y, sun_z, 0.0005],
            // edge_softness, ray_spread_rad, min_shadow, dist_scale
            p2: [0.5, 1.0, 0.2, 0.005],
        };

        if let Some(buffer) = &self.height_uniform_buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&height_uniforms));
        }
        if let Some(buffer) = &self.rock_uniform_buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&rock_uniforms));
        }
        if let Some(buffer) = &self.shadow_uniform_buffer {
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&shadow_uniforms));
        }

        let sampler = self.sampler.as_ref().expect("terrain sampler");

        let shadow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Shadow Bind Group"),
            layout: &self.shadow_pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&height_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.shadow_uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Composite Bind Group"),
            layout: &self.composite_pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&height_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&rock_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&shadow_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        let caustics_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Caustics Bind Group"),
            layout: &self.caustics_pipeline.as_ref().unwrap().get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&height_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.caustics_uniform_buffer.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });
        let terrain_caustics_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Terrain Caustics Composite Bind Group"),
            layout: &self
                .terrain_caustics_composite_pipeline
                .as_ref()
                .unwrap()
                .get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&caustics_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&height_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Terrain Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Height Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &height_texture_view,
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
            render_pass.set_pipeline(self.height_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.height_bind_group.as_ref().unwrap(), &[]);
            render_pass.draw(0..4, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Rock Noise Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &rock_noise_texture_view,
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
            render_pass.set_pipeline(self.rock_noise_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, self.rock_bind_group.as_ref().unwrap(), &[]);
            render_pass.draw(0..4, 0..1);
        }

        {
            // Multi-angle raymarch: one invocation per pixel, 2D dispatch.
            const SHADOW_WG_X: u32 = 8;
            const SHADOW_WG_Y: u32 = 8;
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Terrain Shadow Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &shadow_bind_group, &[]);
            let workgroups_x = (width + SHADOW_WG_X - 1) / SHADOW_WG_X;
            let workgroups_y = (height + SHADOW_WG_Y - 1) / SHADOW_WG_Y;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Composite Pass"),
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
            render_pass.set_pipeline(self.composite_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, &composite_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));

        self.texture = Some(texture);
        self.texture_view = Some(texture_view);
        self.composite_temp_texture = Some(composite_temp_texture);
        self.composite_temp_view = Some(composite_temp_view);
        self.caustics_texture = Some(caustics_texture);
        self.caustics_texture_view = Some(caustics_texture_view);
        self.height_texture = Some(height_texture);
        self.height_texture_view = Some(height_texture_view);
        self.rock_noise_texture = Some(rock_noise_texture);
        self.rock_noise_texture_view = Some(rock_noise_texture_view);
        self.shadow_texture = Some(shadow_texture);
        self.shadow_texture_view = Some(shadow_texture_view);
        self.shadow_bind_group = Some(shadow_bind_group);
        self.composite_bind_group = Some(composite_bind_group);
        self.caustics_bind_group = Some(caustics_bind_group);
        self.terrain_caustics_composite_bind_group = Some(terrain_caustics_composite_bind_group);
        self.needs_update = false;
        self.texture_generation = self.texture_generation.wrapping_add(1);
    }

    /// Texture generation counter (increments when textures regenerate)
    pub fn texture_generation(&self) -> u64 {
        self.texture_generation
    }

    /// Get the texture view for rendering
    pub fn texture_view(&self) -> Option<&wgpu::TextureView> {
        self.texture_view.as_ref()
    }

    /// Get the display texture view (terrain with caustics if available)
    pub fn display_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.composite_temp_view.as_ref().or(self.texture_view.as_ref())
    }

    /// Get the heightmap texture view for caustics
    pub fn height_texture_view(&self) -> Option<&wgpu::TextureView> {
        self.height_texture_view.as_ref()
    }

    /// Update caustics and composite into the temp texture each frame.
    pub fn update_caustics(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, time: f32) {
        let caustics_view = match self.caustics_texture_view.as_ref() {
            Some(view) => view,
            None => return,
        };
        let composite_temp_view = match self.composite_temp_view.as_ref() {
            Some(view) => view,
            None => return,
        };
        let caustics_bind_group = match self.caustics_bind_group.as_ref() {
            Some(bg) => bg,
            None => return,
        };
        let terrain_caustics_composite_bind_group =
            match self.terrain_caustics_composite_bind_group.as_ref() {
                Some(bg) => bg,
                None => return,
            };

        if let Some(buffer) = &self.caustics_uniform_buffer {
            let bounds = self.expanded_bounds();
            let uniforms = CausticsUniforms {
                camera_pos_zoom_thickness: [0.0, 0.0, 1.0, 0.0],
                view_size_grid: [bounds.width, bounds.height, 0.0, 0.0],
                grid_threshold_padding: [0.0, 0.0, 0.0, 0.0],
                border_color: [0.0, 0.0, 0.0, 0.0],
                bounds: [bounds.left, bounds.top, bounds.right(), bounds.bottom()],
                padding: [0.0, 0.0, 0.0, 0.0],
                time_params: [time, 0.0, 0.0, 0.0],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Terrain Caustics Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Caustics Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: caustics_view,
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
            render_pass.set_pipeline(self.caustics_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, caustics_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Terrain Caustics Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: composite_temp_view,
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
            render_pass.set_pipeline(self.terrain_caustics_composite_pipeline.as_ref().unwrap());
            render_pass.set_bind_group(0, terrain_caustics_composite_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainHeightUniforms {
    seed: u32,
    base_frequency: f32,
    octave_count: u32,
    _padding0: u32,
    frequency_falloff: f32,
    amplitude_falloff: f32,
    floor_height: f32,
    _padding1: f32,
    offset: [f32; 2],
    resolution: [f32; 2],
    warp_frequency: f32,
    warp_strength: f32,
    warp_octave_count: u32,
    _padding2: u32,
    warp_frequency_falloff: f32,
    warp_amplitude_falloff: f32,
    _padding3: f32,
    _padding4: f32,
    band_min: f32,
    band_max: f32,
    band_steps: u32,
    _padding5: u32,
    detail_frequency_mult: f32,
    detail_amplitude: f32,
    _padding6: f32,
    _padding7: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainNoiseUniforms {
    seed: u32,
    base_frequency: f32,
    octave_count: u32,
    _padding0: u32,
    frequency_falloff: f32,
    amplitude_falloff: f32,
    _pad1: f32,
    _pad2: f32,
    offset: [f32; 2],
    resolution: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowSunRayUniforms {
    // (width, height, max_steps, reserved)
    dims: [u32; 4],
    // (height_scale, bias, strength, step_size_texels)
    p0: [f32; 4],
    // (sun_x_texels, sun_y_texels, sun_z_height_units, distance_falloff)
    p1: [f32; 4],
    // (edge_softness_height_units, ray_spread_rad, min_shadow, dist_scale)
    p2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CausticsUniforms {
    camera_pos_zoom_thickness: [f32; 4],
    view_size_grid: [f32; 4],
    grid_threshold_padding: [f32; 4],
    border_color: [f32; 4],
    bounds: [f32; 4],
    padding: [f32; 4],
    time_params: [f32; 4],
}

impl Default for ShadowSunRayUniforms {
    fn default() -> Self {
        Self {
            dims: [1, 1, 1, 0],
            p0: [20.0, 0.25, 1.0, 1.0],
            p1: [-0.5, -0.5, 80.0, 0.001],
            p2: [0.5, 0.18, 0.15, 0.005],
        }
    }
}
