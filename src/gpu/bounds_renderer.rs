use puffin::profile_scope;
use wgpu;

use crate::utils::math::{Rect, Vec2};
use crate::gpu::wgsl::{CAUSTICS_BLIT_SHADER, CAUSTICS_COMPOSITE_SHADER, CAUSTICS_SHADER, ENV_TEXTURE_SHADER};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoundsUniform {
    camera_pos_zoom_thickness: [f32; 4],
    view_size_grid: [f32; 4],
    grid_threshold_padding: [f32; 4],
    border_color: [f32; 4],
    bounds: [f32; 4],
    padding: [f32; 4],
    time_params: [f32; 4],
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct RenderState {
    bounds: Rect,
    camera_pos: Vec2,
    zoom: f32,
    view_size: [f32; 2],
    border_color: [f32; 4],
    grid_opacity: f32,
    time: f32,
}

const LINE_THICKNESS_WORLD: f32 = 2.0;
const GRID_SPACING_WORLD: f32 = 20.0;
const GRID_ZOOM_THRESHOLD: f32 = 20.0;
const CAUSTICS_DOWNSCALE: u32 = 8;
const PARALLAX_STRENGTH: f32 = 0.6;
const PARALLAX_VIEW_HEIGHT: f32 = 3.0;
const PARALLAX_MIN_STEPS: f32 = 10.0;
const PARALLAX_MAX_STEPS: f32 = 36.0;

pub struct BoundsRenderer {
    pipeline: wgpu::RenderPipeline,
    caustics_pipeline: wgpu::RenderPipeline,
    caustics_blit_pipeline: wgpu::RenderPipeline,
    caustics_composite_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    caustics_blit_bind_group_layout: wgpu::BindGroupLayout,
    caustics_composite_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    uniform_data: BoundsUniform,
    uniform_dirty: bool,
    planet_bind_group: Option<(wgpu::BindGroup, usize)>,
    planet_texture_ptr: Option<usize>,
    planet_height_ptr: Option<usize>,
    planet_texture_generation: Option<u64>,
    _fallback_texture: wgpu::Texture,
    fallback_texture_view: wgpu::TextureView,
    caustics_texture: wgpu::Texture,
    caustics_texture_view: wgpu::TextureView,
    caustics_composite_bind_group: wgpu::BindGroup,
    caustics_composite_base_ptr: Option<usize>,
    caustics_composite_height_ptr: Option<usize>,
    composite_texture: wgpu::Texture,
    composite_texture_view: wgpu::TextureView,
    composite_blit_bind_group: wgpu::BindGroup,
    composite_size: [u32; 2],
    caustics_size: [u32; 2],
    surface_format: wgpu::TextureFormat,
    cached_state: Option<RenderState>,
}

impl BoundsRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bounds Shader"),
            source: ENV_TEXTURE_SHADER.clone(),
        });
        let caustics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Shader"),
            source: CAUSTICS_SHADER.clone(),
        });
        let caustics_blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Blit Shader"),
            source: CAUSTICS_BLIT_SHADER.clone(),
        });
        let caustics_composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Caustics Composite Shader"),
            source: CAUSTICS_COMPOSITE_SHADER.clone(),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bounds Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let uniform_data = BoundsUniform {
            camera_pos_zoom_thickness: [0.0, 0.0, 1.0, LINE_THICKNESS_WORLD],
            view_size_grid: [
                surface_config.width as f32,
                surface_config.height as f32,
                GRID_SPACING_WORLD,
                0.0,
            ],
            grid_threshold_padding: [GRID_ZOOM_THRESHOLD, 0.0, 0.0, 0.0],
            border_color: [0.0, 0.0, 0.0, 0.5],
            bounds: [0.0, 0.0, 0.0, 0.0],
            padding: [
                PARALLAX_STRENGTH,
                PARALLAX_VIEW_HEIGHT,
                PARALLAX_MIN_STEPS,
                PARALLAX_MAX_STEPS,
            ],
            time_params: [0.0, 0.0, 0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bounds Uniform Buffer"),
            size: std::mem::size_of::<BoundsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform_data));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bounds Bind Group Layout"),
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

        let caustics_blit_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Caustics Blit Bind Group Layout"),
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let caustics_composite_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Caustics Composite Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bounds Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bounds Render Pipeline"),
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
                    format: surface_config.format,
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
            label: Some("Caustics Render Pipeline"),
            layout: Some(&pipeline_layout),
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
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let caustics_composite_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Caustics Composite Pipeline Layout"),
            bind_group_layouts: &[&caustics_composite_bind_group_layout],
            push_constant_ranges: &[],
        });

        let caustics_composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustics Composite Pipeline"),
            layout: Some(&caustics_composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &caustics_composite_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_composite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
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

        let caustics_blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Caustics Blit Pipeline Layout"),
            bind_group_layouts: &[&caustics_blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let caustics_blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Caustics Blit Pipeline"),
            layout: Some(&caustics_blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &caustics_blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &caustics_blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
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

        let fallback_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Bounds Fallback Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &fallback_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let fallback_texture_view = fallback_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let caustics_size = [
            (surface_config.width / CAUSTICS_DOWNSCALE).max(1),
            (surface_config.height / CAUSTICS_DOWNSCALE).max(1),
        ];
        let caustics_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics Low-Res Texture"),
            size: wgpu::Extent3d {
                width: caustics_size[0],
                height: caustics_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let caustics_texture_view = caustics_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let caustics_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Composite Bind Group"),
            layout: &caustics_composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&fallback_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&caustics_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&fallback_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics Composite Texture"),
            size: wgpu::Extent3d {
                width: surface_config.width.max(1),
                height: surface_config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let composite_texture_view = composite_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let composite_blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Blit Bind Group"),
            layout: &caustics_blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&composite_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            pipeline,
            caustics_pipeline,
            caustics_blit_pipeline,
            caustics_composite_pipeline,
            bind_group_layout,
            caustics_blit_bind_group_layout,
            caustics_composite_bind_group_layout,
            sampler,
            uniform_buffer,
            uniform_data,
            uniform_dirty: true,
            planet_bind_group: None,
            planet_texture_ptr: None,
            planet_height_ptr: None,
            planet_texture_generation: None,
            _fallback_texture: fallback_texture,
            fallback_texture_view,
            caustics_texture,
            caustics_texture_view,
            caustics_composite_bind_group,
            caustics_composite_base_ptr: None,
            caustics_composite_height_ptr: None,
            composite_texture,
            composite_texture_view,
            composite_blit_bind_group,
            composite_size: [surface_config.width.max(1), surface_config.height.max(1)],
            caustics_size,
            surface_format: surface_config.format,
            cached_state: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_bounds(
        &mut self,
        _queue: &wgpu::Queue,
        _bounds_corners: [Vec2; 4],
        bounds: Rect,
        camera_pos: Vec2,
        zoom: f32,
        view_width: f32,
        view_height: f32,
        border_color: [f32; 4],
        time: f32,
    ) {
        let grid_opacity = (zoom * 10.0).clamp(10.0, 60.0) / 255.0;
        let new_state = RenderState {
            bounds,
            camera_pos,
            zoom,
            view_size: [view_width, view_height],
            border_color,
            grid_opacity,
            time,
        };

        if self.cached_state.map_or(true, |state| state != new_state) {
            self.cached_state = Some(new_state);
            self.uniform_dirty = true;
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        planet_texture_view: Option<&wgpu::TextureView>,
        height_texture_view: Option<&wgpu::TextureView>,
        planet_texture_generation: u64,
    ) {
        profile_scope!("Render Bounds");
        let state = match self.cached_state {
            Some(state) => state,
            None => return,
        };

        self.ensure_planet_bind_group(
            device,
            planet_texture_view,
            height_texture_view,
            planet_texture_generation,
        );

        if self.uniform_dirty {
            profile_scope!("Write Bounds Uniform");
            self.write_uniform(queue, state);
        }

        let bind_group = self.planet_bind_group.as_ref().map(|(bg, _)| bg);
        if let Some(bind_group) = bind_group {
            profile_scope!("Encode Bounds Pass");
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bounds Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
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

            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
    }

    fn ensure_planet_bind_group(
        &mut self,
        device: &wgpu::Device,
        planet_texture_view: Option<&wgpu::TextureView>,
        height_texture_view: Option<&wgpu::TextureView>,
        planet_texture_generation: u64,
    ) {
        profile_scope!("Ensure Planet Bind Group");
        let texture_view = planet_texture_view.unwrap_or(&self.fallback_texture_view);
        let height_view = height_texture_view.unwrap_or(&self.fallback_texture_view);
        let ptr = texture_view as *const _ as usize;
        let height_ptr = height_view as *const _ as usize;
        if self.planet_texture_ptr == Some(ptr)
            && self.planet_height_ptr == Some(height_ptr)
            && self.planet_texture_generation == Some(planet_texture_generation)
        {
            return;
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planet Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        self.planet_bind_group = Some((bind_group, ptr));
        self.planet_texture_ptr = Some(ptr);
        self.planet_height_ptr = Some(height_ptr);
        self.planet_texture_generation = Some(planet_texture_generation);
        self.uniform_dirty = true;
    }

    fn write_uniform(&mut self, queue: &wgpu::Queue, state: RenderState) {
        self.uniform_data.camera_pos_zoom_thickness = [
            state.camera_pos.x,
            state.camera_pos.y,
            state.zoom,
            LINE_THICKNESS_WORLD,
        ];
        self.uniform_data.view_size_grid = [
            state.view_size[0],
            state.view_size[1],
            GRID_SPACING_WORLD,
            state.grid_opacity,
        ];
        self.uniform_data.grid_threshold_padding = [GRID_ZOOM_THRESHOLD, 0.0, 0.0, 0.0];
        self.uniform_data.border_color = state.border_color;
        self.uniform_data.bounds = [
            state.bounds.left,
            state.bounds.top,
            state.bounds.right(),
            state.bounds.bottom(),
        ];
        self.uniform_data.padding = [
            PARALLAX_STRENGTH,
            PARALLAX_VIEW_HEIGHT,
            PARALLAX_MIN_STEPS,
            PARALLAX_MAX_STEPS,
        ];
        self.uniform_data.time_params = [state.time, 0.0, 0.0, 0.0];

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform_data));
        self.uniform_dirty = false;
    }

    fn ensure_caustics_texture(&mut self, device: &wgpu::Device, state: RenderState) {
        let bounds_width = state.bounds.width.max(1.0);
        let bounds_height = state.bounds.height.max(1.0);
        let pixels_per_unit_x = state.view_size[0] / bounds_width;
        let pixels_per_unit_y = state.view_size[1] / bounds_height;
        let pixels_per_unit =
            (pixels_per_unit_x + pixels_per_unit_y) * 0.5 / CAUSTICS_DOWNSCALE as f32;
        let target_width = (bounds_width * pixels_per_unit).round().max(1.0) as u32;
        let target_height = (bounds_height * pixels_per_unit).round().max(1.0) as u32;
        if self.caustics_size == [target_width, target_height] {
            return;
        }

        self.caustics_size = [target_width, target_height];
        self.caustics_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics Low-Res Texture"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.caustics_texture_view = self.caustics_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.caustics_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Composite Bind Group"),
            layout: &self.caustics_composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.caustics_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&self.fallback_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        self.caustics_composite_base_ptr = None;
        self.caustics_composite_height_ptr = None;
    }


    fn ensure_composite_texture(&mut self, device: &wgpu::Device, state: RenderState) {
        let target_width = state.view_size[0].round().max(1.0) as u32;
        let target_height = state.view_size[1].round().max(1.0) as u32;
        if self.composite_size == [target_width, target_height] {
            return;
        }

        self.composite_size = [target_width, target_height];
        self.composite_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Caustics Composite Texture"),
            size: wgpu::Extent3d {
                width: target_width,
                height: target_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.composite_texture_view = self.composite_texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.composite_blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Blit Bind Group"),
            layout: &self.caustics_blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.composite_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
    }

    fn ensure_caustics_composite_bind_group(
        &mut self,
        device: &wgpu::Device,
        base_view: &wgpu::TextureView,
        height_view: Option<&wgpu::TextureView>,
    ) {
        let height_view = height_view.unwrap_or(&self.fallback_texture_view);
        let base_ptr = base_view as *const _ as usize;
        let height_ptr = height_view as *const _ as usize;
        if self.caustics_composite_base_ptr == Some(base_ptr)
            && self.caustics_composite_height_ptr == Some(height_ptr)
        {
            return;
        }

        self.caustics_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Caustics Composite Bind Group"),
            layout: &self.caustics_composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.caustics_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(height_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        self.caustics_composite_base_ptr = Some(base_ptr);
        self.caustics_composite_height_ptr = Some(height_ptr);
    }


    pub fn render_caustics(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        planet_texture_view: Option<&wgpu::TextureView>,
        height_texture_view: Option<&wgpu::TextureView>,
        planet_texture_generation: u64,
    ) {
        profile_scope!("Render Caustics");
        let state = match self.cached_state {
            Some(state) => state,
            None => return,
        };

        self.ensure_planet_bind_group(
            device,
            planet_texture_view,
            height_texture_view,
            planet_texture_generation,
        );
        self.ensure_caustics_texture(device, state);
        self.ensure_composite_texture(device, state);
        self.ensure_caustics_composite_bind_group(device, view, height_texture_view);

        if self.uniform_dirty {
            profile_scope!("Write Bounds Uniform");
            self.write_uniform(queue, state);
        }

        let bind_group = self.planet_bind_group.as_ref().map(|(bg, _)| bg);
        if let Some(bind_group) = bind_group {
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Caustics Low-Res Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.caustics_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.caustics_pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Caustics Composite Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.composite_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.caustics_composite_pipeline);
                render_pass.set_bind_group(0, &self.caustics_composite_bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }

            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Caustics Blit Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.caustics_blit_pipeline);
                render_pass.set_bind_group(0, &self.composite_blit_bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }
    }

}
