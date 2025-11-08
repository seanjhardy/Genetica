use puffin::profile_scope;
use wgpu;

use crate::utils::math::{Rect, Vec2};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoundsUniform {
    camera_pos_zoom_thickness: [f32; 4],
    view_size_grid: [f32; 4],
    grid_threshold_padding: [f32; 4],
    border_color: [f32; 4],
    bounds: [f32; 4],
    padding: [f32; 4],
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct RenderState {
    bounds: Rect,
    camera_pos: Vec2,
    zoom: f32,
    view_size: [f32; 2],
    border_color: [f32; 4],
    grid_opacity: f32,
}

const LINE_THICKNESS_WORLD: f32 = 2.0;
const GRID_SPACING_WORLD: f32 = 20.0;
const GRID_ZOOM_THRESHOLD: f32 = 0.5;

pub struct BoundsRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    uniform_data: BoundsUniform,
    uniform_dirty: bool,
    planet_bind_group: Option<(wgpu::BindGroup, usize)>,
    planet_texture_ptr: Option<usize>,
    _fallback_texture: wgpu::Texture,
    fallback_texture_view: wgpu::TextureView,
    cached_state: Option<RenderState>,
}

impl BoundsRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bounds Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/planet_texture.wgsl").into()),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bounds Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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
            padding: [0.0, 0.0, 0.0, 0.0],
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            uniform_data,
            uniform_dirty: true,
            planet_bind_group: None,
            planet_texture_ptr: None,
            _fallback_texture: fallback_texture,
            fallback_texture_view,
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
    ) {
        let grid_opacity = (zoom * 10.0).clamp(10.0, 60.0) / 255.0;
        let new_state = RenderState {
            bounds,
            camera_pos,
            zoom,
            view_size: [view_width, view_height],
            border_color,
            grid_opacity,
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
    ) {
        profile_scope!("Render Bounds");
        let state = match self.cached_state {
            Some(state) => state,
            None => return,
        };

        self.ensure_planet_bind_group(device, planet_texture_view);

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

    fn ensure_planet_bind_group(&mut self, device: &wgpu::Device, planet_texture_view: Option<&wgpu::TextureView>) {
        profile_scope!("Ensure Planet Bind Group");
        let texture_view = planet_texture_view.unwrap_or(&self.fallback_texture_view);
        let ptr = texture_view as *const _ as usize;
        if self.planet_texture_ptr == Some(ptr) {
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
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });

        self.planet_bind_group = Some((bind_group, ptr));
        self.planet_texture_ptr = Some(ptr);
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
        self.uniform_data.padding = [0.0, 0.0, 0.0, 0.0];

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform_data));
        self.uniform_dirty = false;
    }
}
