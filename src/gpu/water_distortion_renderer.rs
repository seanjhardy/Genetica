use puffin::profile_scope;
use wgpu;

use crate::utils::math::{Rect, Vec2};
use crate::gpu::wgsl::WATER_DISTORTION_SHADER;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaterUniform {
    camera_pos_zoom: [f32; 4],     // camera_x, camera_y, zoom, _padding
    view_size: [f32; 4],           // view_width, view_height, _padding, _padding
    bounds: [f32; 4],              // left, top, right, bottom
    time_params: [f32; 4],         // time, _padding, _padding, _padding
}

pub struct WaterDistortionRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    uniform_data: WaterUniform,
    surface_format: wgpu::TextureFormat,
    // Intermediate texture to render distorted result
    distorted_texture: wgpu::Texture,
    distorted_texture_view: wgpu::TextureView,
    distorted_size: [u32; 2],
}

impl WaterDistortionRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Water Distortion Shader"),
            source: WATER_DISTORTION_SHADER.clone(),
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Water Distortion Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_data = WaterUniform {
            camera_pos_zoom: [0.0, 0.0, 1.0, 0.0],
            view_size: [
                surface_config.width as f32,
                surface_config.height as f32,
                0.0,
                0.0,
            ],
            bounds: [0.0, 0.0, 0.0, 0.0],
            time_params: [0.0, 0.0, 0.0, 0.0],
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Water Distortion Uniform Buffer"),
            size: std::mem::size_of::<WaterUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform_data));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Water Distortion Bind Group Layout"),
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
            label: Some("Water Distortion Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Water Distortion Pipeline"),
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

        // Create intermediate texture
        let distorted_size = [surface_config.width.max(1), surface_config.height.max(1)];
        let distorted_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Water Distorted Texture"),
            size: wgpu::Extent3d {
                width: distorted_size[0],
                height: distorted_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let distorted_texture_view = distorted_texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buffer,
            uniform_data,
            surface_format: surface_config.format,
            distorted_texture,
            distorted_texture_view,
            distorted_size,
        }
    }

    pub fn update(
        &mut self,
        queue: &wgpu::Queue,
        bounds: Rect,
        camera_pos: Vec2,
        zoom: f32,
        view_width: f32,
        view_height: f32,
        time: f32,
    ) {
        self.uniform_data.camera_pos_zoom = [camera_pos.x, camera_pos.y, zoom, 0.0];
        self.uniform_data.view_size = [view_width, view_height, 0.0, 0.0];
        self.uniform_data.bounds = [bounds.left, bounds.top, bounds.right(), bounds.bottom()];
        self.uniform_data.time_params = [time, 0.0, 0.0, 0.0];

        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform_data));
    }

    pub fn apply_distortion(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
        source_texture: &wgpu::Texture,
        target_view: &wgpu::TextureView,
        viewport_size: (u32, u32),
    ) {
        profile_scope!("Apply Water Distortion");

        // Ensure intermediate texture matches viewport size
        if self.distorted_size != [viewport_size.0, viewport_size.1] {
            self.distorted_size = [viewport_size.0, viewport_size.1];
            self.distorted_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Water Distorted Intermediate Texture"),
                size: wgpu::Extent3d {
                    width: viewport_size.0,
                    height: viewport_size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            self.distorted_texture_view = self.distorted_texture.create_view(&wgpu::TextureViewDescriptor::default());
        }

        // Create bind group for this frame's source texture
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Water Distortion Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
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

        // Render distorted version to intermediate texture
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Water Distortion Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.distorted_texture_view,
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
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Copy the intermediate texture back to the source texture
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.distorted_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: viewport_size.0,
                height: viewport_size.1,
                depth_or_array_layers: 1,
            },
        );
    }
}
