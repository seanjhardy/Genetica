// GPU-based simulation bounds rendering (background + border)

use wgpu;
use wgpu::util::DeviceExt;
use crate::utils::math::{Rect, Vec2};

/// Vertex structure for bounds line rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BoundsVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

/// Uniforms for planet/bounds transform (camera and bounds)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BoundsTransform {
    camera_pos: [f32; 2],      // offset 0, 8 bytes
    zoom: f32,                  // offset 8, 4 bytes
    _padding1: f32,             // offset 12, 4 bytes
    view_size: [f32; 2],        // offset 16, 8 bytes
    _padding2: [f32; 2],        // offset 24, 8 bytes (align bounds to 16-byte boundary)
    bounds: [f32; 4],           // offset 32, 16 bytes (vec4 needs 16-byte alignment)
}

/// GPU renderer for simulation bounds (planet background + border + grid)
pub struct BoundsRenderer {
    // Border rendering
    vertex_buffer: wgpu::Buffer,
    num_vertices: u32,
    border_pipeline: wgpu::RenderPipeline,
    
    // Planet background rendering
    background_pipeline: wgpu::RenderPipeline,
    background_bind_group_layout: wgpu::BindGroupLayout,
    background_sampler: wgpu::Sampler,
    background_uniform_buffer: wgpu::Buffer,
    
    // Gridlines rendering (separate pipeline with LineList topology)
    grid_vertex_buffer: wgpu::Buffer,
    num_grid_vertices: u32,
    grid_pipeline: wgpu::RenderPipeline,
}

const LINE_THICKNESS_PX: f32 = 2.0;

impl BoundsRenderer {
    pub fn new(device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration) -> Self {
        // ===== BORDER PIPELINE =====
        // Create a simple line rendering shader
        let border_shader_source = include_str!("../shaders/bounds.wgsl");

        let border_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bounds Border Shader"),
            source: wgpu::ShaderSource::Wgsl(border_shader_source.into()),
        });

        let border_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bounds Border Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let border_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bounds Border Pipeline"),
            layout: Some(&border_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &border_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BoundsVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &border_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
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

        // Create vertex buffer with initial size (will be reused each frame)
        let initial_vertices = vec![
            BoundsVertex { position: [0.0, 0.0], color: [0.0, 0.0, 0.0, 1.0] };
            24
        ];
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounds Vertex Buffer"),
            contents: bytemuck::cast_slice(&initial_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // ===== PLANET BACKGROUND PIPELINE =====
        let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planet Background Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/planet_texture.wgsl").into()),
        });
        
        // Create sampler for background texture
        let background_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Planet Background Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        // Create uniform buffer for transform data
        let background_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Background Transform Buffer"),
            size: std::mem::size_of::<BoundsTransform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout for background
        let background_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planet Background Bind Group Layout"),
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
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout for background
        let background_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planet Background Pipeline Layout"),
            bind_group_layouts: &[&background_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create background render pipeline
        let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planet Background Render Pipeline"),
            layout: Some(&background_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &background_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &background_shader,
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

        // ===== GRIDLINES PIPELINE =====
        // Same shader as border but with TriangleList topology for rectangles (6 vertices per line)
        let grid_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Grid Vertex Buffer"),
            contents: bytemuck::cast_slice(&vec![BoundsVertex { position: [0.0, 0.0], color: [0.0, 0.0, 0.0, 0.3] }; 3000]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        
        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grid Pipeline"),
            layout: Some(&border_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &border_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BoundsVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &border_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING), // Enable alpha blending for semi-transparent grid
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList, // Grid lines rendered as rectangles (triangles)
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

        Self {
            vertex_buffer,
            num_vertices: 0,
            border_pipeline,
            background_pipeline,
            background_bind_group_layout,
            background_sampler,
            background_uniform_buffer,
            grid_vertex_buffer,
            num_grid_vertices: 0,
            grid_pipeline,
        }
    }

    pub fn update_bounds(
        &mut self,
        queue: &wgpu::Queue,
        _bounds_corners: [Vec2; 4],
        bounds: Rect,
        camera_pos: Vec2,
        zoom: f32,
        view_width: f32,
        view_height: f32,
        color: [f32; 4],
    ) {
        // Transform bounds corners from world space to clip space
        let visible_width = view_width / zoom;
        let visible_height = view_height / zoom;

        // === BORDER VERTICES ===
        let half_thickness_x = (LINE_THICKNESS_PX / 2.0) / visible_width * 2.0;
        let half_thickness_y = (LINE_THICKNESS_PX / 2.0) / visible_height * 2.0;

        let clip_left = ((bounds.left - camera_pos.x) / visible_width) * 2.0;
        let clip_right = ((bounds.right() - camera_pos.x) / visible_width) * 2.0;
        let clip_top = -((bounds.top - camera_pos.y) / visible_height) * 2.0;
        let clip_bottom = -((bounds.bottom() - camera_pos.y) / visible_height) * 2.0;

        let mut vertices = Vec::with_capacity(24);

        // Top edge rectangle
        let top_outer = clip_top - half_thickness_y;
        let top_inner = clip_top + half_thickness_y;
        vertices.push(BoundsVertex { position: [clip_left, top_outer], color });
        vertices.push(BoundsVertex { position: [clip_right, top_outer], color });
        vertices.push(BoundsVertex { position: [clip_left, top_inner], color });
        vertices.push(BoundsVertex { position: [clip_right, top_outer], color });
        vertices.push(BoundsVertex { position: [clip_right, top_inner], color });
        vertices.push(BoundsVertex { position: [clip_left, top_inner], color });

        // Bottom edge rectangle
        let bottom_outer = clip_bottom - half_thickness_y;
        let bottom_inner = clip_bottom + half_thickness_y;
        vertices.push(BoundsVertex { position: [clip_left, bottom_outer], color });
        vertices.push(BoundsVertex { position: [clip_right, bottom_outer], color });
        vertices.push(BoundsVertex { position: [clip_left, bottom_inner], color });
        vertices.push(BoundsVertex { position: [clip_right, bottom_outer], color });
        vertices.push(BoundsVertex { position: [clip_right, bottom_inner], color });
        vertices.push(BoundsVertex { position: [clip_left, bottom_inner], color });

        // Left edge rectangle
        let left_outer = clip_left - half_thickness_x;
        let left_inner = clip_left + half_thickness_x;
        vertices.push(BoundsVertex { position: [left_outer, top_outer], color });
        vertices.push(BoundsVertex { position: [left_inner, top_outer], color });
        vertices.push(BoundsVertex { position: [left_outer, bottom_inner], color });
        vertices.push(BoundsVertex { position: [left_inner, top_outer], color });
        vertices.push(BoundsVertex { position: [left_inner, bottom_inner], color });
        vertices.push(BoundsVertex { position: [left_outer, bottom_inner], color });

        // Right edge rectangle
        let right_outer = clip_right - half_thickness_x;
        let right_inner = clip_right + half_thickness_x;
        vertices.push(BoundsVertex { position: [right_outer, top_outer], color });
        vertices.push(BoundsVertex { position: [right_inner, top_outer], color });
        vertices.push(BoundsVertex { position: [right_outer, bottom_inner], color });
        vertices.push(BoundsVertex { position: [right_inner, top_outer], color });
        vertices.push(BoundsVertex { position: [right_inner, bottom_inner], color });
        vertices.push(BoundsVertex { position: [right_outer, bottom_inner], color });
        
        self.num_vertices = vertices.len() as u32;
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        
        // === GRIDLINES VERTICES ===
        // Grid spacing: 20 pixels (matching C++ version)
        const GRID_SPACING: f32 = 20.0;
        
        // Calculate opacity based on zoom (10-60 range in C++)
        let opacity = (zoom * 10.0).clamp(10.0, 60.0) / 255.0;
        let grid_color = [0.0, 0.0, 0.0, opacity];
        
        let mut grid_vertices = Vec::new();
        
        // Convert thickness to clip space
        let half_thickness_x = (LINE_THICKNESS_PX / 2.0) / visible_width * 2.0;
        let half_thickness_y = (LINE_THICKNESS_PX / 2.0) / visible_height * 2.0;
        
        // Vertical lines (as rectangles)
        let num_vertical = (bounds.width / GRID_SPACING).floor() as i32;
        for i in 1..(num_vertical - 1) {
            let x = bounds.left + i as f32 * GRID_SPACING;
            
            // Top and bottom Y positions
            let rel_y1 = bounds.top - camera_pos.y;
            let rel_y2 = bounds.bottom() - camera_pos.y;
            let clip_y1 = -(rel_y1 / visible_height) * 2.0;
            let clip_y2 = -(rel_y2 / visible_height) * 2.0;
            
            // Left and right X positions (with thickness)
            let rel_x = x - camera_pos.x;
            let clip_x = (rel_x / visible_width) * 2.0;
            let clip_x_left = clip_x - half_thickness_x;
            let clip_x_right = clip_x + half_thickness_x;
            
            // Create rectangle as two triangles (6 vertices)
            // Triangle 1: top-left, top-right, bottom-left
            grid_vertices.push(BoundsVertex { position: [clip_x_left, clip_y1], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x_right, clip_y1], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x_left, clip_y2], color: grid_color });
            // Triangle 2: top-right, bottom-right, bottom-left
            grid_vertices.push(BoundsVertex { position: [clip_x_right, clip_y1], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x_right, clip_y2], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x_left, clip_y2], color: grid_color });
        }
        
        // Horizontal lines (as rectangles)
        let num_horizontal = (bounds.height / GRID_SPACING).floor() as i32;
        for i in 1..(num_horizontal - 1) {
            let y = bounds.top + i as f32 * GRID_SPACING;
            
            // Left and right X positions
            let rel_x1 = bounds.left - camera_pos.x;
            let rel_x2 = bounds.right() - camera_pos.x;
            let clip_x1 = (rel_x1 / visible_width) * 2.0;
            let clip_x2 = (rel_x2 / visible_width) * 2.0;
            
            // Top and bottom Y positions (with thickness)
            let rel_y = y - camera_pos.y;
            let clip_y = -(rel_y / visible_height) * 2.0;
            let clip_y_top = clip_y - half_thickness_y;
            let clip_y_bottom = clip_y + half_thickness_y;
            
            // Create rectangle as two triangles (6 vertices)
            // Triangle 1: top-left, top-right, bottom-left
            grid_vertices.push(BoundsVertex { position: [clip_x1, clip_y_top], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x2, clip_y_top], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x1, clip_y_bottom], color: grid_color });
            // Triangle 2: top-right, bottom-right, bottom-left
            grid_vertices.push(BoundsVertex { position: [clip_x2, clip_y_top], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x2, clip_y_bottom], color: grid_color });
            grid_vertices.push(BoundsVertex { position: [clip_x1, clip_y_bottom], color: grid_color });
        }
        
        self.num_grid_vertices = grid_vertices.len() as u32;
        if !grid_vertices.is_empty() {
            queue.write_buffer(&self.grid_vertex_buffer, 0, bytemuck::cast_slice(&grid_vertices));
        }
    }

    /// Render planet background (if texture is provided) and bounds border
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        planet_texture_view: Option<&wgpu::TextureView>,
        camera_pos: Vec2,
        zoom: f32,
        view_size: Vec2,
        bounds: Rect,
    ) {
        // Render planet background if available
        if let Some(planet_texture) = planet_texture_view {
            // Update transform uniforms
            let transform = BoundsTransform {
                camera_pos: [camera_pos.x, camera_pos.y],
                zoom,
                _padding1: 0.0,
                view_size: [view_size.x, view_size.y],
                _padding2: [0.0, 0.0],
                bounds: [bounds.left, bounds.top, bounds.right(), bounds.bottom()],
            };
            queue.write_buffer(&self.background_uniform_buffer, 0, bytemuck::bytes_of(&transform));
            
            // Create bind group for planet texture
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planet Background Bind Group"),
                layout: &self.background_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(planet_texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.background_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.background_uniform_buffer.as_entire_binding(),
                    },
                ],
            });
            
            // Render planet background
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Planet Background Render Pass"),
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
            
            render_pass.set_pipeline(&self.background_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..4, 0..1); // Full-screen quad
        }
        
        // Render gridlines (semi-transparent black overlay)
        let render_grid = zoom > 4.0;
        if self.num_grid_vertices > 0 && render_grid {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Grid Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Load existing content (planet background)
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });

            render_pass.set_pipeline(&self.grid_pipeline);
            render_pass.set_vertex_buffer(0, self.grid_vertex_buffer.slice(..));
            render_pass.draw(0..self.num_grid_vertices, 0..1);
        }
        
        // Render bounds border (white outline)
        if self.num_vertices > 0 {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bounds Border Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Load existing content (planet + grid)
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });

            render_pass.set_pipeline(&self.border_pipeline);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }
    }
}

