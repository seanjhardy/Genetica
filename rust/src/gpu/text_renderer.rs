// Text renderer using glyph_brush for on-screen text

use wgpu;
use wgpu::util::DeviceExt;
use glyph_brush::{GlyphBrush, GlyphBrushBuilder, Section, Text};
use ab_glyph::FontArc;

pub struct TextRenderer {
    glyph_brush: GlyphBrush<()>,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    texture_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    screen_width: u32,
    screen_height: u32,
    vertices: Vec<TextVertex>,
    indices: Vec<u16>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextVertex {
    position: [f32; 2],
    tex_coords: [f32; 2],
    color: [f32; 4],
}

impl TextRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Load font
        let font_data = include_bytes!("../../../assets/fonts/russoone-regular.ttf");
        let font = FontArc::try_from_slice(font_data).unwrap();
        
        // Create glyph brush
        let mut glyph_brush = GlyphBrushBuilder::using_font(font).build();
        
        // Create texture for glyph atlas (will be resized as needed)
        let texture_size = 512u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Text Atlas Texture"),
            size: wgpu::Extent3d {
                width: texture_size,
                height: texture_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create texture bind group layout
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Texture Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            }],
        });
        
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Text Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Text Texture Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Text Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/text.wgsl").into()),
        });
        
        // Create render pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Text Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Text Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TextVertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        
        Self {
            glyph_brush,
            texture,
            texture_view,
            texture_bind_group,
            pipeline,
            screen_width: surface_config.width,
            screen_height: surface_config.height,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn queue_text(&mut self, queue: &wgpu::Queue, text: &str, x: f32, y: f32, color: [f32; 4]) {
        let section = Section::default()
            .add_text(Text::new(text)
                .with_scale(20.0)
                .with_color(color));
        
        // Set screen position via Section
        let section = Section {
            screen_position: (x, y),
            ..section
        };
        
        self.glyph_brush.queue(section);
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        use glyph_brush::BrushAction;
        
        // Clear previous frame's data
        self.vertices.clear();
        self.indices.clear();
        
        // Process queued glyphs
        // The callbacks are called for each rect/tex_data and each vertex
        match self.glyph_brush.process_queued(
            |rect, tex_data| {
                // Update texture with new glyph data
                let width = rect.width();
                let height = rect.height();
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: rect.min[0],
                            y: rect.min[1],
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    tex_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width),
                        rows_per_image: Some(height),
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            },
            |glyph_vertex| {
                // Convert each glyph vertex to our vertex format
                // This callback is called for each vertex
                // Check GlyphVertex structure - it has fields like x, y, u, v, etc.
                let vertex_idx = self.vertices.len();
                self.vertices.push(TextVertex {
                    position: [glyph_vertex.pixel_coords.x as f32, glyph_vertex.pixel_coords.y as f32],
                    tex_coords: [glyph_vertex.tex_coords.x, glyph_vertex.tex_coords.y],
                    color: glyph_vertex.extra,
                });
                
                // Each glyph quad has 4 vertices, we need 6 indices (2 triangles)
                // Generate indices when we have a complete quad (every 4 vertices)
                if vertex_idx % 4 == 0 && vertex_idx >= 3 {
                    let base = (vertex_idx - 3) as u16;
                    // First triangle: 0-1-2
                    self.indices.push(base);
                    self.indices.push(base + 1);
                    self.indices.push(base + 2);
                    // Second triangle: 1-2-3
                    self.indices.push(base + 1);
                    self.indices.push(base + 2);
                    self.indices.push(base + 3);
                }
            },
        ) {
            Ok(BrushAction::Draw(_)) => {
                // Vertices are ready to draw
            }
            Ok(BrushAction::ReDraw) => {
                // Reuse last frame's vertices
            }
            Err(e) => {
                // Handle errors (e.g., texture too small)
                eprintln!("Glyph brush error: {:?}", e);
                return;
            }
        }
        
        // Create/update vertex and index buffers
        if !self.vertices.is_empty() && !self.indices.is_empty() {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Text Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            
            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Text Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            
            self.vertex_buffer = Some(vertex_buffer);
            self.index_buffer = Some(index_buffer);
        }
        
        // Draw text
        if let (Some(vertex_buffer), Some(index_buffer)) = (&self.vertex_buffer, &self.index_buffer) {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Text Render Pass"),
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
            
            render_pass.set_pipeline(&self.pipeline);
            render_pass.set_bind_group(0, &self.texture_bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..(self.indices.len() as u32), 0, 0..1);
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration, queue: &wgpu::Queue) {
        self.screen_width = new_size.width;
        self.screen_height = new_size.height;
        // Resize glyph brush view dimensions
        // Note: glyph_brush doesn't have resize_view, we'll need to recreate or handle differently
        // For now, just update our stored dimensions
    }
}
