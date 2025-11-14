// Text renderer using glyph_brush directly for simple on-screen text rendering

use wgpu;
use glyph_brush::{GlyphBrush, GlyphBrushBuilder, Section, Text, OwnedSection, OwnedText, Layout};
use ab_glyph::FontArc;
use wgpu::util::DeviceExt;

use crate::gpu::wgsl::TEXT_SHADER;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    screen_width: f32,
    screen_height: f32,
}

pub struct TextRenderer {
    glyph_brush: GlyphBrush<()>,
    screen_width: u32,
    screen_height: u32,
    texture_size: u32,
    vertices: Vec<TextVertex>,
    indices: Vec<u16>,
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    texture_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    section_id: f32, // Unique ID for each section to prevent hash collisions
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
        _queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        // Load font
        let font_data = include_bytes!("../../../assets/fonts/russoone-regular.ttf");
        let font = FontArc::try_from_slice(font_data).unwrap();
        
        // Create glyph brush
        let glyph_brush = GlyphBrushBuilder::using_font(font).build();
        
        // Create texture for glyph atlas
        // glyph_brush uses a default texture size (typically 256x256 or 512x512)
        // We need to match glyph_brush's internal texture size for coordinates to align
        // Based on the debug output, glyph_brush appears to use 256x256 by default
        // The first texture update is 256 pixels wide, suggesting a 256-wide texture
        let texture_size = 256u32;
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
        
        // Create uniform buffer
        let uniforms = Uniforms {
            screen_width: surface_config.width as f32,
            screen_height: surface_config.height as f32,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Text Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create texture bind group layout
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Text Texture Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Text Shader"),
            source: TEXT_SHADER.clone(),
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
            screen_width: surface_config.width,
            screen_height: surface_config.height,
            texture_size,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
            texture,
            texture_view,
            uniform_buffer,
            texture_bind_group,
            pipeline,
            section_id: 0.0,
        }
    }


    pub fn queue_text_with_size(&mut self, _queue: &wgpu::Queue, text: &str, x: f32, y: f32, color: [f32; 4], font_size: f32) {
        // Round positions to pixel boundaries to prevent sub-pixel jitter
        // This ensures the glyph brush cache works properly
        let x_rounded = x.round();
        let y_rounded = y.round();
        
        let section = OwnedSection {
            screen_position: (x_rounded, y_rounded),
            bounds: (f32::INFINITY, f32::INFINITY), // No clipping
            text: vec![OwnedText {
                text: text.to_string(),
                scale: font_size.into(),
                font_id: glyph_brush::FontId(0),
                extra: glyph_brush::Extra {
                    color: color,
                    z: self.section_id, // Unique z makes each section's hash unique
                },
            }],
            layout: Layout::default(), // Default single-line layout
        };
        
        self.section_id += 1.0; // Increment for next section
        self.glyph_brush.queue(&section);
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) -> Option<wgpu::CommandBuffer> {
        // Reset section ID for next frame
        self.section_id = 0.0;
        
        // Update uniform buffer
        let uniforms = Uniforms {
            screen_width: self.screen_width as f32,
            screen_height: self.screen_height as f32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        use glyph_brush::BrushAction;
        use std::cell::RefCell;
        
        // BUG FIX: Invalidate the glyph brush's internal section hash cache
        // This forces it to regenerate all vertices every frame, preventing 
        // sections from disappearing when their text doesn't change
        // Without this, sections with unchanged text get ReDraw but aren't included
        // in the vertex output, causing them to flicker/disappear
        self.glyph_brush.resize_texture(self.texture_size, self.texture_size);
        
        // Clear previous frame's data (but keep buffers if ReDraw)
        let vertices = RefCell::new(Vec::new());
        let indices = RefCell::new(Vec::new());
        
        // Process queued glyphs
        match self.glyph_brush.process_queued(
            |rect, tex_data| {
                // Update texture with new glyph data
                // rect coordinates are in pixel coordinates relative to glyph_brush's internal texture
                // These coordinates are where to write the glyph data in the texture atlas
                let width = rect.width();
                let height = rect.height();
                let x = rect.min[0] as u32;
                let y = rect.min[1] as u32;
                
                // Ensure coordinates are within texture bounds
                if x + width > self.texture_size || y + height > self.texture_size {
                    eprintln!("Warning: Texture update out of bounds! x={}, y={}, width={}, height={}, texture_size={}", 
                        x, y, width, height, self.texture_size);
                    return;
                }
                
                // wgpu requires bytes_per_row to be aligned to 256 bytes for optimal performance
                // For R8Unorm (1 byte per pixel), we need to align the row size
                // But glyph_brush provides data in a packed format, so we use the actual width
                // Note: wgpu will handle padding internally if needed
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x,
                            y,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    tex_data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(width), // R8Unorm = 1 byte per pixel
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
                // Convert glyph vertex to our vertex format
                // glyph_brush calls this callback ONCE per glyph, not per vertex
                // Each glyph_vertex contains the full rect for that glyph
                // We need to generate 4 vertices (one per corner) from each glyph
                let mut verts = vertices.borrow_mut();
                let mut inds = indices.borrow_mut();
                
                let base_idx = verts.len();
                let color = glyph_vertex.extra.color;
                
                // glyph_brush provides texture coordinates already normalized to [0,1] range
                // These coordinates are relative to glyph_brush's internal texture atlas
                // We need to ensure our texture writes match where these coordinates point
                // Generate 4 vertices for this glyph quad (one per corner)
                // Order: top-left, top-right, bottom-right, bottom-left
                verts.push(TextVertex {
                    position: [glyph_vertex.pixel_coords.min.x as f32, glyph_vertex.pixel_coords.min.y as f32],
                    tex_coords: [glyph_vertex.tex_coords.min.x, glyph_vertex.tex_coords.min.y],
                    color,
                });
                verts.push(TextVertex {
                    position: [glyph_vertex.pixel_coords.max.x as f32, glyph_vertex.pixel_coords.min.y as f32],
                    tex_coords: [glyph_vertex.tex_coords.max.x, glyph_vertex.tex_coords.min.y],
                    color,
                });
                verts.push(TextVertex {
                    position: [glyph_vertex.pixel_coords.max.x as f32, glyph_vertex.pixel_coords.max.y as f32],
                    tex_coords: [glyph_vertex.tex_coords.max.x, glyph_vertex.tex_coords.max.y],
                    color,
                });
                verts.push(TextVertex {
                    position: [glyph_vertex.pixel_coords.min.x as f32, glyph_vertex.pixel_coords.max.y as f32],
                    tex_coords: [glyph_vertex.tex_coords.min.x, glyph_vertex.tex_coords.max.y],
                    color,
                });
                
                // Generate indices for the quad (2 triangles)
                let base = base_idx as u16;
                // First triangle: 0-1-2 (top-left, top-right, bottom-right)
                inds.push(base);
                inds.push(base + 1);
                inds.push(base + 2);
                // Second triangle: 0-2-3 (top-left, bottom-right, bottom-left)
                inds.push(base);
                inds.push(base + 2);
                inds.push(base + 3);
            },
        ) {
            Ok(BrushAction::Draw(_)) => {
                self.vertices = vertices.into_inner();
                self.indices = indices.into_inner();
            }
            Ok(BrushAction::ReDraw) => {
                // BUG FIX: The glyph brush returns ReDraw when the text/position hasn't changed.
                // In this case, we keep using the cached vertices from the previous Draw.
                // The glyph brush maintains its internal texture atlas across frames.
            }
            Err(e) => {
                eprintln!("Glyph brush error: {:?}", e);
                return None;
            }
        }
        
        // Create vertex/index buffers from current cached geometry
        // Note: Even on ReDraw, we must recreate buffers as they're consumed by the render pass
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
        
        None
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, _device: &wgpu::Device, _surface_config: &wgpu::SurfaceConfiguration, queue: &wgpu::Queue) {
        self.screen_width = new_size.width;
        self.screen_height = new_size.height;
        // Update uniform buffer with new screen size
        let uniforms = Uniforms {
            screen_width: new_size.width as f32,
            screen_height: new_size.height as f32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }
}
