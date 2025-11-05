// UI renderer that draws components to WGPU

use super::components::{Component, ComponentType};
use super::styles::{Color, Shadow};
use crate::gpu::text_renderer::TextRenderer;
use wgpu;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TextureVertex {
    position: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct UiUniforms {
    screen_width: f32,
    screen_height: f32,
}

pub struct UiRenderer {
    screen_width: u32,
    screen_height: u32,
    text_renderer: TextRenderer,
    rect_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    vertices: Vec<UiVertex>,
    indices: Vec<u16>,
    viewport_textures: std::collections::HashMap<String, (wgpu::Texture, wgpu::TextureView, u32, u32)>,
    surface_format: wgpu::TextureFormat, // Store surface format for viewport textures
    // Texture sprite rendering
    texture_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_sampler: wgpu::Sampler,
    texture_vertices: Vec<TextureVertex>,
    texture_indices: Vec<u16>,
    viewport_bind_groups: std::collections::HashMap<String, wgpu::BindGroup>,
}

impl UiRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let text_renderer = TextRenderer::new(device, queue, surface_config);

        // Load UI rectangle shader
        let shader_source = include_str!("../shaders/ui_rect.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Rectangle Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create uniform buffer
        let uniforms = UiUniforms {
            screen_width: surface_config.width as f32,
            screen_height: surface_config.height as f32,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("UI Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UI Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Bind group will be created per-frame in render() since uniform buffer is updated each frame

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let rect_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Rectangle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<UiVertex>() as u64,
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

        // Create texture sprite rendering pipeline
        let texture_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Viewport Texture Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/viewport_texture.wgsl").into()),
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Viewport Texture Bind Group Layout"),
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

        let texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Viewport Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Viewport Texture Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let texture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Viewport Texture Pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &texture_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<TextureVertex>() as u64,
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
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &texture_shader,
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
            screen_width: surface_config.width,
            screen_height: surface_config.height,
            text_renderer,
            rect_pipeline,
            bind_group_layout,
            uniform_buffer,
            vertices: Vec::new(),
            indices: Vec::new(),
            viewport_textures: std::collections::HashMap::new(),
            surface_format: surface_config.format, // Store format for viewport textures
            texture_pipeline,
            texture_bind_group_layout,
            texture_sampler,
            texture_vertices: Vec::new(),
            texture_indices: Vec::new(),
            viewport_bind_groups: std::collections::HashMap::new(),
        }
    }

    pub fn compute_layout(&mut self, component: &mut Component) {
        // Compute layout first - this updates component.layout fields
        let screen_width = self.screen_width as f32;
        let screen_height = self.screen_height as f32;
        
        // Ensure root component has proper size (100% width/height from CSS)
        // If the root component doesn't have explicit size, default to full screen
        if matches!(component.style.width, super::styles::Size::Auto) && 
           matches!(component.style.height, super::styles::Size::Auto) {
            component.style.width = super::styles::Size::Percent(100.0);
            component.style.height = super::styles::Size::Percent(100.0);
        }
        
        // Create a temporary layout to compute, then copy back
        let mut temp_layout = component.layout.clone();
        temp_layout.compute_layout(component, screen_width, screen_height);
        component.layout = temp_layout;
    }

    /// Unified UI rendering method that renders everything in HTML order
    /// This renders backgrounds, viewport textures as sprites, overlays, and text all in one pass
    pub fn render(
        &mut self,
        component: &mut Component,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        // Update uniform buffer
        let uniforms = UiUniforms {
            screen_width: self.screen_width as f32,
            screen_height: self.screen_height as f32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Compute layout (sizes)
        self.compute_layout(component);
        
        // Update layout (positions relative to parents) for View components
        // This is necessary to position absolute children correctly
        if let ComponentType::View(ref mut view) = component.component_type {
            view.update_layout(
                0.0, // Root component starts at 0,0
                0.0,
                component.layout.computed_width,
                component.layout.computed_height,
                component.style.padding
            );
        }

        // Clear previous frame's rectangles
        self.vertices.clear();
        self.indices.clear();

        // Collect backgrounds and viewport textures in HTML order
        // We need to separate viewport backgrounds from other backgrounds
        // to render them in the correct order
        self.texture_vertices.clear();
        self.texture_indices.clear();
        self.vertices.clear();
        self.indices.clear();
        
        // Collect all backgrounds (this also collects viewport texture quads)
        self.collect_backgrounds_only(component, 0.0, 0.0);
        
        // Render in correct HTML z-order:
        // - Viewport (child 0) should render first (behind)
        // - UI (child 1) should render second (on top)
        // So render viewport textures first, then UI backgrounds
        
        // First, clear surface and render viewport textures (behind layer)
        self.render_viewport_textures(device, encoder, view);
        
        // Then render UI backgrounds (on top layer) - these render on top of viewports
        self.render_rectangles(device, encoder, view);

        // Clear for overlays
        self.vertices.clear();
        self.indices.clear();

        // Collect overlays (borders, shadows) in HTML order
        self.collect_overlays_only(component, 0.0, 0.0);
        
        // Render overlays
        self.render_rectangles(device, encoder, view);

        // Queue and draw all text in HTML order
        self.queue_text_components(component, queue, 0.0, 0.0);
        self.text_renderer.draw(device, queue, encoder, view);
    }


    pub fn render_rectangles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        if !self.vertices.is_empty() && !self.indices.is_empty() {
            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UI Rectangle Vertex Buffer"),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("UI Rectangle Index Buffer"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            // Create bind group (needed for uniform buffer)
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("UI Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                }],
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Rectangle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Load existing content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });

            render_pass.set_pipeline(&self.rect_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..(self.indices.len() as u32), 0, 0..1);
        }
    }

    fn render_viewport_textures(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        // Always clear the surface first, even if we have no viewport textures
        // This ensures we start with a clean slate
        {
            let _clear_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Clear Surface Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),  // Clear surface to transparent
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });
            // Render pass ends here automatically when it goes out of scope, clearing the surface
        }

        if !self.texture_vertices.is_empty() && !self.texture_indices.is_empty() {
            // Find the first viewport (for now, we'll render all viewports with the same texture)
            // In the future, we might want to track which viewport each quad belongs to
            let viewport_id = "simulation"; // For now, assume we only have one viewport
            
            if let Some(bind_group) = self.viewport_bind_groups.get(viewport_id) {
                let texture_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Viewport Texture Vertex Buffer"),
                    contents: bytemuck::cast_slice(&self.texture_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

                let texture_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Viewport Texture Index Buffer"),
                    contents: bytemuck::cast_slice(&self.texture_indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                // Now render viewport textures on the cleared surface
                // IMPORTANT: The viewport texture should render with alpha blending
                // so that UI backgrounds rendered after can show through
                // But since the viewport texture is opaque (black background from simulation),
                // it will still cover everything. We need to ensure UI backgrounds render AFTER.
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Viewport Texture Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,  // Load cleared transparent content
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    ..Default::default()
                });

                render_pass.set_pipeline(&self.texture_pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.set_vertex_buffer(0, texture_vertex_buffer.slice(..));
                render_pass.set_index_buffer(texture_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..(self.texture_indices.len() as u32), 0, 0..1);
            }
        }
    }

    fn collect_backgrounds_only(
        &mut self,
        component: &Component,
        parent_x: f32,
        parent_y: f32,
    ) {
        if !component.visible {
            return;
        }

        let x = parent_x + component.layout.position_x + component.style.margin.left;
        let y = parent_y + component.layout.position_y + component.style.margin.top;
        let width = component.layout.computed_width.max(0.0);
        let height = component.layout.computed_height.max(0.0);
        
        // For View components, try to render background even if size is zero
        // (children might still have valid sizes)
        let should_render_bg = width > 0.0 && height > 0.0;
        
        // Process children first (even if parent has zero size)
        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if child.absolute {
                    // Absolute children relative to parent
                    self.collect_backgrounds_only(child, x, y);
                } else {
                    // Non-absolute children relative to parent + padding
                    let child_x = x + component.style.padding.left;
                    let child_y = y + component.style.padding.top;
                    self.collect_backgrounds_only(child, child_x, child_y);
                }
            }
        }
        
        // Skip background rendering if width or height is zero or invalid
        if !should_render_bg {
            return;
        }

        // Handle viewport components - render as texture sprite
        // Viewports are rendered in HTML order (later children above earlier ones)
        // The viewport texture contains the simulation content, so we render it as a sprite
        if matches!(component.component_type, ComponentType::Viewport(_)) {
            if let Some(viewport_id) = &component.id {
                if width > 0.0 && height > 0.0 {
                    // Add texture vertices with UV coordinates
                    // This will be rendered BEFORE other backgrounds
                    if self.get_viewport_texture_view(viewport_id).is_some() {
                        self.add_texture_quad(x, y, width, height);
                    }
                }
            }
            return; // Viewports don't have children
        }

        // Collect ONLY background (no shadows, borders, or other overlays)
        // Render background if alpha > 0 and element has valid size
        // If computed size is zero but style has explicit pixel size, use that as fallback
        let render_width = if width > 0.0 {
            width
        } else if let super::styles::Size::Pixels(w) = component.style.width {
            w
        } else {
            0.0
        };
        
        let render_height = if height > 0.0 {
            height
        } else if let super::styles::Size::Pixels(h) = component.style.height {
            h
        } else {
            0.0
        };
        
        if component.style.background_color.a > 0.001 && render_width > 0.0 && render_height > 0.0 {
            self.add_rect(
                x,
                y,
                render_width,
                render_height,
                component.style.border.radius,
                component.style.background_color,
            );
        }
    }


    fn collect_overlays_only(
        &mut self,
        component: &Component,
        parent_x: f32,
        parent_y: f32,
    ) {
        if !component.visible {
            return;
        }

        let x = parent_x + component.layout.position_x + component.style.margin.left;
        let y = parent_y + component.layout.position_y + component.style.margin.top;
        let width = component.layout.computed_width.max(0.0);
        let height = component.layout.computed_height.max(0.0);
        
        // Skip if width or height is zero or invalid
        if width <= 0.0 || height <= 0.0 {
            // Still process children even if parent has zero size
            if let ComponentType::View(view) = &component.component_type {
                for child in &view.children {
                    if child.absolute {
                        // Absolute children relative to parent
                        self.collect_overlays_only(child, x, y);
                    } else {
                        // Non-absolute children relative to parent + padding
                        let child_x = x + component.style.padding.left;
                        let child_y = y + component.style.padding.top;
                        self.collect_overlays_only(child, child_x, child_y);
                    }
                }
            }
            return;
        }

        // Collect shadow (overlay)
        if component.style.shadow.blur > 0.0 || component.style.shadow.spread > 0.0 {
            self.add_shadow_rect(
                x,
                y,
                width,
                height,
                &component.style.shadow,
            );
        }

        // Collect border (overlay)
        if component.style.border.width > 0.0 {
            self.add_border_rect(
                x,
                y,
                width,
                height,
                component.style.border.radius,
                component.style.border.width,
                component.style.border.color,
            );
        }

        // Collect children
        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if child.absolute {
                    // Absolute children relative to parent
                    self.collect_overlays_only(child, x, y);
                } else {
                    // Non-absolute children relative to parent + padding
                    let child_x = x + component.style.padding.left;
                    let child_y = y + component.style.padding.top;
                    self.collect_overlays_only(child, child_x, child_y);
                }
            }
        }
    }

    /// Find viewport components and ensure their textures are created
    pub fn ensure_viewport_textures(
        &mut self,
        component: &mut Component,
        device: &wgpu::Device,
    ) {
        if !component.visible {
            return;
        }


        // If this is a viewport component, create its texture
        if let ComponentType::Viewport(_) = &component.component_type {
            if let Some(viewport_id) = &component.id {
                let width = component.layout.computed_width.max(0.0) as u32;
                let height = component.layout.computed_height.max(0.0) as u32;
                if width > 0 && height > 0 {
                    self.get_viewport_texture(viewport_id, width, height, device);
                }
            }
        }

        // Recursively check children
        if let ComponentType::View(view) = &mut component.component_type {
            for child in &mut view.children {
                self.ensure_viewport_textures(child, device);
            }
        }
    }

    pub fn get_viewport_texture(
        &mut self,
        viewport_id: &str,
        width: u32,
        height: u32,
        device: &wgpu::Device,
    ) -> Option<&wgpu::TextureView> {
        // Check if texture exists and is the right size
        let needs_creation = match self.viewport_textures.get(viewport_id) {
            Some((_texture, _texture_view, existing_width, existing_height)) => {
                *existing_width != width || *existing_height != height
            }
            None => true,
        };

        if needs_creation {
            // Create or resize viewport texture
            // Use the same format as the surface so render pipelines are compatible
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Viewport Texture: {}", viewport_id)),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.surface_format, // Use surface format for compatibility
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });

            let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            // Create bind group for this viewport texture
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Viewport Texture Bind Group: {}", viewport_id)),
                layout: &self.texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.texture_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            self.viewport_textures.insert(
                viewport_id.to_string(),
                (texture, texture_view, width, height),
            );
            self.viewport_bind_groups.insert(viewport_id.to_string(), bind_group);
        }

        self.viewport_textures
            .get(viewport_id)
            .map(|(_t, tv, _w, _h)| tv)
    }

    /// Get the texture view for a viewport by ID
    pub fn get_viewport_texture_view(&self, viewport_id: &str) -> Option<&wgpu::TextureView> {
        self.viewport_textures
            .get(viewport_id)
            .map(|(_t, tv, _w, _h)| tv)
    }

    fn add_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius: f32,
        color: Color,
    ) {
        // Clamp radius to maximum possible (half of smallest dimension)
        let radius = radius.min(width.min(height) / 2.0);

        if radius < 1.0 {
            // Simple rectangle
            self.add_simple_rect(x, y, width, height, color);
        } else {
            // Generate rounded rectangle
            self.add_rounded_rect(x, y, width, height, radius, color);
        }
    }

    fn add_rounded_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius: f32,
        color: Color,
    ) {
        // Generate a rounded rectangle by creating a mesh
        // We'll use multiple triangles to approximate the rounded corners
        let color_arr = color.to_array();
        let segments = 8; // Number of segments per corner (higher = smoother)
        
        // Inner rectangle bounds (excluding rounded corners)
        let inner_x = x + radius;
        let inner_y = y + radius;
        let inner_width = width - 2.0 * radius;
        let inner_height = height - 2.0 * radius;

        // Generate center rectangle (the flat part)
        if inner_width > 0.0 && inner_height > 0.0 {
            self.add_simple_rect(inner_x, inner_y, inner_width, inner_height, color);
        }

        // Generate 4 rounded corners
        // Top-left corner
        self.add_rounded_corner(
            inner_x, inner_y,
            radius, radius,
            std::f32::consts::PI, // Start angle
            segments,
            color_arr,
        );

        // Top-right corner
        self.add_rounded_corner(
            inner_x + inner_width, inner_y,
            radius, radius,
            std::f32::consts::PI / 2.0, // Start angle
            segments,
            color_arr,
        );

        // Bottom-right corner
        self.add_rounded_corner(
            inner_x + inner_width, inner_y + inner_height,
            radius, radius,
            0.0, // Start angle
            segments,
            color_arr,
        );

        // Bottom-left corner
        self.add_rounded_corner(
            inner_x, inner_y + inner_height,
            radius, radius,
            3.0 * std::f32::consts::PI / 2.0, // Start angle
            segments,
            color_arr,
        );

        // Generate top and bottom flat sections
        if inner_width > 0.0 {
            self.add_simple_rect(inner_x, y, inner_width, radius, color);
            self.add_simple_rect(inner_x, y + height - radius, inner_width, radius, color);
        }

        // Generate left and right flat sections
        if inner_height > 0.0 {
            self.add_simple_rect(x, inner_y, radius, inner_height, color);
            self.add_simple_rect(x + width - radius, inner_y, radius, inner_height, color);
        }
    }

    fn add_rounded_corner(
        &mut self,
        center_x: f32,
        center_y: f32,
        radius_x: f32,
        radius_y: f32,
        start_angle: f32,
        segments: usize,
        color: [f32; 4],
    ) {
        let base_idx = self.vertices.len() as u16;
        let angle_step = (std::f32::consts::PI / 2.0) / segments as f32;

        // Add center vertex
        self.vertices.push(UiVertex {
            position: [center_x, center_y],
            color,
        });

        // Generate vertices around the arc
        for i in 0..=segments {
            let angle = start_angle + angle_step * i as f32;
            let px = center_x + radius_x * angle.cos();
            let py = center_y + radius_y * angle.sin();
            
            self.vertices.push(UiVertex {
                position: [px, py],
                color,
            });
        }

        // Generate triangles from center to arc
        for i in 0..segments {
            self.indices.push(base_idx); // Center
            self.indices.push(base_idx + i as u16 + 1);
            self.indices.push(base_idx + i as u16 + 2);
        }
    }

    fn add_simple_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        color: Color,
    ) {
        let base_idx = self.vertices.len() as u16;
        let color_arr = color.to_array();

        // Generate 4 vertices for the rectangle
        // Order: top-left, top-right, bottom-right, bottom-left
        self.vertices.push(UiVertex {
            position: [x, y],
            color: color_arr,
        });
        self.vertices.push(UiVertex {
            position: [x + width, y],
            color: color_arr,
        });
        self.vertices.push(UiVertex {
            position: [x + width, y + height],
            color: color_arr,
        });
        self.vertices.push(UiVertex {
            position: [x, y + height],
            color: color_arr,
        });

        // Generate indices for 2 triangles
        self.indices.push(base_idx);
        self.indices.push(base_idx + 1);
        self.indices.push(base_idx + 2);
        self.indices.push(base_idx);
        self.indices.push(base_idx + 2);
        self.indices.push(base_idx + 3);
    }

    fn add_texture_quad(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) {
        let base_idx = self.texture_vertices.len() as u16;

        // Generate 4 vertices with UV coordinates
        // Order: top-left, top-right, bottom-right, bottom-left
        // UV coordinates: (0,0) top-left, (1,1) bottom-right
        self.texture_vertices.push(TextureVertex {
            position: [x, y],
            uv: [0.0, 0.0],
        });
        self.texture_vertices.push(TextureVertex {
            position: [x + width, y],
            uv: [1.0, 0.0],
        });
        self.texture_vertices.push(TextureVertex {
            position: [x + width, y + height],
            uv: [1.0, 1.0],
        });
        self.texture_vertices.push(TextureVertex {
            position: [x, y + height],
            uv: [0.0, 1.0],
        });

        // Generate indices for 2 triangles
        self.texture_indices.push(base_idx);
        self.texture_indices.push(base_idx + 1);
        self.texture_indices.push(base_idx + 2);
        self.texture_indices.push(base_idx);
        self.texture_indices.push(base_idx + 2);
        self.texture_indices.push(base_idx + 3);
    }

    fn add_border_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius: f32,
        border_width: f32,
        color: Color,
    ) {
        // Clamp radius to maximum possible (half of smallest dimension)
        let radius = radius.min(width.min(height) / 2.0);
        
        if radius < 1.0 {
            // Simple border (no rounded corners)
            // Render border as outline using 4 rectangles (top, right, bottom, left)
            // Top border
            self.add_simple_rect(x, y, width, border_width, color);
            // Right border
            self.add_simple_rect(x + width - border_width, y, border_width, height, color);
            // Bottom border
            self.add_simple_rect(x, y + height - border_width, width, border_width, color);
            // Left border
            self.add_simple_rect(x, y, border_width, height, color);
        } else {
            // Rounded border - render as an outline
            // We'll render 4 straight segments and 4 rounded corner pieces
            self.add_rounded_border_outline(x, y, width, height, radius, border_width, color);
        }
    }
    
    fn add_rounded_border_outline(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius: f32,
        border_width: f32,
        color: Color,
    ) {
        // Render rounded border by creating an outline
        // We'll render the border as separate pieces:
        // 1. Top border (straight segment, minus corner areas)
        // 2. Right border (straight segment, minus corner areas)
        // 3. Bottom border (straight segment, minus corner areas)
        // 4. Left border (straight segment, minus corner areas)
        // 5. 4 rounded corner pieces
        
        let inner_radius = (radius - border_width).max(0.0);
        
        // Top border (minus corner areas)
        if width > 2.0 * radius {
            self.add_simple_rect(x + radius, y, width - 2.0 * radius, border_width, color);
        }
        
        // Right border (minus corner areas)
        if height > 2.0 * radius {
            self.add_simple_rect(x + width - border_width, y + radius, border_width, height - 2.0 * radius, color);
        }
        
        // Bottom border (minus corner areas)
        if width > 2.0 * radius {
            self.add_simple_rect(x + radius, y + height - border_width, width - 2.0 * radius, border_width, color);
        }
        
        // Left border (minus corner areas)
        if height > 2.0 * radius {
            self.add_simple_rect(x, y + radius, border_width, height - 2.0 * radius, color);
        }
        
        // Top-left rounded corner
        if radius > 0.0 {
            self.add_rounded_corner_border(
                x + radius, y + radius, radius, inner_radius,
                std::f32::consts::PI, // Start angle (180 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Top-right rounded corner
        if radius > 0.0 {
            self.add_rounded_corner_border(
                x + width - radius, y + radius, radius, inner_radius,
                std::f32::consts::PI / 2.0, // Start angle (90 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Bottom-right rounded corner
        if radius > 0.0 {
            self.add_rounded_corner_border(
                x + width - radius, y + height - radius, radius, inner_radius,
                0.0, // Start angle (0 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Bottom-left rounded corner
        if radius > 0.0 {
            self.add_rounded_corner_border(
                x + radius, y + height - radius, radius, inner_radius,
                3.0 * std::f32::consts::PI / 2.0, // Start angle (270 degrees)
                8, // segments
                color.to_array(),
            );
        }
    }
    
    fn add_rounded_corner_border(
        &mut self,
        center_x: f32,
        center_y: f32,
        outer_radius: f32,
        inner_radius: f32,
        start_angle: f32,
        segments: usize,
        color: [f32; 4],
    ) {
        // Render a rounded corner border segment as a ring (outer arc - inner arc)
        let base_idx = self.vertices.len() as u16;
        let angle_step = (std::f32::consts::PI / 2.0) / segments as f32;
        
        // Generate vertices for outer and inner arcs
        for i in 0..=segments {
            let angle = start_angle + angle_step * i as f32;
            
            // Outer arc vertex
            let outer_px = center_x + outer_radius * angle.cos();
            let outer_py = center_y + outer_radius * angle.sin();
            self.vertices.push(UiVertex {
                position: [outer_px, outer_py],
                color,
            });
            
            // Inner arc vertex (if inner_radius > 0)
            if inner_radius > 0.0 {
                let inner_px = center_x + inner_radius * angle.cos();
                let inner_py = center_y + inner_radius * angle.sin();
                self.vertices.push(UiVertex {
                    position: [inner_px, inner_py],
                    color,
                });
            } else {
                // If no inner radius, add center point
                self.vertices.push(UiVertex {
                    position: [center_x, center_y],
                    color,
                });
            }
        }
        
        // Generate triangles connecting outer and inner arcs
        for i in 0..segments {
            let outer_idx = base_idx + (i * 2) as u16;
            let inner_idx = base_idx + (i * 2 + 1) as u16;
            let next_outer_idx = base_idx + ((i + 1) * 2) as u16;
            let next_inner_idx = base_idx + ((i + 1) * 2 + 1) as u16;
            
            // First triangle: outer, inner, next outer
            self.indices.push(outer_idx);
            self.indices.push(inner_idx);
            self.indices.push(next_outer_idx);
            
            // Second triangle: inner, next inner, next outer
            self.indices.push(inner_idx);
            self.indices.push(next_inner_idx);
            self.indices.push(next_outer_idx);
        }
    }

    fn add_shadow_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        shadow: &Shadow,
    ) {
        // Simple shadow: render as a rectangle with offset and blur approximation
        let offset_x = shadow.offset_x;
        let offset_y = shadow.offset_y;
        let spread = shadow.spread;
        
        // Expand shadow by spread and blur
        let blur_approx = shadow.blur * 0.5; // Approximate blur as expansion
        let shadow_x = x + offset_x - spread - blur_approx;
        let shadow_y = y + offset_y - spread - blur_approx;
        let shadow_width = width + spread * 2.0 + blur_approx * 2.0;
        let shadow_height = height + spread * 2.0 + blur_approx * 2.0;

        // Use shadow color with reduced alpha for blur effect
        let mut shadow_color = shadow.color;
        shadow_color.a *= 0.5; // Reduce alpha for blur effect

        self.add_simple_rect(shadow_x, shadow_y, shadow_width, shadow_height, shadow_color);
    }

    fn queue_text_components(
        &mut self,
        component: &Component,
        queue: &wgpu::Queue,
        parent_x: f32,
        parent_y: f32,
    ) {
        if !component.visible {
            return;
        }

        // For text/button elements, position should be at the element's content box
        // (position + margin), not including padding (padding is inside the element)
        let x = parent_x + component.layout.position_x + component.style.margin.left;
        let y = parent_y + component.layout.position_y + component.style.margin.top;

        match &component.component_type {
            ComponentType::Text(text) => {
                // Skip empty text
                if text.content.is_empty() {
                    return;
                }
                
                // Ensure font size is at least 12.0 (minimum readable size)
                let font_size = text.font_size.max(12.0);
                
                // Simple positioning: place text at element position + padding
                // Use computed size if available, otherwise use text size
                let content_width = (component.layout.computed_width.max(0.0) - component.style.padding.left - component.style.padding.right).max(0.0);
                let content_height = (component.layout.computed_height.max(0.0) - component.style.padding.top - component.style.padding.bottom).max(0.0);
                
                // Approximate text size
                let text_width = font_size * text.content.len() as f32 * 0.6;
                let text_height = font_size;
                
                // Position text - center if container is larger, otherwise just offset by padding
                let text_x = if content_width > 0.0 && content_width > text_width {
                    x + component.style.padding.left + (content_width - text_width) / 2.0
                } else {
                    x + component.style.padding.left
                };
                
                let text_y = if content_height > 0.0 && content_height > text_height {
                    y + component.style.padding.top + (content_height - text_height) / 2.0
                } else {
                    y + component.style.padding.top
                };
                
                // Ensure text color is visible (default to white if transparent or black)
                let color = if text.color.a < 0.01 || (text.color.r < 0.1 && text.color.g < 0.1 && text.color.b < 0.1 && text.color.a > 0.9) {
                    // If color is transparent or very dark, use white
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    text.color.to_array()
                };
                
                self.text_renderer.queue_text_with_size(
                    queue,
                    &text.content,
                    text_x,
                    text_y,
                    color,
                    font_size,
                );
            }
            ComponentType::Button(button) => {
                // Skip empty button text
                if button.label.is_empty() {
                    return;
                }
                
                // Ensure font size is at least 12.0 (minimum readable size)
                let font_size = button.font_size.max(12.0);
                
                // Simple positioning: place button text at element position + padding
                let content_width = (component.layout.computed_width.max(0.0) - component.style.padding.left - component.style.padding.right).max(0.0);
                let content_height = (component.layout.computed_height.max(0.0) - component.style.padding.top - component.style.padding.bottom).max(0.0);
                
                // Approximate text size
                let text_width = font_size * button.label.len() as f32 * 0.6;
                let text_height = font_size;
                
                // Position text - center if container is larger
                let text_x = if content_width > 0.0 && content_width > text_width {
                    x + component.style.padding.left + (content_width - text_width) / 2.0
                } else {
                    x + component.style.padding.left
                };
                
                let text_y = if content_height > 0.0 && content_height > text_height {
                    y + component.style.padding.top + (content_height - text_height) / 2.0
                } else {
                    y + component.style.padding.top
                };
                
                // Ensure text color is visible (default to white if transparent or black)
                let color = if button.text_color.a < 0.01 || (button.text_color.r < 0.1 && button.text_color.g < 0.1 && button.text_color.b < 0.1 && button.text_color.a > 0.9) {
                    // If color is transparent or very dark, use white
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    button.text_color.to_array()
                };
                
                self.text_renderer.queue_text_with_size(
                    queue,
                    &button.label,
                    text_x,
                    text_y,
                    color,
                    font_size,
                );
            }
            ComponentType::View(view) => {
                // For absolute children, position relative to parent (x - padding, y - padding)
                // For non-absolute children, position relative to parent + padding
                let base_x = x - component.style.padding.left;
                let base_y = y - component.style.padding.top;
                for child in &view.children {
                    if child.absolute {
                        // Absolute children relative to parent
                        self.queue_text_components(child, queue, base_x, base_y);
                    } else {
                        // Non-absolute children relative to parent + padding
                        let child_x = base_x + component.style.padding.left;
                        let child_y = base_y + component.style.padding.top;
                        self.queue_text_components(child, queue, child_x, child_y);
                    }
                }
            }
            ComponentType::Viewport(_) => {
                // Viewports don't have text children
            }
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration, queue: &wgpu::Queue) {
        self.screen_width = new_size.width;
        self.screen_height = new_size.height;
        self.text_renderer.resize(new_size, device, surface_config, queue);
        // Viewport textures will be recreated on next ensure_viewport_textures call
    }
}
