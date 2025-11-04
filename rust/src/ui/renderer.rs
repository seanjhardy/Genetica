// UI renderer that draws components to WGPU

use super::component::{Component, ComponentType};
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
    vertex_buffer: Option<wgpu::Buffer>,
    index_buffer: Option<wgpu::Buffer>,
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

        Self {
            screen_width: surface_config.width,
            screen_height: surface_config.height,
            text_renderer,
            rect_pipeline,
            bind_group_layout,
            uniform_buffer,
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    fn compute_layout(&mut self, component: &mut Component) {
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

    pub fn render_backgrounds(
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

        // Compute layout
        self.compute_layout(component);

        // Clear previous frame's rectangles
        self.vertices.clear();
        self.indices.clear();

        // Collect ONLY background rectangles (no borders, shadows, or overlays)
        self.collect_backgrounds_only(component, 0.0, 0.0);
        
        eprintln!("UI Render Backgrounds: Collected {} vertices, {} indices", self.vertices.len(), self.indices.len());

        // Render backgrounds
        self.render_rectangles(device, encoder, view);
    }

    pub fn render_overlays(
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

        // Compute layout (already done, but ensure it's up to date)
        self.compute_layout(component);

        // Clear previous frame's rectangles
        self.vertices.clear();
        self.indices.clear();

        // Collect borders, shadows, and other overlays (no backgrounds)
        self.collect_overlays_only(component, 0.0, 0.0);
        
        eprintln!("UI Render Overlays: Collected {} vertices, {} indices", self.vertices.len(), self.indices.len());

        // Render overlays
        self.render_rectangles(device, encoder, view);

        // Queue and draw all text
        self.queue_text_components(component, queue, 0.0, 0.0);
        self.text_renderer.draw(device, queue, encoder, view);
    }

    fn render_rectangles(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        eprintln!("UI Render: About to render - {} vertices, {} indices", self.vertices.len(), self.indices.len());
        if !self.vertices.is_empty() && !self.indices.is_empty() {
            eprintln!("UI Render: Creating buffers...");
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
            eprintln!("UI Render: Creating bind group...");
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("UI Bind Group"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                }],
            });

            eprintln!("UI Render: Starting render pass with {} indices, screen_size=({}, {})", 
                     self.indices.len(), self.screen_width, self.screen_height);
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

            eprintln!("UI Render: Setting up render pass state...");
            render_pass.set_pipeline(&self.rect_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            eprintln!("UI Render: Drawing {} indices...", self.indices.len());
            render_pass.draw_indexed(0..(self.indices.len() as u32), 0, 0..1);
            eprintln!("UI Render: Draw call completed");
        } else {
            eprintln!("UI Render: No vertices or indices to render! vertices={}, indices={}", 
                     self.vertices.len(), self.indices.len());
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
        
        // Skip if width or height is zero or invalid
        if width <= 0.0 || height <= 0.0 {
            // Still process children even if parent has zero size
            if let ComponentType::View(view) = &component.component_type {
                let child_x = x + component.style.padding.left;
                let child_y = y + component.style.padding.top;
                for child in &view.children {
                    self.collect_backgrounds_only(child, child_x, child_y);
                }
            }
            return;
        }

        // Collect ONLY background (no shadows, borders, or other overlays)
        if component.style.background_color.a > 0.0 {
            eprintln!("UI Render: Adding background rect for {:?} - x:{}, y:{}, w:{}, h:{}, color: {:?}", 
                     component.id, x, y, width, height, component.style.background_color);
            self.add_rect(
                x,
                y,
                width,
                height,
                component.style.border.radius,
                component.style.background_color,
            );
        }

        // Collect children
        if let ComponentType::View(view) = &component.component_type {
            let child_x = x + component.style.padding.left;
            let child_y = y + component.style.padding.top;
            for child in &view.children {
                self.collect_backgrounds_only(child, child_x, child_y);
            }
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
                let child_x = x + component.style.padding.left;
                let child_y = y + component.style.padding.top;
                for child in &view.children {
                    self.collect_overlays_only(child, child_x, child_y);
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
            let child_x = x + component.style.padding.left;
            let child_y = y + component.style.padding.top;
            for child in &view.children {
                self.collect_overlays_only(child, child_x, child_y);
            }
        }
    }

    fn collect_rectangles(
        &mut self,
        component: &Component,
        parent_x: f32,
        parent_y: f32,
    ) {
        if !component.visible {
            // Skipping invisible component
            return;
        }

        let x = parent_x + component.layout.position_x + component.style.margin.left;
        let y = parent_y + component.layout.position_y + component.style.margin.top;
        let width = component.layout.computed_width.max(0.0);
        let height = component.layout.computed_height.max(0.0);
        
        
        // Skip if width or height is zero or invalid
        if width <= 0.0 || height <= 0.0 {
            // Skipping component with zero size
            // Still process children even if parent has zero size
            if let ComponentType::View(view) = &component.component_type {
                let child_x = x + component.style.padding.left;
                let child_y = y + component.style.padding.top;
                for child in &view.children {
                    self.collect_rectangles(child, child_x, child_y);
                }
            }
            return;
        }

        // Collect shadow
        if component.style.shadow.blur > 0.0 || component.style.shadow.spread > 0.0 {
            self.add_shadow_rect(
                x,
                y,
                width,
                height,
                &component.style.shadow,
            );
        }

        // Collect background - always render if there's a non-transparent background
        // For Views, we always want to render backgrounds if they have one
        if component.style.background_color.a > 0.0 {
            eprintln!("UI Render: Adding background rect for {:?} - x:{}, y:{}, w:{}, h:{}, color: {:?}", 
                     component.id, x, y, width, height, component.style.background_color);
            self.add_rect(
                x,
                y,
                width,
                height,
                component.style.border.radius,
                component.style.background_color,
            );
        } else {
            if component.id.as_ref().map(|s| s == "root").unwrap_or(false) || 
               component.id.as_ref().map(|s| s == "simulation").unwrap_or(false) {
                eprintln!("UI Render: Component {:?} has transparent background (a={})", 
                         component.id, component.style.background_color.a);
            }
        }

        // Collect border
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
            let child_x = x + component.style.padding.left;
            let child_y = y + component.style.padding.top;
            eprintln!("UI Render: Component {:?} has {} children", component.id, view.children.len());
            for (i, child) in view.children.iter().enumerate() {
                eprintln!("UI Render: Processing child {} of {:?}: id={:?}, visible={}, computed_size=({}, {}), bg_color={:?}", 
                         i, component.id, child.id, child.visible, 
                         child.layout.computed_width, child.layout.computed_height,
                         child.style.background_color);
                self.collect_rectangles(child, child_x, child_y);
            }
        }
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
        eprintln!("UI Render: add_rect - x:{}, y:{}, w:{}, h:{}, radius:{}, color:{:?}", 
                 x, y, width, height, radius, color);
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

    fn add_border_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        _radius: f32,
        border_width: f32,
        color: Color,
    ) {
        // Render border as outline using 4 rectangles (top, right, bottom, left)
        // Top border
        self.add_simple_rect(x, y, width, border_width, color);
        // Right border
        self.add_simple_rect(x + width - border_width, y, border_width, height, color);
        // Bottom border
        self.add_simple_rect(x, y + height - border_width, width, border_width, color);
        // Left border
        self.add_simple_rect(x, y, border_width, height, color);
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

        let x = parent_x + component.layout.position_x + component.style.margin.left + component.style.padding.left;
        let y = parent_y + component.layout.position_y + component.style.margin.top + component.style.padding.top;

        match &component.component_type {
            ComponentType::Text(text) => {
                self.text_renderer.queue_text_with_size(
                    queue,
                    &text.content,
                    x,
                    y,
                    text.color.to_array(),
                    text.font_size,
                );
            }
            ComponentType::Button(button) => {
                self.text_renderer.queue_text_with_size(
                    queue,
                    &button.label,
                    x,
                    y,
                    button.text_color.to_array(),
                    button.font_size,
                );
            }
            ComponentType::View(view) => {
                let child_x = x - component.style.padding.left; // Children use parent's full area
                let child_y = y - component.style.padding.top;
                for child in &view.children {
                    self.queue_text_components(child, queue, child_x, child_y);
                }
            }
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration, queue: &wgpu::Queue) {
        self.screen_width = new_size.width;
        self.screen_height = new_size.height;
        self.text_renderer.resize(new_size, device, surface_config, queue);
    }
}