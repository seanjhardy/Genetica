// UI renderer that draws components to WGPU

use super::components::{Component, ComponentType, ImageResizeMode};
use super::styles::{Color, Shadow};
use crate::gpu::wgsl::{IMAGE_TEXTURE_SHADER, POST_PROCESSING_SHADER, UI_RECT_SHADER};
use crate::utils::gpu::text_renderer::TextRenderer;
use puffin::profile_scope;
use wgpu;
use wgpu::util::DeviceExt;
use std::collections::HashMap;

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
    time: f32,              // Time for animated effects
    _padding: f32,          // Alignment padding
}

// Renderable element grouped by z-index
#[derive(Clone, Debug)]
struct RenderableElement {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    background_color: Color,
    border: super::styles::Border,
    shadow: Shadow,
    border_radius_tl: f32,
    border_radius_tr: f32,
    border_radius_br: f32,
    border_radius_bl: f32,
    is_viewport: bool,
    viewport_id: Option<String>,
    text_content: Option<(String, f32, Color)>, // content, font_size, color
    text_x: f32,
    text_y: f32,
    image: Option<RenderableImage>,
}

#[derive(Clone, Debug)]
struct RenderableImage {
    source: String,
    tint: Color,
    resize_mode: ImageResizeMode,
    dest_x: f32,
    dest_y: f32,
    dest_width: f32,
    dest_height: f32,
    natural_width: f32,
    natural_height: f32,
}

enum DrawCommand {
    Rect { start: u32, count: u32 },
    Image { start: u32, count: u32, bind_group: wgpu::BindGroup },
    Viewport { start: u32, count: u32, bind_group: wgpu::BindGroup },
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageVertex {
    position: [f32; 2],
    uv: [f32; 2],
    tint: [f32; 4],
}

struct ImageTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

pub struct UiRenderer {
    screen_width: u32,
    screen_height: u32,
    text_renderer: TextRenderer,
    rect_pipeline: wgpu::RenderPipeline,
    rect_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    rect_vertex_buffer: wgpu::Buffer,
    rect_index_buffer: wgpu::Buffer,
    rect_vertex_capacity: usize,
    rect_index_capacity: usize,
    vertices: Vec<UiVertex>,
    indices: Vec<u16>,
    surface_format: wgpu::TextureFormat, // Store surface format for viewport textures
    // Texture sprite rendering
    texture_pipeline: wgpu::RenderPipeline,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_sampler: wgpu::Sampler,
    texture_vertex_buffer: wgpu::Buffer,
    texture_index_buffer: wgpu::Buffer,
    texture_vertex_capacity: usize,
    texture_index_capacity: usize,
    texture_vertices: Vec<TextureVertex>,
    texture_indices: Vec<u16>,
    image_pipeline: wgpu::RenderPipeline,
    image_vertex_buffer: wgpu::Buffer,
    image_index_buffer: wgpu::Buffer,
    image_vertex_capacity: usize,
    image_index_capacity: usize,
    image_vertices: Vec<ImageVertex>,
    image_indices: Vec<u16>,
    image_cache: HashMap<String, ImageTexture>,
    image_bind_groups: HashMap<String, wgpu::BindGroup>,
    draw_commands: Vec<DrawCommand>,
    // Z-index grouped elements (collected at start of frame)
    z_index_groups: std::collections::BTreeMap<i32, Vec<RenderableElement>>,
    // Time tracking for animated effects
    start_time: std::time::Instant,
}

impl UiRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> Self {
        let text_renderer = TextRenderer::new(device, queue, surface_config);

        // Load UI rectangle shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Rectangle Shader"),
            source: UI_RECT_SHADER.clone(),
        });

        // Create uniform buffer
        let uniforms = UiUniforms {
            screen_width: surface_config.width as f32,
            screen_height: surface_config.height as f32,
            time: 0.0,
            _padding: 0.0,
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

        // Bind group can be reused; the uniform buffer contents are updated in place
        let rect_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Preallocate dynamic buffers for textured quads (resized on demand)
        let texture_vertex_capacity = 512;
        let texture_index_capacity = 1024;
        let texture_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Texture Vertex Buffer"),
            size: (texture_vertex_capacity * std::mem::size_of::<TextureVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport Texture Index Buffer"),
            size: (texture_index_capacity * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        // Preallocate dynamic buffers for rectangles (resized on demand)
        let rect_vertex_capacity = 4096;
        let rect_index_capacity = 8192;
        let rect_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Rectangle Vertex Buffer"),
            size: (rect_vertex_capacity * std::mem::size_of::<UiVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rect_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Rectangle Index Buffer"),
            size: (rect_index_capacity * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create texture sprite rendering pipeline
        let post_processing_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Processing Shader"),
            source: POST_PROCESSING_SHADER.clone(),
        });

        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Processing Bind Group Layout"),
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
            label: Some("Post Processing Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let texture_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post Processing Pipeline Layout"),
            bind_group_layouts: &[&texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let texture_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post Processing Pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &post_processing_shader,
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
                module: &post_processing_shader,
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

        // Create image rendering pipeline with tint support
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Texture Shader"),
            source: IMAGE_TEXTURE_SHADER.clone(),
        });

        let image_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Image Texture Pipeline"),
            layout: Some(&texture_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &image_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ImageVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2, // position
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as u64,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2, // uv
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 2,
                            format: wgpu::VertexFormat::Float32x4, // tint
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &image_shader,
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

        let image_vertex_capacity = 512;
        let image_index_capacity = 1024;
        let image_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image Vertex Buffer"),
            size: (image_vertex_capacity * std::mem::size_of::<ImageVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let image_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image Index Buffer"),
            size: (image_index_capacity * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            screen_width: surface_config.width,
            screen_height: surface_config.height,
            text_renderer,
            rect_pipeline,
            rect_bind_group,
            uniform_buffer,
            rect_vertex_buffer,
            rect_index_buffer,
            rect_vertex_capacity,
            rect_index_capacity,
            vertices: Vec::new(),
            indices: Vec::new(),
            surface_format: surface_config.format, // Store format for viewport textures
            texture_pipeline: texture_pipeline.clone(),
            texture_bind_group_layout,
            texture_sampler,
            texture_vertex_buffer,
            texture_index_buffer,
            texture_vertex_capacity,
            texture_index_capacity,
            texture_vertices: Vec::new(),
            texture_indices: Vec::new(),
            image_pipeline,
            image_vertex_buffer,
            image_index_buffer,
            image_vertex_capacity,
            image_index_capacity,
            image_vertices: Vec::new(),
            image_indices: Vec::new(),
            image_cache: HashMap::new(),
            image_bind_groups: HashMap::new(),
            draw_commands: Vec::new(),
            z_index_groups: std::collections::BTreeMap::new(),
            start_time: std::time::Instant::now(),
        }
    }

    pub fn compute_layout(&mut self, component: &mut Component) {
        // For views, recursively compute children first, then calculate own size
        if let ComponentType::View(ref mut view) = component.component_type {
            // Recursively compute layout for children first
            for child in &mut view.children {
                if child.visible {
                    self.compute_layout(child);
                }
            }

            // Now calculate this view's size based on its children
            let calculated_width = view.calculate_width(component.style.padding);

            component.layout.computed_width = match calculated_width {
                super::styles::Size::Pixels(content_width) => {
                    // Use content width, but respect minimum constraints from styles
                    match component.style.width {
                        super::styles::Size::Pixels(min_width) => content_width.max(min_width),
                        super::styles::Size::Percent(percent) => {
                            // For absolute positioned elements, allow content to exceed percent bounds
                            if component.absolute {
                                content_width
                            } else {
                                // For normal elements, use percent as minimum
                                let percent_width = self.screen_width as f32 * percent / 100.0;
                                content_width.max(percent_width)
                            }
                        },
                        _ => content_width,
                    }
                },
                _ => {
                    // Fallback to style-based sizing
                    match component.style.width {
                        super::styles::Size::Pixels(value) => value,
                        super::styles::Size::Percent(value) => self.screen_width as f32 * value / 100.0,
                        super::styles::Size::Flex(_) | super::styles::Size::Auto => self.screen_width as f32,
                    }
                },
            };

            component.layout.computed_height = match view.calculate_height(component.style.padding) {
                super::styles::Size::Pixels(content_height) => {
                    // Use content height, but respect minimum constraints from styles
                    match component.style.height {
                        super::styles::Size::Pixels(min_height) => content_height.max(min_height),
                        super::styles::Size::Percent(percent) => {
                            // For absolute positioned elements, allow content to exceed percent bounds
                            if component.absolute {
                                content_height
                            } else {
                                // For normal elements, use percent as minimum
                                let percent_height = self.screen_height as f32 * percent / 100.0;
                                content_height.max(percent_height)
                            }
                        },
                        _ => content_height,
                    }
                },
                _ => {
                    // Fallback to style-based sizing
                    match component.style.height {
                        super::styles::Size::Pixels(value) => value,
                        super::styles::Size::Percent(value) => self.screen_height as f32 * value / 100.0,
                        super::styles::Size::Flex(_) | super::styles::Size::Auto => self.screen_height as f32,
                    }
                },
            };

            // Only debug TopLeftHUD and its children
            if let Some(id) = &component.id {
                if id == "TopLeftHUD" {
                }
            }
        } else {
            // For non-view components, use content-based sizing where appropriate
            if let ComponentType::Text(ref text) = component.component_type {
                // For text components, computed size should be based on content
                component.layout.computed_width = match component.style.width {
                    super::styles::Size::Pixels(value) => value,
                    super::styles::Size::Percent(value) => self.screen_width as f32 * value / 100.0,
                    super::styles::Size::Flex(_) | super::styles::Size::Auto => {
                        // Use text content width plus padding
                        text.cached_width() + component.style.padding.left + component.style.padding.right
                    },
                };

                component.layout.computed_height = match component.style.height {
                    super::styles::Size::Pixels(value) => value,
                    super::styles::Size::Percent(value) => self.screen_height as f32 * value / 100.0,
                    super::styles::Size::Flex(_) | super::styles::Size::Auto => {
                        // Use text content height plus padding
                        text.cached_height() + component.style.padding.top + component.style.padding.bottom
                    },
                };
            } else {
                // For other non-view components, use style-based sizing
                component.layout.computed_width = match component.style.width {
                    super::styles::Size::Pixels(value) => value,
                    super::styles::Size::Percent(value) => self.screen_width as f32 * value / 100.0,
                    super::styles::Size::Flex(_) | super::styles::Size::Auto => self.screen_width as f32,
                };

                component.layout.computed_height = match component.style.height {
                    super::styles::Size::Pixels(value) => value,
                    super::styles::Size::Percent(value) => self.screen_height as f32 * value / 100.0,
                    super::styles::Size::Flex(_) | super::styles::Size::Auto => self.screen_height as f32,
                };
            }

        }
    }

    /// Collect all elements grouped by z-index at the start of frame
    /// Z-index is calculated implicitly based on child order (later children have higher z-index)
    fn collect_elements_by_z_index(&mut self, component: &mut Component, parent_x: f32, parent_y: f32, base_z_index: i32) {
        if !component.visible {
            return;
        }

        let x = parent_x + component.layout.position_x;
        let y = parent_y + component.layout.position_y;
        let width = component.layout.computed_width.max(0.0);
        let height = component.layout.computed_height.max(0.0);
        
        // Z-index is calculated from base_z_index + explicit z-index if set
        // If explicit z-index is 0 (default), use base_z_index (implicit ordering)
        // If explicit z-index is non-zero, use it (allows explicit control)
        let z_index = if component.style.z_index != 0 {
            component.style.z_index
        } else {
            base_z_index
        };

        // Always create renderable element (we'll filter rendering based on size later)
        // Create renderable element for this component
        let mut element = RenderableElement {
            x,
            y,
            width: component.layout.computed_width,
            height: component.layout.computed_height,
            background_color: component.style.background_color,
            border: component.style.border,
            shadow: component.style.shadow,
            border_radius_tl: component.style.border.radius_tl,
            border_radius_tr: component.style.border.radius_tr,
            border_radius_br: component.style.border.radius_br,
            border_radius_bl: component.style.border.radius_bl,
            is_viewport: matches!(component.component_type, ComponentType::Viewport(_)),
            viewport_id: if matches!(component.component_type, ComponentType::Viewport(_)) {
                component.id.clone()
            } else {
                None
            },
            text_content: None,
            text_x: 0.0,
            text_y: 0.0,
            image: None,
        };

        // Extract text content if this is a text component
        // Always collect text content, even if empty, to ensure it's rendered when updated
        match &component.component_type {
            ComponentType::Text(text) => {
                // Always set text content, even if empty - this ensures it's rendered when updated
                let content_width = (width - component.style.padding.left - component.style.padding.right).max(0.0);
                let content_height = (height - component.style.padding.top - component.style.padding.bottom).max(0.0);
                // Use cached text bounds - same as layout system
                let text_width = text.cached_width();
                let text_height = text.cached_height();
                
                let text_x = if content_width > 0.0 && content_width > text_width {
                    match text.text_align {
                        super::styles::TextAlign::Left => x + component.style.padding.left,
                        super::styles::TextAlign::Center => x + component.style.padding.left + (content_width - text_width) / 2.0,
                        super::styles::TextAlign::Right => x + component.style.padding.left + content_width - text_width,
                    }
                } else {
                    x + component.style.padding.left
                };
                
                let text_y = if content_height > 0.0 && content_height > text_height {
                    y + component.style.padding.top + (content_height - text_height) / 2.0
                } else {
                    y + component.style.padding.top
                };
                
                // Always set text content - even if empty, it will be updated later
                element.text_content = Some((
                    text.content.clone(),
                    text.font_size.max(12.0),
                    text.color,
                ));
                element.text_x = text_x;
                element.text_y = text_y;
            }
            ComponentType::Image(image) => {
                // Collect image data for rendering
                if let Some(source) = &image.source {
                    let content_x = x + component.style.padding.left;
                    let content_y = y + component.style.padding.top;
                    let content_width = (width - component.style.padding.left - component.style.padding.right).max(0.0);
                    let content_height = (height - component.style.padding.top - component.style.padding.bottom).max(0.0);
                    
                    // Calculate actual render dimensions based on resize mode
                    let (dest_x, dest_y, dest_width, dest_height) = if image.natural_width > 0.0 && image.natural_height > 0.0 {
                        let image_aspect = image.natural_width / image.natural_height;
                        let container_aspect = if content_height > 0.0 {
                            content_width / content_height
                        } else {
                            image_aspect
                        };
                        
                        match image.resize_mode {
                            super::components::ImageResizeMode::Contain => {
                                // Scale to fit inside container, maintaining aspect ratio
                                let (scaled_width, scaled_height) = if image_aspect > container_aspect {
                                    // Width-constrained
                                    (content_width, content_width / image_aspect)
                                } else {
                                    // Height-constrained
                                    (content_height * image_aspect, content_height)
                                };
                                
                                // Center the image within the content area
                                let offset_x = (content_width - scaled_width) / 2.0;
                                let offset_y = (content_height - scaled_height) / 2.0;
                                
                                (
                                    content_x + offset_x,
                                    content_y + offset_y,
                                    scaled_width,
                                    scaled_height,
                                )
                            }
                            super::components::ImageResizeMode::Cover => {
                                // Scale to cover entire container, maintaining aspect ratio (may crop)
                                let (scaled_width, scaled_height) = if image_aspect > container_aspect {
                                    // Height-constrained (crop sides)
                                    (content_height * image_aspect, content_height)
                                } else {
                                    // Width-constrained (crop top/bottom)
                                    (content_width, content_width / image_aspect)
                                };
                                
                                // Center the image within the content area
                                let offset_x = (content_width - scaled_width) / 2.0;
                                let offset_y = (content_height - scaled_height) / 2.0;
                                
                                (
                                    content_x + offset_x,
                                    content_y + offset_y,
                                    scaled_width,
                                    scaled_height,
                                )
                            }
                            super::components::ImageResizeMode::Stretch => {
                                // Stretch to fill container (ignore aspect ratio)
                                (content_x, content_y, content_width, content_height)
                            }
                        }
                    } else {
                        // No natural dimensions available, use content size
                        (content_x, content_y, content_width, content_height)
                    };
                    
                    element.image = Some(RenderableImage {
                        source: source.clone(),
                        tint: image.tint,
                        resize_mode: image.resize_mode,
                        dest_x,
                        dest_y,
                        dest_width,
                        dest_height,
                        natural_width: image.natural_width,
                        natural_height: image.natural_height,
                    });
                }
            }
            _ => {}
        }

        // Add element to z-index group
        self.z_index_groups.entry(z_index).or_insert_with(Vec::new).push(element);

        // Recursively collect children
        // Z-index is calculated based on layer index: all children in the same layer share the same z-index
        // This ensures layers are ordered correctly (base layer first, then absolute children on higher layers)
        if let ComponentType::View(view) = &mut component.component_type {
            // Collect children in layer order (base layer first, then absolute children)
            // Z-index is based on the layer index, not the child index within the layer
            // All children in the same layer render at the same z-index
            for (layer_idx, layer) in view.layers.iter().enumerate() {
                // Calculate z-index: parent's z-index + layer_index + 1
                // +1 ensures children appear above their parent
                // All children in this layer share the same z-index
                let layer_z_index = z_index + layer_idx as i32 + 1;
                
                for &child_idx in layer {
                    if child_idx < view.children.len() {
                        let child = &mut view.children[child_idx];
                        
                            self.collect_elements_by_z_index(child, x, y, layer_z_index);
                    }
                }
            }
        }
    }

    /// Unified UI rendering method that renders everything in z-index order
    /// Elements are collected at the start of the frame, then rendered in z-index order (lowest to highest)
    pub fn render(
        &mut self,
        component: &mut Component,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        profile_scope!("UI Render");

        // Update uniform buffer
        {
            profile_scope!("UI Uniforms");
            let elapsed_time = self.start_time.elapsed().as_secs_f32();
            let uniforms = UiUniforms {
                screen_width: self.screen_width as f32,
                screen_height: self.screen_height as f32,
                time: elapsed_time,
                _padding: 0.0,
            };
            queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        }

        // Use conservative selective layout - only skip when completely safe
        let needs_full_layout = self.needs_layout_update(component);
        let needs_selective_layout = self.needs_selective_layout_update(component);

        if needs_full_layout {
            // Full layout recalculation for any dirty components
            {
                profile_scope!("UI Compute Layout");
                self.compute_layout(component);
            }

            // Update layout (positions relative to parents) for View components
            // This must be called after compute_layout so children have their sizes
            if let ComponentType::View(ref mut view) = component.component_type {
                // Rebuild layers before updating layout (ensures children are organized correctly)
                profile_scope!("UI Update Layout");
                view.rebuild_layers();

                // Update layout for root component (positions children)
                view.update_layout(
                    0.0,
                    0.0,
                    component.layout.computed_width,
                    component.layout.computed_height,
                    component.style.padding
                );
            }

            // Clear dirty flags after layout update
            self.clear_layout_dirty_flags(component);
        } else if needs_selective_layout {
            // Selective layout for size changes only
            {
                profile_scope!("UI Selective Layout");
                self.compute_selective_layout(component);
            }

            // Update layout (positions relative to parents) for View components
            // This must be called after compute_layout so children have their sizes
            if let ComponentType::View(ref mut view) = component.component_type {
                // Rebuild layers before updating layout (ensures children are organized correctly)
                profile_scope!("UI Update Layout");
                view.rebuild_layers();

                // Update layout for root component (positions children)
                view.update_layout(
                    0.0,
                    0.0,
                    component.layout.computed_width,
                    component.layout.computed_height,
                    component.style.padding
                );
            }

            // Clear dirty flags after layout update
            self.clear_layout_dirty_flags(component);
        }

        // Clear z-index groups
        {
            profile_scope!("UI Clear Z-Index Groups");
            self.z_index_groups.clear();
        }

        // Collect all elements grouped by z-index
        // Root element starts with z-index 0
        {
            profile_scope!("UI Collect Elements");
            self.collect_elements_by_z_index(component, 0.0, 0.0, 0);
        }

        // Render in z-index order (lowest to highest)
        // BTreeMap automatically sorts by key (z-index)
        // Clone the groups to avoid borrowing issues
        let z_index_groups: Vec<(i32, Vec<RenderableElement>)> = self.z_index_groups.iter().map(|(k, v)| (*k, v.clone())).collect();
        
        // Debug: Check if we have any elements to render
        if z_index_groups.is_empty() {
            // If no elements, render nothing (surface was already cleared to transparent)
            return;
        }

        // Accumulate geometry and draw commands in-order, then render in a single pass
        {
            profile_scope!("UI Clear Draw Commands");
            self.draw_commands.clear();
            self.vertices.clear();
            self.indices.clear();
            self.image_vertices.clear();
            self.image_indices.clear();
            self.texture_vertices.clear();
            self.texture_indices.clear();
        }

        for (_z_index, elements) in z_index_groups {
            profile_scope!("UI Z-Layer");
            let rect_start = self.indices.len();

            // Batch render shadows for this z-layer
            {
                profile_scope!("UI Shadows");
                for element in &elements {
                    if element.width > 0.0
                        && element.height > 0.0
                        && (element.shadow.blur > 0.0 || element.shadow.spread > 0.0)
                    {
                        self.add_shadow_rect(
                            element.x,
                            element.y,
                            element.width,
                            element.height,
                            &element.shadow,
                            element.border_radius_tl,
                            element.border_radius_tr,
                            element.border_radius_br,
                            element.border_radius_bl,
                        );
                    }
                }
            }

            // Batch render backgrounds for this z-layer
            {
                profile_scope!("UI Backgrounds");
                for element in &elements {
                    if element.width <= 0.0
                        || element.height <= 0.0
                        || element.background_color.a <= 0.001
                    {
                        continue;
                    }

                    let has_uniform_radius = element.border_radius_tl == element.border_radius_tr
                        && element.border_radius_tr == element.border_radius_br
                        && element.border_radius_br == element.border_radius_bl;

                    if has_uniform_radius {
                        self.add_rounded_rect(
                            element.x,
                            element.y,
                            element.width,
                            element.height,
                            element.border_radius_tl,
                            element.border_radius_tr,
                            element.border_radius_br,
                            element.border_radius_bl,
                            element.background_color,
                        );
                    } else {
                        let max_dim = element.width.min(element.height) / 2.0;
                        let clamped_radius_tl = element.border_radius_tl.min(max_dim);
                        let clamped_radius_tr = element.border_radius_tr.min(max_dim);
                        let clamped_radius_br = element.border_radius_br.min(max_dim);
                        let clamped_radius_bl = element.border_radius_bl.min(max_dim);

                        if clamped_radius_tl <= 1.0
                            && clamped_radius_tr <= 1.0
                            && clamped_radius_br <= 1.0
                            && clamped_radius_bl <= 1.0
                        {
                            self.add_simple_rect(
                                element.x,
                                element.y,
                                element.width,
                                element.height,
                                element.background_color,
                            );
                        } else {
                            self.add_rounded_rect(
                                element.x,
                                element.y,
                                element.width,
                                element.height,
                                element.border_radius_tl,
                                element.border_radius_tr,
                                element.border_radius_br,
                                element.border_radius_bl,
                                element.background_color,
                            );
                        }
                    }
                }
            }
            {
                profile_scope!("UI Borders");
                for element in &elements {
                    if element.width <= 0.0 || element.height <= 0.0 {
                        continue;
                    }
                    if element.border.width <= 0.0 {
                        continue;
                    }

                    self.add_border_rect(
                        element.x,
                        element.y,
                        element.width,
                        element.height,
                        element.border_radius_tl,
                        element.border_radius_tr,
                        element.border_radius_br,
                        element.border_radius_bl,
                        element.border.width,
                        element.border.color,
                    );
                }
            }

            let rect_count = self.indices.len().saturating_sub(rect_start);
            if rect_count > 0 {
                profile_scope!("UI Rect Draw Command");
                self.draw_commands.push(DrawCommand::Rect {
                    start: rect_start as u32,
                    count: rect_count as u32,
                });
            }

            // Collect images for this z-layer (bind groups reused)
            {
                profile_scope!("UI Images");
                for element in &elements {
                    if let Some(ref image_data) = element.image {
                        if image_data.dest_width > 0.0 && image_data.dest_height > 0.0 {
                            if !self.image_cache.contains_key(&image_data.source) {
                                self.load_image_texture(device, queue, &image_data.source);
                            }

                            if self.image_cache.contains_key(&image_data.source) {
                                if let Some(bind_group) =
                                    self.get_or_create_image_bind_group(device, &image_data.source)
                                {
                                    let image_start = self.image_indices.len();
                                    self.add_image_quad(
                                        image_data.dest_x,
                                        image_data.dest_y,
                                        image_data.dest_width,
                                        image_data.dest_height,
                                        image_data.tint,
                                    );
                                    let image_count =
                                        self.image_indices.len().saturating_sub(image_start);
                                    if image_count > 0 {
                                        self.draw_commands.push(DrawCommand::Image {
                                            start: image_start as u32,
                                            count: image_count as u32,
                                            bind_group,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Render viewport textures (per element due to unique textures)
            {
                profile_scope!("UI Viewports");
                for element in &elements {
                    if !element.is_viewport {
                        continue;
                    }

                    let Some(viewport_id) = &element.viewport_id else {
                        continue;
                    };

                    if element.width <= 0.0 || element.height <= 0.0 {
                        continue;
                    }

                    if let Some(texture_view) = self.get_viewport_texture_view(component, viewport_id) {
                        let texture_start = self.texture_indices.len();
                        self.add_texture_quad(element.x, element.y, element.width, element.height);

                        let texture_count = self.texture_indices.len().saturating_sub(texture_start);
                        if texture_count > 0 {
                            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some(&format!("Viewport Texture Bind Group: {}", viewport_id)),
                                layout: &self.texture_bind_group_layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: wgpu::BindingResource::TextureView(texture_view),
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

                            self.draw_commands.push(DrawCommand::Viewport {
                                start: texture_start as u32,
                                count: texture_count as u32,
                                bind_group,
                            });
                        }
                    }
                }
            }
            // Queue text for this z-layer
            {
                profile_scope!("UI Text Queue");
                for element in &elements {
                    if let Some((content, font_size, color)) = &element.text_content {
                        let color_arr = if color.a < 0.01 {
                            [1.0, 1.0, 1.0, 1.0]
                        } else if color.r < 0.1 && color.g < 0.1 && color.b < 0.1 && color.a > 0.9 {
                            [1.0, 1.0, 1.0, color.a]
                        } else {
                            color.to_array()
                        };

                        if !content.is_empty() && color_arr[3] > 0.01 {
                            self.text_renderer.queue_text_with_size(
                                queue,
                                content,
                                element.text_x,
                                element.text_y,
                                color_arr,
                                *font_size,
                            );
                        }
                    }
                }
            }
        }

        // Upload geometry once per frame
        if !self.vertices.is_empty() && !self.indices.is_empty() {
            profile_scope!("UI Rect Buffer Capacity");
            self.ensure_rect_buffer_capacity(device, self.vertices.len(), self.indices.len());
            queue.write_buffer(
                &self.rect_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.vertices),
            );
            queue.write_buffer(
                &self.rect_index_buffer,
                0,
                bytemuck::cast_slice(&self.indices),
            );
        }

        if !self.image_vertices.is_empty() && !self.image_indices.is_empty() {
            profile_scope!("UI Image Buffer Capacity");
            self.ensure_image_buffer_capacity(device, self.image_vertices.len(), self.image_indices.len());
            queue.write_buffer(
                &self.image_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.image_vertices),
            );
            queue.write_buffer(
                &self.image_index_buffer,
                0,
                bytemuck::cast_slice(&self.image_indices),
            );
        }

        if !self.texture_vertices.is_empty() && !self.texture_indices.is_empty() {
            profile_scope!("UI Texture Buffer Capacity");
            self.ensure_texture_buffer_capacity(
                device,
                self.texture_vertices.len(),
                self.texture_indices.len(),
            );
            queue.write_buffer(
                &self.texture_vertex_buffer,
                0,
                bytemuck::cast_slice(&self.texture_vertices),
            );
            queue.write_buffer(
                &self.texture_index_buffer,
                0,
                bytemuck::cast_slice(&self.texture_indices),
            );
        }

        if self.draw_commands.is_empty() {
            return;
        }

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("UI Batched Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
            ..Default::default()
        });

        let rect_vertex_size = (self.vertices.len() * std::mem::size_of::<UiVertex>()) as u64;
        let image_vertex_size = (self.image_vertices.len() * std::mem::size_of::<ImageVertex>()) as u64;
        let texture_vertex_size =
            (self.texture_vertices.len() * std::mem::size_of::<TextureVertex>()) as u64;

        for cmd in &self.draw_commands {
            profile_scope!("UI Draw Command");
            match cmd {
                DrawCommand::Rect { start, count } => {
                    render_pass.set_pipeline(&self.rect_pipeline);
                    render_pass.set_bind_group(0, &self.rect_bind_group, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        self.rect_vertex_buffer.slice(0..rect_vertex_size),
                    );
                    let start_byte = (*start as usize * std::mem::size_of::<u16>()) as u64;
                    let end_byte = ((*start + *count) as usize * std::mem::size_of::<u16>()) as u64;
                    render_pass.set_index_buffer(
                        self.rect_index_buffer.slice(start_byte..end_byte),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass.draw_indexed(0..*count, 0, 0..1);
                }
                DrawCommand::Image { start, count, bind_group } => {
                    render_pass.set_pipeline(&self.image_pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        self.image_vertex_buffer.slice(0..image_vertex_size),
                    );
                    let start_byte = (*start as usize * std::mem::size_of::<u16>()) as u64;
                    let end_byte = ((*start + *count) as usize * std::mem::size_of::<u16>()) as u64;
                    render_pass.set_index_buffer(
                        self.image_index_buffer.slice(start_byte..end_byte),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass.draw_indexed(0..*count, 0, 0..1);
                }
                DrawCommand::Viewport { start, count, bind_group } => {
                    render_pass.set_pipeline(&self.texture_pipeline);
                    render_pass.set_bind_group(0, bind_group, &[]);
                    render_pass.set_vertex_buffer(
                        0,
                        self.texture_vertex_buffer.slice(0..texture_vertex_size),
                    );
                    let start_byte = (*start as usize * std::mem::size_of::<u16>()) as u64;
                    let end_byte = ((*start + *count) as usize * std::mem::size_of::<u16>()) as u64;
                    render_pass.set_index_buffer(
                        self.texture_index_buffer.slice(start_byte..end_byte),
                        wgpu::IndexFormat::Uint16,
                    );
                    render_pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
        }
    }
    
    /// Render all queued text - should be called once after all UI elements have been rendered
    pub fn render_text(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        self.text_renderer.draw(device, queue, encoder, view);
    }

    fn ensure_rect_buffer_capacity(
        &mut self,
        device: &wgpu::Device,
        vertex_count: usize,
        index_count: usize,
    ) {
        if vertex_count > self.rect_vertex_capacity {
            self.rect_vertex_capacity = vertex_count.next_power_of_two().max(4);
            self.rect_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UI Rectangle Vertex Buffer"),
                size: (self.rect_vertex_capacity * std::mem::size_of::<UiVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if index_count > self.rect_index_capacity {
            self.rect_index_capacity = index_count.next_power_of_two().max(6);
            self.rect_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("UI Rectangle Index Buffer"),
                size: (self.rect_index_capacity * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    fn ensure_texture_buffer_capacity(
        &mut self,
        device: &wgpu::Device,
        vertex_count: usize,
        index_count: usize,
    ) {
        if vertex_count > self.texture_vertex_capacity {
            self.texture_vertex_capacity = vertex_count.next_power_of_two().max(4);
            self.texture_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Viewport Texture Vertex Buffer"),
                size: (self.texture_vertex_capacity * std::mem::size_of::<TextureVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if index_count > self.texture_index_capacity {
            self.texture_index_capacity = index_count.next_power_of_two().max(6);
            self.texture_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Viewport Texture Index Buffer"),
                size: (self.texture_index_capacity * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    fn ensure_image_buffer_capacity(
        &mut self,
        device: &wgpu::Device,
        vertex_count: usize,
        index_count: usize,
    ) {
        if vertex_count > self.image_vertex_capacity {
            self.image_vertex_capacity = vertex_count.next_power_of_two().max(4);
            self.image_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Image Vertex Buffer"),
                size: (self.image_vertex_capacity * std::mem::size_of::<ImageVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if index_count > self.image_index_capacity {
            self.image_index_capacity = index_count.next_power_of_two().max(6);
            self.image_index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Image Index Buffer"),
                size: (self.image_index_capacity * std::mem::size_of::<u16>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
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
        if let ComponentType::Viewport(ref mut viewport) = component.component_type {
            let width = component.layout.computed_width.max(0.0) as u32;
            let height = component.layout.computed_height.max(0.0) as u32;
            
            // Check if texture needs to be created or resized
            let needs_creation = viewport.texture.is_none() || 
                viewport.width != width || 
                viewport.height != height;
            
            if needs_creation && width > 0 && height > 0 {
                // Create or resize viewport texture
                // Use the same format as the surface so render pipelines are compatible
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Viewport Texture: {:?}", component.id)),
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

                // Store texture in viewport component
                viewport.texture = Some(texture);
                viewport.texture_view = Some(texture_view);
                viewport.width = width;
                viewport.height = height;
            }
        }

        // Recursively check children
        if let ComponentType::View(view) = &mut component.component_type {
            for child in &mut view.children {
                self.ensure_viewport_textures(child, device);
            }
        }
    }

    /// Get the texture view for a viewport by ID from the component tree
    pub fn get_viewport_texture_view<'a>(&self, component: &'a Component, viewport_id: &str) -> Option<&'a wgpu::TextureView> {
        // Traverse the component tree to find the viewport with matching ID
        if let Some(component_id) = &component.id {
            if component_id == viewport_id {
                if let ComponentType::Viewport(viewport) = &component.component_type {
                    return viewport.texture_view.as_ref();
                }
            }
        }

        // Recursively search children
        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if let Some(texture_view) = self.get_viewport_texture_view(child, viewport_id) {
                    return Some(texture_view);
                }
            }
        }

        None
    }

    /// Get the size for a viewport by ID from the component tree
    pub fn get_viewport_size(&self, component: &Component, viewport_id: &str) -> Option<(u32, u32)> {
        if let Some(component_id) = &component.id {
            if component_id == viewport_id {
                if let ComponentType::Viewport(viewport) = &component.component_type {
                    return Some((viewport.width, viewport.height));
                }
            }
        }

        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if let Some(size) = self.get_viewport_size(child, viewport_id) {
                    return Some(size);
                }
            }
        }

        None
    }

    fn add_rounded_rect(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        radius_tl: f32,
        radius_tr: f32,
        radius_br: f32,
        radius_bl: f32,
        color: Color,
    ) {
        // Generate a rounded rectangle with per-corner radii
        // We'll use multiple triangles to approximate the rounded corners
        let color_arr = color.to_array();
        let segments = 8; // Number of segments per corner (higher = smoother)
        
        // Clamp radii to ensure they don't overlap
        let max_horizontal = width / 2.0;
        let max_vertical = height / 2.0;
        let radius_tl = radius_tl.min(max_horizontal).min(max_vertical);
        let radius_tr = radius_tr.min(max_horizontal).min(max_vertical);
        let radius_br = radius_br.min(max_horizontal).min(max_vertical);
        let radius_bl = radius_bl.min(max_horizontal).min(max_vertical);
        
        // Calculate bounds for inner rectangle and edge pieces
        let left_radius = radius_tl.max(radius_bl);
        let right_radius = radius_tr.max(radius_br);
        let top_radius = radius_tl.max(radius_tr);
        let bottom_radius = radius_bl.max(radius_br);
        
        // Inner rectangle bounds (excluding all rounded corners)
        let inner_x = x + left_radius;
        let inner_y = y + top_radius;
        let inner_width = (width - left_radius - right_radius).max(0.0);
        let inner_height = (height - top_radius - bottom_radius).max(0.0);

        // Generate center rectangle (the flat part)
        if inner_width > 0.0 && inner_height > 0.0 {
            self.add_simple_rect(inner_x, inner_y, inner_width, inner_height, color);
        }

        // Generate 4 rounded corners with individual radii
        // Top-left corner
        if radius_tl > 0.0 {
        self.add_rounded_corner(
                x + radius_tl, y + radius_tl,
                radius_tl, radius_tl,
            std::f32::consts::PI, // Start at 180 degrees (left), sweep to 90 degrees (down)
            segments,
            color_arr,
        );
        }

        // Top-right corner
        if radius_tr > 0.0 {
        self.add_rounded_corner(
                x + width - radius_tr, y + radius_tr,
                radius_tr, radius_tr,
                3.0 * std::f32::consts::PI / 2.0, // Start at 270 degrees, sweep to 0 degrees
            segments,
            color_arr,
        );
        }

        // Bottom-right corner
        if radius_br > 0.0 {
        self.add_rounded_corner(
                x + width - radius_br, y + height - radius_br,
                radius_br, radius_br,
                0.0, // Start at 0 degrees (right), sweep to 90 degrees
            segments,
            color_arr,
        );
        }

        // Bottom-left corner
        if radius_bl > 0.0 {
        self.add_rounded_corner(
                x + radius_bl, y + height - radius_bl,
                radius_bl, radius_bl,
                std::f32::consts::PI / 2.0, // Start at 90 degrees, sweep to 180 degrees
            segments,
            color_arr,
        );
        }

        // Generate edge flat sections (between corners)
        // Top edge (between top-left and top-right corners)
        let top_edge_x = x + radius_tl;
        let top_edge_width = (width - radius_tl - radius_tr).max(0.0);
        if top_edge_width > 0.0 {
            self.add_simple_rect(top_edge_x, y, top_edge_width, top_radius, color);
        }

        // Bottom edge (between bottom-left and bottom-right corners)
        let bottom_edge_x = x + radius_bl;
        let bottom_edge_width = (width - radius_bl - radius_br).max(0.0);
        if bottom_edge_width > 0.0 {
            self.add_simple_rect(bottom_edge_x, y + height - bottom_radius, bottom_edge_width, bottom_radius, color);
        }

        // Left edge (between top-left and bottom-left corners)
        let left_edge_y = y + radius_tl;
        let left_edge_height = (height - radius_tl - radius_bl).max(0.0);
        if left_edge_height > 0.0 {
            self.add_simple_rect(x, left_edge_y, left_radius, left_edge_height, color);
        }

        // Right edge (between top-right and bottom-right corners)
        let right_edge_y = y + radius_tr;
        let right_edge_height = (height - radius_tr - radius_br).max(0.0);
        if right_edge_height > 0.0 {
            self.add_simple_rect(x + width - right_radius, right_edge_y, right_radius, right_edge_height, color);
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
        tl_radius: f32,
        tr_radius: f32,
        br_radius: f32,
        bl_radius: f32,
        border_width: f32,
        color: Color,
    ) { 
        if tl_radius < 1.0 && tr_radius < 1.0 && br_radius < 1.0 && bl_radius < 1.0 {
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
            let max_tl_radius = tl_radius.min(width / 2.0).min(height / 2.0);
            let max_tr_radius = tr_radius.min(width / 2.0).min(height / 2.0);
            let max_br_radius = br_radius.min(width / 2.0).min(height / 2.0);
            let max_bl_radius = bl_radius.min(width / 2.0).min(height / 2.0);
            self.add_rounded_border_outline(x, y, width, height, max_tl_radius, max_tr_radius, max_br_radius, max_bl_radius, border_width, color);
        }
    }
    
    fn add_rounded_border_outline(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        tl_radius: f32,
        tr_radius: f32,
        br_radius: f32,
        bl_radius: f32,
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
        
        let inner_tl_radius = (tl_radius - border_width).max(0.0);
        let inner_tr_radius = (tr_radius - border_width).max(0.0);
        let inner_br_radius = (br_radius - border_width).max(0.0);
        let inner_bl_radius = (bl_radius - border_width).max(0.0);
        
        // Top border (minus corner areas)
        if width > (tl_radius + tr_radius) {
            self.add_simple_rect(x + tl_radius, y, width - (tl_radius + tr_radius), border_width, color);
        }
        
        // Right border (minus corner areas)
        if height > (tr_radius + br_radius) {
            self.add_simple_rect(x + width - border_width, y + tr_radius, border_width, height - (tr_radius + br_radius), color);
        }
        
        // Bottom border (minus corner areas)
        if width > (br_radius + bl_radius) {
            self.add_simple_rect(x + br_radius, y + height - border_width, width - (br_radius + bl_radius), border_width, color);
        }
        
        // Left border (minus corner areas)
        if height > (bl_radius + tl_radius) {
            self.add_simple_rect(x, y + tl_radius, border_width, height - (bl_radius + tl_radius), color);
        }
        
        // Top-left rounded corner
        if tl_radius > 0.0 {
            self.add_rounded_corner_border(
                x + tl_radius, y + tl_radius, tl_radius, inner_tl_radius,
                std::f32::consts::PI, // Start angle (180 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Top-right rounded corner
        if tr_radius > 0.0 {
            self.add_rounded_corner_border(
                x + width - tr_radius, y + tr_radius, tr_radius, inner_tr_radius,
                3.0 * std::f32::consts::PI / 2.0, // Start angle (90 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Bottom-right rounded corner
        if br_radius > 0.0 {
            self.add_rounded_corner_border(
                x + width - br_radius, y + height - br_radius, br_radius, inner_br_radius,
                0.0, // Start angle (0 degrees)
                8, // segments
                color.to_array(),
            );
        }
        
        // Bottom-left rounded corner
        if bl_radius > 0.0 {
            self.add_rounded_corner_border(
                x + bl_radius, y + height - bl_radius, bl_radius, inner_bl_radius,
                std::f32::consts::PI / 2.0, // Start angle (270 degrees)
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
        tl_radius: f32,
        tr_radius: f32,
        br_radius: f32,
        bl_radius: f32,
    ) {
        // Simple shadow: render as a rectangle with offset and blur approximation
        // For now, we'll use simple rectangles even for rounded elements
        // TODO: Support rounded shadows matching element border-radius
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

        // Only render shadow if it's visible (alpha > 0)
        if shadow_color.a > 0.001 && shadow_width > 0.0 && shadow_height > 0.0 {
            if tl_radius > 0.0 || tr_radius > 0.0 || br_radius > 0.0 || bl_radius > 0.0 {
                // Clamp radius to not exceed half of the smaller dimension
                let max_tl_radius = tl_radius.min(shadow_width / 2.0).min(shadow_height / 2.0);
                let max_tr_radius = tr_radius.min(shadow_width / 2.0).min(shadow_height / 2.0);
                let max_br_radius = br_radius.min(shadow_width / 2.0).min(shadow_height / 2.0);
                let max_bl_radius = bl_radius.min(shadow_width / 2.0).min(shadow_height / 2.0);
                self.add_rounded_rect(shadow_x, shadow_y, shadow_width, shadow_height, max_tl_radius, max_tr_radius, max_br_radius, max_bl_radius, shadow_color);
            } else {
                self.add_simple_rect(shadow_x, shadow_y, shadow_width, shadow_height, shadow_color);
            }
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>, device: &wgpu::Device, surface_config: &wgpu::SurfaceConfiguration, queue: &wgpu::Queue) {
        self.screen_width = new_size.width;
        self.screen_height = new_size.height;
        self.text_renderer.resize(new_size, device, surface_config, queue);
        // Viewport textures will be recreated on next ensure_viewport_textures call
    }

    /// Load an image from disk and upload to GPU texture
    fn load_image_texture(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, source: &str) -> Option<()> {
        // Check if already in cache
        if self.image_cache.contains_key(source) {
            return Some(());
        }

        // Resolve image path using the same logic as Image component
        let resolved_path = super::components::image::resolve_image_path_public(source)?;
        
        // Load image using the image crate
        let img = image::open(&resolved_path).ok()?;
        let rgba_image = img.to_rgba8();
        let dimensions = rgba_image.dimensions();
        
        // Create GPU texture
        let texture_size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Image Texture: {}", source)),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        // Upload image data to GPU
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba_image,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.image_cache.insert(
            source.to_string(),
            ImageTexture {
                texture,
                view,
                width: dimensions.0,
                height: dimensions.1,
            },
        );
        // Any existing bind group for this image should be recreated to match the new view
        self.image_bind_groups.remove(source);
        
        Some(())
    }

    fn get_or_create_image_bind_group(
        &mut self,
        device: &wgpu::Device,
        source: &str,
    ) -> Option<wgpu::BindGroup> {
        if let Some(bg) = self.image_bind_groups.get(source) {
            return Some(bg.clone());
        }

        let texture_view = self.image_cache.get(source)?.view.clone();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Image Bind Group: {}", source)),
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

        self.image_bind_groups
            .insert(source.to_string(), bind_group.clone());
        Some(bind_group)
    }
    
    /// Add an image quad to the vertex buffer with proper UVs and tint
    fn add_image_quad(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        tint: Color,
    ) {
        let base_idx = self.image_vertices.len() as u16;
        
        // Generate 4 vertices with UV coordinates and tint color
        self.image_vertices.push(ImageVertex {
            position: [x, y],
            uv: [0.0, 0.0],
            tint: tint.to_array(),
        });
        self.image_vertices.push(ImageVertex {
            position: [x + width, y],
            uv: [1.0, 0.0],
            tint: tint.to_array(),
        });
        self.image_vertices.push(ImageVertex {
            position: [x + width, y + height],
            uv: [1.0, 1.0],
            tint: tint.to_array(),
        });
        self.image_vertices.push(ImageVertex {
            position: [x, y + height],
            uv: [0.0, 1.0],
            tint: tint.to_array(),
        });
        
        // Generate indices for 2 triangles
        self.image_indices.push(base_idx);
        self.image_indices.push(base_idx + 1);
        self.image_indices.push(base_idx + 2);
        self.image_indices.push(base_idx);
        self.image_indices.push(base_idx + 2);
        self.image_indices.push(base_idx + 3);
    }

    fn needs_layout_update(&self, component: &Component) -> bool {
        if component.layout.is_dirty() {
            return true;
        }

        // Check children recursively
        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if self.needs_layout_update(child) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if component needs selective layout update (only when sizes changed)
    fn needs_selective_layout_update(&self, component: &Component) -> bool {
        if component.layout.has_size_changed() {
            return true;
        }

        // Check children recursively for size changes
        if let ComponentType::View(view) = &component.component_type {
            for child in &view.children {
                if self.needs_selective_layout_update(child) {
                    return true;
                }
            }
        }

        false
    }

    /// Compute layout selectively - only for components that have size changes
    /// Falls back to full layout if any uncertainty exists
    fn compute_selective_layout(&mut self, component: &mut Component) {
        // For safety, if this component is marked as dirty (not just size changed),
        // fall back to full layout
        if component.layout.is_dirty() && !component.layout.has_size_changed() {
            self.compute_layout(component);
            return;
        }

        // If this component has size changes, recompute it fully
        if component.layout.has_size_changed() {
            self.compute_layout(component);
            return;
        }

        // For views, check if any children need updates
        if let ComponentType::View(ref mut view) = component.component_type {
            let mut needs_full_recalc = false;

            // Check each child first
            for child in &mut view.children {
                if child.visible {
                    // If any child needs full layout, we need full layout for this component too
                    if child.layout.is_dirty() {
                        needs_full_recalc = true;
                        break;
                    }
                }
            }

            if needs_full_recalc {
                // Fall back to full layout
                self.compute_layout(component);
                return;
            }

            // Selective update for children with size changes
            let mut needs_parent_recalc = false;

            for child in &mut view.children {
                if child.visible {
                    let had_size_change = child.layout.has_size_changed();

                    // Recursively compute selective layout for child
                    self.compute_selective_layout(child);

                    // If child's size changed during computation, parent needs recalc
                    if child.layout.has_size_changed() && !had_size_change {
                        needs_parent_recalc = true;
                    }
                }
            }

            // If any child had size changes, recompute this view's size
            if needs_parent_recalc {
                let old_width = component.layout.computed_width;
                let old_height = component.layout.computed_height;

                // Recompute this view's size based on children
                let calculated_width = view.calculate_width(component.style.padding);
                component.layout.computed_width = match calculated_width {
                    super::styles::Size::Pixels(content_width) => {
                        match component.style.width {
                            super::styles::Size::Pixels(min_width) => content_width.max(min_width),
                            super::styles::Size::Percent(percent) => {
                                if component.absolute {
                                    content_width
                                } else {
                                    let percent_width = self.screen_width as f32 * percent / 100.0;
                                    content_width.max(percent_width)
                                }
                            },
                            _ => content_width,
                        }
                    },
                    _ => component.layout.computed_width, // Keep existing if calculation failed
                };

                component.layout.computed_height = match view.calculate_height(component.style.padding) {
                    super::styles::Size::Pixels(content_height) => {
                        match component.style.height {
                            super::styles::Size::Pixels(min_height) => content_height.max(min_height),
                            super::styles::Size::Percent(percent) => {
                                if component.absolute {
                                    content_height
                                } else {
                                    let percent_height = self.screen_height as f32 * percent / 100.0;
                                    content_height.max(percent_height)
                                }
                            },
                            _ => content_height,
                        }
                    },
                    _ => component.layout.computed_height, // Keep existing if calculation failed
                };

                // Mark this component as having size changes if its size actually changed
                if (component.layout.computed_width - old_width).abs() > 0.1 ||
                   (component.layout.computed_height - old_height).abs() > 0.1 {
                    component.layout.mark_size_changed();
                }
            }
        }
    }

    fn clear_layout_dirty_flags(&self, component: &mut Component) {
        component.layout.clear_dirty();

        // Clear flags in children recursively
        if let ComponentType::View(ref mut view) = component.component_type {
            for child in &mut view.children {
                self.clear_layout_dirty_flags(child);
            }
        }
    }

}
