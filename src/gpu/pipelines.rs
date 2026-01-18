// GPU pipelines module - manages compute and render pipelines

use wgpu;
use wgpu::util::DeviceExt;
use std::path::PathBuf;

use crate::gpu::wgsl::{CELLS_KERNEL, CELLS_SHADER, LINKS_KERNEL, LINKS_SHADER, NUTRIENTS_KERNEL, NUTRIENTS_SHADER, PERLIN_NOISE_TEXTURE_SHADER, PICK_CELL_KERNEL, SPAWN_CELLS_KERNEL, VERLET_KERNEL};
use crate::gpu::buffers::GpuBuffers;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NoiseTextureUniforms {
    seed: f32,
    base_frequency: f32,
    octave_count: u32,
    _padding0: u32,
    frequency_falloff: f32,
    amplitude_falloff: f32,
    time: f32,
    _padding3: f32,
}
/// Compute pipelines for physics simulation
pub struct ComputePipelines {
    pub update_cells: wgpu::ComputePipeline,
    pub update_cells_bind_group: wgpu::BindGroup,
    pub update_nutrients: wgpu::ComputePipeline,
    pub update_links: wgpu::ComputePipeline,
    pub update_points: wgpu::ComputePipeline,
    pub update_points_bind_group: wgpu::BindGroup,
    pub spawn_cells: wgpu::ComputePipeline,
    pub spawn_cells_bind_groups: [wgpu::BindGroup; 2],
    pub pick_cell: wgpu::ComputePipeline,
    pub pick_cell_bind_group: wgpu::BindGroup,
}

impl ComputePipelines {
    pub fn new(
        device: &wgpu::Device,
        buffers: &GpuBuffers,
    ) -> Self {
        let points_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Shader"),
            source: VERLET_KERNEL.clone(),
        });

        let points_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Points Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let points_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Points Pipeline Layout"),
            bind_group_layouts: &[&points_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_points = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Points Pipeline"),
            layout: Some(&points_pipeline_layout),
            module: &points_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let update_points_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Points Bind Group"),
            layout: &points_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.points.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.link_corrections.as_entire_binding(),
                },
            ],
        });

        // Create spawn points shader and pipeline
        let spawn_cells_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spawn Points Shader"),
            source: SPAWN_CELLS_KERNEL.clone(),
        });

        let spawn_cells_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spawn Cells Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Points buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Points counter
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Points free list
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell buffer bindings
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell counter
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Cell free list
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Event buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Event counter
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Lifeform ID buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Links buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Link free list
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Division requests buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Division counter
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pick_cell_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pick Cell Shader"),
            source: PICK_CELL_KERNEL.clone(),
        });

        let pick_cell_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Pick Cell Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pick_cell_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pick Cell Pipeline Layout"),
            bind_group_layouts: &[&pick_cell_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pick_cell = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pick Cell Pipeline"),
            layout: Some(&pick_cell_pipeline_layout),
            module: &pick_cell_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let pick_cell_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pick Cell Bind Group"),
            layout: &pick_cell_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.cell_pick_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.points.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cells.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.cell_pick_result.as_entire_binding(),
                },
            ],
        });

        let spawn_cells_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spawn Points Pipeline Layout"),
            bind_group_layouts: &[&spawn_cells_bind_group_layout],
            push_constant_ranges: &[],
        });
        let spawn_cells = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Spawn Points Pipeline"),
            layout: Some(&spawn_cells_pipeline_layout),
            module: &spawn_cells_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let make_spawn_bind_group = |event_buffer: &wgpu::Buffer, event_counter: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Spawn Cells Bind Group"),
                layout: &spawn_cells_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffers.uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buffers.points.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffers.points_counter.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: buffers.points.free_list_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: buffers.cells.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buffers.cells_counter.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buffers.cells.free_list_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: event_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: event_counter.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: buffers.lifeform_id.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: buffers.link_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: buffers.link_free_list.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: buffers.division_requests.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: buffers.division_counter.as_entire_binding(),
                    },
                ],
            })
        };

        let (event_buffer_0, event_counter_0) = buffers.event_system.gpu_buffers_for_index(0);
        let (event_buffer_1, event_counter_1) = buffers.event_system.gpu_buffers_for_index(1);
        let spawn_cells_bind_groups = [
            make_spawn_bind_group(event_buffer_0, event_counter_0),
            make_spawn_bind_group(event_buffer_1, event_counter_1),
        ];

        // Create shader module
        let cells_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cells Shader"),
            source: CELLS_KERNEL.clone(),
        });

        // New compact bind group layout matching simplified kernels
        let cells_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cells Bind Group Layout"),
            entries: &[
                // 0 uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1 physics points
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2 physics free list
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3 physics counter
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4 cells
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5 cell free list
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6 cell counter
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7 spawn buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8 nutrient grid
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9 links
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10 link free list
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11 division requests
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12 division counter
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 13 link corrections
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let cells_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&cells_bind_group_layout],
            push_constant_ranges: &[],
        });
        let update_cells = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Compute Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &cells_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let links_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Links Shader"),
            source: LINKS_KERNEL.clone(),
        });

        let update_links = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Links Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &links_shader,
            entry_point: Some("accumulate"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_cells_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cells Bind Group"),
            layout: &cells_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.points.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.points.free_list_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.points_counter.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.cells.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.cells.free_list_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.cells_counter.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.nutrient_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.link_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.link_free_list.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.points_counter.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.division_requests.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: buffers.division_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: buffers.link_corrections.as_entire_binding(),
                },
            ],
        });

        let nutrient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Regeneration Shader"),
            source: NUTRIENTS_KERNEL.clone(),
        });

        let update_nutrients = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nutrient Regenerate Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &nutrient_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            update_cells,
            update_cells_bind_group,
            update_nutrients,
            update_links,
            update_points,
            update_points_bind_group,
            spawn_cells,
            spawn_cells_bind_groups,
            pick_cell,
            pick_cell_bind_group,
        }
    }
}

/// Render pipelines for drawing points
pub struct RenderPipelines {
    pub cells: wgpu::RenderPipeline,
    pub cell_render_bind_group: wgpu::BindGroup,
    pub links: wgpu::RenderPipeline,
    pub link_bind_group: wgpu::BindGroup,
    pub nutrient_overlay: wgpu::RenderPipeline,
    pub nutrient_bind_group: wgpu::BindGroup,
    // Keep texture and sampler alive
    _nucleus_texture: wgpu::Texture,
    _nucleus_sampler: wgpu::Sampler,
    pub perlin_noise_texture: wgpu::Texture,
    pub perlin_noise_sampler: wgpu::Sampler,
    // Components for noise texture regeneration
    noise_pipeline: wgpu::RenderPipeline,
    noise_bind_group: wgpu::BindGroup,
    noise_uniform_buffer: wgpu::Buffer,
    perlin_noise_texture_view: wgpu::TextureView,
}

impl RenderPipelines {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        buffers: &GpuBuffers,
    ) -> Self {
        let link_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Link Render Shader"),
            source: LINKS_SHADER.clone(),
        });

        let nutrient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Overlay Shader"),
            source: NUTRIENTS_SHADER.clone(),
        });
        let cell_render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let link_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Link Render Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let nutrient_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Overlay Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let link_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Link Render Pipeline Layout"),
            bind_group_layouts: &[&link_bind_group_layout],
            push_constant_ranges: &[],
        });

        let cell_render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cell Render Pipeline Layout"),
            bind_group_layouts: &[&cell_render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let cell_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Render Shader"),
            source: CELLS_SHADER.clone(),
        });

        let cells = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Render Pipeline"),
            layout: Some(&cell_render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cell_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cell_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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

        let nutrient_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nutrient Overlay Pipeline Layout"),
            bind_group_layouts: &[&nutrient_bind_group_layout],
            push_constant_ranges: &[],
        });

        let links = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Link Render Pipeline"),
            layout: Some(&link_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &link_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &link_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        let nutrient_overlay = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Nutrient Overlay Pipeline"),
            layout: Some(&nutrient_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &nutrient_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &nutrient_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        // Load nucleus texture
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let nucleus_path = manifest_dir.join("assets/textures/nucleus.png");
        
        let nucleus_texture = if nucleus_path.exists() {
            let img = image::open(&nucleus_path)
                .expect("Failed to open nucleus.png")
                .to_rgba8();
            let dimensions = img.dimensions();
            
            let texture_size = wgpu::Extent3d {
                width: dimensions.0,
                height: dimensions.1,
                depth_or_array_layers: 1,
            };
            
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Nucleus Texture"),
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            
            // Upload image data to GPU
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &img,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * dimensions.0),
                    rows_per_image: Some(dimensions.1),
                },
                texture_size,
            );
            
            texture
        } else {
            // Create a 1x1 white texture as fallback
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Nucleus Texture Fallback"),
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
            })
        };
        
        let nucleus_texture_view = nucleus_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Create sampler for nucleus texture
        let nucleus_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Nucleus Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Generate perlin noise texture (400x400, single channel stored in RGBA)
        const NOISE_TEXTURE_SIZE: u32 = 400;
        let perlin_noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Perlin Noise Texture"),
            size: wgpu::Extent3d {
                width: NOISE_TEXTURE_SIZE,
                height: NOISE_TEXTURE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm, // Use RGBA for compatibility
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let perlin_noise_texture_view = perlin_noise_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create shader for perlin noise texture generation
        let perlin_noise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Perlin Noise Texture Shader"),
            source: PERLIN_NOISE_TEXTURE_SHADER.clone(),
        });

        // Create uniform buffer for perlin noise texture

        let noise_uniforms = NoiseTextureUniforms {
            seed: 0.2, // Default seed
            base_frequency: 4.0, // Base frequency for first octave - increased for more detail
            octave_count: 6, // Number of octaves for fractal noise
            _padding0: 0,
            frequency_falloff: 0.4, // Each octave has double the frequency (standard Perlin)
            amplitude_falloff: 0.5, // Each octave has half the amplitude (standard Perlin)
            time: 0.0, // Animation time for 3D noise
            _padding3: 0.0,
        };

        let noise_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Perlin Noise Uniform Buffer"),
            contents: bytemuck::bytes_of(&noise_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout for perlin noise texture generation
        let noise_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Perlin Noise Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let noise_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Perlin Noise Bind Group"),
            layout: &noise_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: noise_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline for perlin noise texture generation
        let noise_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Perlin Noise Pipeline Layout"),
            bind_group_layouts: &[&noise_bind_group_layout],
            push_constant_ranges: &[],
        });

        let noise_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Perlin Noise Texture Pipeline"),
            layout: Some(&noise_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &perlin_noise_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &perlin_noise_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        // Render perlin noise to texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Perlin Noise Texture Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Perlin Noise Texture Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &perlin_noise_texture_view,
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

            render_pass.set_pipeline(&noise_pipeline);
            render_pass.set_bind_group(0, &noise_bind_group, &[]);
            render_pass.draw(0..4, 0..1); // Full-screen quad
        }

        queue.submit(std::iter::once(encoder.finish()));

        // Create sampler for perlin noise texture
        let perlin_noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Perlin Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat, // Repeat for tiling
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let cell_render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Render Bind Group"),
            layout: &cell_render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.points.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cells.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&perlin_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&perlin_noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.link_buffer.as_entire_binding(),
                },
            ],
        });

        let link_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Link Render Bind Group"),
            layout: &link_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.points.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cells.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.link_buffer.as_entire_binding(),
                },
            ],
        });

        let nutrient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Overlay Bind Group"),
            layout: &nutrient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.nutrient_grid.as_entire_binding(),
                },
            ],
        });

        Self {
            cells,
            cell_render_bind_group,
            links,
            link_bind_group,
            nutrient_overlay,
            nutrient_bind_group,
            _nucleus_texture: nucleus_texture,
            _nucleus_sampler: nucleus_sampler,
            perlin_noise_texture: perlin_noise_texture,
            perlin_noise_sampler: perlin_noise_sampler,
            noise_pipeline,
            noise_bind_group,
            noise_uniform_buffer,
            perlin_noise_texture_view,
        }
    }

    pub fn regenerate_noise_texture(&self, device: &wgpu::Device, queue: &wgpu::Queue, time: f32) {
        // Update the time uniform
        let time_uniforms = NoiseTextureUniforms {
            seed: 0.2,
            base_frequency: 4.0,
            octave_count: 6,
            _padding0: 0,
            frequency_falloff: 0.4,
            amplitude_falloff: 0.5,
            time,
            _padding3: 0.0,
        };

        queue.write_buffer(&self.noise_uniform_buffer, 0, bytemuck::bytes_of(&time_uniforms));

        // Regenerate the texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Regenerate Perlin Noise Texture Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Regenerate Perlin Noise Texture Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.perlin_noise_texture_view,
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

            render_pass.set_pipeline(&self.noise_pipeline);
            render_pass.set_bind_group(0, &self.noise_bind_group, &[]);
            render_pass.draw(0..4, 0..1); // Full-screen quad
        }

        queue.submit(std::iter::once(encoder.finish()));
    }
}
