// GPU pipelines module - manages compute and render pipelines

use wgpu;
use wgpu::util::DeviceExt;
use std::path::PathBuf;

use crate::gpu::wgsl::{CELLS_KERNEL, LINKS_KERNEL, NUTRIENTS_KERNEL, SEQUENCE_GRN_KERNEL, GENOME_EVENTS_KERNEL, CELLS_SHADER, LINKS_SHADER, NUTRIENTS_SHADER, PERLIN_NOISE_TEXTURE_SHADER};

/// Compute pipelines for physics simulation
pub struct ComputePipelines {
    pub reset_cell_hash: wgpu::ComputePipeline,
    pub build_cell_hash: wgpu::ComputePipeline,
    pub update_cells: wgpu::ComputePipeline,
    pub update_cells_bind_group: wgpu::BindGroup,
    pub update_nutrients: wgpu::ComputePipeline,
    pub update_nutrients_bind_group: wgpu::BindGroup,
    pub update_links: wgpu::ComputePipeline,
    pub process_genome_events: wgpu::ComputePipeline,
    pub sequence_grn: wgpu::ComputePipeline,
    pub sequence_grn_bind_group: wgpu::BindGroup,
}

impl ComputePipelines {
    pub fn new(
        device: &wgpu::Device,
        cell_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        cell_free_list_buffer: &wgpu::Buffer,
        cell_counter_buffer: &wgpu::Buffer,
        spawn_buffer: &wgpu::Buffer,
        nutrient_grid_buffer: &wgpu::Buffer,
        link_buffer: &wgpu::Buffer,
        link_free_list_buffer: &wgpu::Buffer,
        spatial_hash_bucket_heads_buffer: &wgpu::Buffer,
        spatial_hash_next_indices_buffer: &wgpu::Buffer,
        grn_descriptor_buffer: &wgpu::Buffer,
        grn_units_buffer: &wgpu::Buffer,
        lifeforms_buffer: &wgpu::Buffer,
        lifeform_free_buffer: &wgpu::Buffer,
        next_lifeform_id_buffer: &wgpu::Buffer,
        genome_buffer: &wgpu::Buffer,
        species_entries_buffer: &wgpu::Buffer,
        species_free_buffer: &wgpu::Buffer,
        next_species_id_buffer: &wgpu::Buffer,
        next_gene_id_buffer: &wgpu::Buffer,
        lifeform_counter_buffer: &wgpu::Buffer,
        species_counter_buffer: &wgpu::Buffer,
        position_changes_buffer: &wgpu::Buffer,
        genome_event_buffer: &wgpu::Buffer,
    ) -> Self {
        // Create shader module
        let cells_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cells Shader"),
            source: CELLS_KERNEL.clone(),
        });

        // Create bind group layout
        // Bindings must match cells.wgsl and links.wgsl shaders exactly:
        // 0: uniform - uniforms (Uniforms)
        // 1: storage, read_write - cells (array<Cell>)
        // 2: storage, read_write - cell_free_list (CellFreeList)
        // 3: storage, read_write - cell_counter (Counter)
        // 4: storage, read_write - spawn_buffer (SpawnBuffer)
        // 5: storage, read_write - nutrient_grid (NutrientGrid)
        // 6: storage, read_write - links (array<Link>)
        // 7: storage, read_write - link_free_list (FreeList)
        // 8: storage, read_write - cell_bucket_heads (array<atomic<i32>>)
        // 11: storage, read_write - cell_hash_next (array<i32>)
        // 12: storage, read_write - grn_descriptors (array<GrnDescriptor>)
                // 13: storage, read_write - grn_units (array<CompiledRegulatoryUnit>) [READ-WRITE for Sequence GRN]
        // 14: storage, read_write - lifeforms (array<Lifeform>)
        // 15: storage, read_write - lifeform_free (FreeList)
        // 16: storage, read_write - next_lifeform_id (Counter)
        // 17: storage, read_write - genomes (array<GenomeEntry>)
        // 18: storage, read_write - species_entries (array<SpeciesEntry>)
        // 19: storage, read_write - species_free (FreeList)
        // 20: storage, read_write - next_species_id (Counter)
        // 21: storage, read_write - next_gene_id (Counter)
        // 22: storage, read_write - lifeform_counter (Counter)
        // 23: storage, read_write - species_counter (Counter)
        // 24: storage, read_write - position_changes (array<PositionChangeEntry>)
        // 25: storage, read_write - genome_events (GenomeEventBuffer)
        let cells_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cells Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 19,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 22,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 23,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 24,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 25,
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

        let reset_cell_hash = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Reset Cell Hash Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &cells_shader,
            entry_point: Some("reset_bucket_heads"),
            compilation_options: Default::default(),
            cache: None,
        });

        let build_cell_hash = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Build Cell Hash Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &cells_shader,
            entry_point: Some("build_spatial_hash"),
            compilation_options: Default::default(),
            cache: None,
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
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let genome_events_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Genome Events Shader"),
            source: GENOME_EVENTS_KERNEL.clone(),
        });

        let process_genome_events = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Process Genome Events Pipeline"),
            layout: Some(&cells_pipeline_layout),
            module: &genome_events_shader,
            entry_point: Some("process_genome_events"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_cells_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cells Bind Group"),
            layout: &cells_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_free_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spawn_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: nutrient_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: link_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: link_free_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: spatial_hash_bucket_heads_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: spatial_hash_next_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: grn_descriptor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: grn_units_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: lifeforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: lifeform_free_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: next_lifeform_id_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: genome_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: species_entries_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: species_free_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 20,
                    resource: next_species_id_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 21,
                    resource: next_gene_id_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 22,
                    resource: lifeform_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 23,
                    resource: species_counter_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 24,
                    resource: position_changes_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 25,
                    resource: genome_event_buffer.as_entire_binding(),
                },
            ],
        });

        let nutrient_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nutrient Regeneration Shader"),
            source: NUTRIENTS_KERNEL.clone(),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Nutrient Compute Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nutrient Compute Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let update_nutrients = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Nutrient Regenerate Pipeline"),
            layout: Some(&pipeline_layout),
            module: &nutrient_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_nutrients_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Compute Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nutrient_grid_buffer.as_entire_binding(),
                },
            ],
        });

        // Sequence GRN pipeline - sequences genomes and compiles GRNs
        let sequence_grn_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sequence GRN Shader"),
            source: SEQUENCE_GRN_KERNEL.clone(),
        });

        // Create dedicated bind group layout for Sequence GRN pipeline
        // Only includes the bindings actually used: 0, 12, 13, 14, 17
        let sequence_grn_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sequence GRN Bind Group Layout"),
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
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
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

        let sequence_grn_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sequence GRN Pipeline Layout"),
            bind_group_layouts: &[&sequence_grn_bind_group_layout],
            push_constant_ranges: &[],
        });

        let sequence_grn = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sequence GRN Pipeline"),
            layout: Some(&sequence_grn_pipeline_layout),
            module: &sequence_grn_shader,
            entry_point: Some("sequence_and_compile"),
            compilation_options: Default::default(),
            cache: None,
        });

        let sequence_grn_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sequence GRN Bind Group"),
            layout: &sequence_grn_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: grn_descriptor_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: grn_units_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: lifeforms_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: genome_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            reset_cell_hash,
            build_cell_hash,
            update_cells,
            update_cells_bind_group,
            update_nutrients,
            update_nutrients_bind_group,
            update_links,
            process_genome_events,
            sequence_grn,
            sequence_grn_bind_group,
        }
    }
}

/// Render pipelines for drawing points
pub struct RenderPipelines {
    pub points: wgpu::RenderPipeline,
    pub cell_render_bind_group: wgpu::BindGroup,
    pub links: wgpu::RenderPipeline,
    pub link_bind_group: wgpu::BindGroup,
    pub nutrient_overlay: wgpu::RenderPipeline,
    pub nutrient_bind_group: wgpu::BindGroup,
    // Keep texture and sampler alive
    _nucleus_texture: wgpu::Texture,
    _nucleus_sampler: wgpu::Sampler,
    _perlin_noise_texture: wgpu::Texture,
    _perlin_noise_sampler: wgpu::Sampler,
}

impl RenderPipelines {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_config: &wgpu::SurfaceConfiguration,
        cell_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        cell_free_list_buffer: &wgpu::Buffer,
        link_buffer: &wgpu::Buffer,
        nutrient_buffer: &wgpu::Buffer,
        _spatial_hash_bucket_heads_buffer: &wgpu::Buffer,
        spatial_hash_bucket_heads_readonly_buffer: &wgpu::Buffer,
        spatial_hash_next_indices_buffer: &wgpu::Buffer,
    ) -> Self {
        let cell_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cells Shader"),
            source: CELLS_SHADER.clone(),
        });

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
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
                    binding: 6,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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

        let points = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        // Generate perlin noise texture (200x200, single channel stored in RGBA)
        const NOISE_TEXTURE_SIZE: u32 = 200;
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
        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct NoiseTextureUniforms {
            seed: f32,
            base_frequency: f32,
            octave_count: u32,
            _padding0: u32,
            frequency_falloff: f32,
            amplitude_falloff: f32,
            _padding1: f32,
            _padding2: f32,
        }

        let noise_uniforms = NoiseTextureUniforms {
            seed: 0.5, // Default seed
            base_frequency: 0.8, // Base frequency for first octave
            octave_count: 6, // Number of octaves for fractal noise
            _padding0: 0,
            frequency_falloff: 0.2, // Each octave has half the frequency (standard Perlin)
            amplitude_falloff: 0.1, // Each octave has half the amplitude (standard Perlin)
            _padding1: 0.0,
            _padding2: 0.0,
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
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_free_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&nucleus_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&nucleus_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: link_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: spatial_hash_bucket_heads_readonly_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: spatial_hash_next_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&perlin_noise_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&perlin_noise_sampler),
                },
            ],
        });

        let link_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Link Render Bind Group"),
            layout: &link_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: link_buffer.as_entire_binding(),
                },
            ],
        });

        let nutrient_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nutrient Overlay Bind Group"),
            layout: &nutrient_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: nutrient_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            points,
            cell_render_bind_group,
            links,
            link_bind_group,
            nutrient_overlay,
            nutrient_bind_group,
            _nucleus_texture: nucleus_texture,
            _nucleus_sampler: nucleus_sampler,
            _perlin_noise_texture: perlin_noise_texture,
            _perlin_noise_sampler: perlin_noise_sampler,
        }
    }
}

