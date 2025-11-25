// GPU pipelines module - manages compute and render pipelines

use wgpu;

use crate::gpu::wgsl::{CELLS_KERNEL, LINKS_KERNEL, NUTRIENTS_KERNEL, SEQUENCE_GRN_KERNEL, CELLS_SHADER, LINKS_SHADER, NUTRIENTS_SHADER};

/// Compute pipelines for physics simulation
pub struct ComputePipelines {
    pub reset_cell_hash: wgpu::ComputePipeline,
    pub build_cell_hash: wgpu::ComputePipeline,
    pub update_cells: wgpu::ComputePipeline,
    pub update_cells_bind_group: wgpu::BindGroup,
    pub update_nutrients: wgpu::ComputePipeline,
    pub update_nutrients_bind_group: wgpu::BindGroup,
    pub update_links: wgpu::ComputePipeline,
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
}

impl RenderPipelines {
    pub fn new(
        device: &wgpu::Device,
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
        }
    }
}

