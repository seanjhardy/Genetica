// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck;

/// GPU-compatible GeneticUnit structure
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGeneticUnit {
    pub sign: u32,           // 0 = false, 1 = true (using u32 for GPU alignment)
    pub modifier: f32,
    pub embedding: [f32; 3],
    pub _padding: f32,       // Padding to align to 16 bytes
}

/// GPU-compatible Gene structure (Factor)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGene {
    pub genetic_unit: GpuGeneticUnit,
    pub factor_type: u32,    // FactorType as u32 enum
    pub extra: [f32; 2],
}

/// GPU-compatible Promoter structure
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPromoter {
    pub genetic_unit: GpuGeneticUnit,
    pub promoter_type: u32,  // PromoterType as u32 enum (0 = Additive, 1 = Multiplicative)
}

/// GPU-compatible Effector structure
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEffector {
    pub genetic_unit: GpuGeneticUnit,
    pub effector_type: u32,  // EffectorType as u32 enum
}

/// GPU-compatible RegulatoryUnit metadata
/// Stores counts and offsets into flat arrays of indices
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRegulatoryUnit {
    pub promoter_count: u32,     // Number of promoters
    pub promoter_offset: u32,    // Offset into promoter_indices array
    pub factor_count: u32,       // Number of factors
    pub factor_offset: u32,      // Offset into factor_indices array
}

/// GPU-compatible GRN metadata per lifeform
/// Stores counts and offsets for all GRN components
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGrnMetadata {
    pub factor_count: u32,
    pub factor_offset: u32,      // Offset into factors array
    pub promoter_count: u32,
    pub promoter_offset: u32,    // Offset into promoters array
    pub effector_count: u32,
    pub effector_offset: u32,   // Offset into effectors array
    pub regulatory_unit_count: u32,
    pub regulatory_unit_offset: u32, // Offset into regulatory_units array
    pub promoter_index_offset: u32,  // Offset into promoter_indices flat array
    pub factor_index_offset: u32,   // Offset into factor_indices flat array
    pub affinity_matrix_offset: u32, // Offset into affinity matrices
    pub affinity_matrix_size: u32,   // Size of affinity matrices (promoters * factors)
}

// Conversion helpers from CPU types to GPU types
impl GpuGeneticUnit {
    pub fn from_cpu(cpu: &crate::genetic_algorithm::systems::morphology::GeneticUnit) -> Self {
        Self {
            sign: if cpu.sign { 1 } else { 0 },
            modifier: cpu.modifier,
            embedding: cpu.embedding,
            _padding: 0.0,
        }
    }
}

impl GpuGene {
    pub fn from_cpu(cpu: &crate::genetic_algorithm::systems::morphology::Gene) -> Self {
        Self {
            genetic_unit: GpuGeneticUnit::from_cpu(&cpu.genetic_unit),
            factor_type: cpu.factor_type as u32,
            extra: cpu.extra,
        }
    }
}

impl GpuPromoter {
    pub fn from_cpu(cpu: &crate::genetic_algorithm::systems::morphology::Promoter) -> Self {
        Self {
            genetic_unit: GpuGeneticUnit::from_cpu(&cpu.genetic_unit),
            promoter_type: cpu.promoter_type as u32,
        }
    }
}

impl GpuEffector {
    pub fn from_cpu(cpu: &crate::genetic_algorithm::systems::morphology::Effector) -> Self {
        Self {
            genetic_unit: GpuGeneticUnit::from_cpu(&cpu.genetic_unit),
            effector_type: cpu.effector_type as u32,
        }
    }
}

/// GPU-compatible Genome structure
/// Genomes are flattened into arrays for GPU storage
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGenome {
    /// Offset into gene_sequences buffer (where this genome's genes start)
    pub sequence_offset: u32,
    /// Number of genes in this genome
    pub gene_count: u32,
    /// Offset into gene_order buffer (where this genome's gene order starts)
    pub order_offset: u32,
    /// Maximum gene length (for bounds checking)
    pub max_gene_length: u32,
}

/// Genome metadata per lifeform
/// Stores offsets into flat genome buffers
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGenomeMetadata {
    /// Genome structure for this lifeform
    pub genome: GpuGenome,
    /// Offset in GRN output buffers where this lifeform's GRN will start
    pub grn_factor_offset: u32,
    pub grn_promoter_offset: u32,
    pub grn_effector_offset: u32,
    pub grn_regulatory_unit_offset: u32,
    pub grn_promoter_index_offset: u32,
    pub grn_factor_index_offset: u32,
    pub grn_affinity_offset: u32,
}

/// Cell structure for GPU processing
/// Stores all cell data needed for parallel updates
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cell {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub energy: f32,
    pub lifeform_idx: u32, // Index into lifeforms array
    pub random_force: [f32; 2], // Random force vector that changes over time
}

impl Cell {
    pub fn new(pos: [f32; 2], lifeform_idx: u32, initial_energy: f32) -> Self {
        Self {
            pos,
            prev_pos: pos,
            energy: initial_energy,
            lifeform_idx,
            random_force: [0.0, 0.0],
        }
    }
}


/// Lifeform structure for GPU processing
/// Stores metadata about lifeforms for efficient parallel access
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Lifeform {
    pub first_cell_idx: u32, // Index of first cell in the cells array
    pub cell_count: u32,     // Number of cells in this lifeform
    pub is_alive: u32,       // 1 if alive, 0 if dead (using u32 for GPU alignment)
    pub _padding: u32,       // Padding to align to 16 bytes
}

impl Lifeform {
    pub fn new(first_cell_idx: u32, cell_count: u32) -> Self {
        Self {
            first_cell_idx,
            cell_count,
            is_alive: 1,
            _padding: 0,
        }
    }
}


