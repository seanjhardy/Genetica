// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck::{self};

pub const CELL_WALL_SAMPLES: usize = 20;

pub const MAX_GRN_RECEPTOR_INPUTS: usize = 2;
pub const MAX_GRN_REGULATORY_UNITS: usize = 2;
pub const MAX_GRN_INPUTS_PER_UNIT: usize = 8;
pub const MAX_GRN_STATE_SIZE: usize = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;

pub const MAX_GENES_PER_GENOME: usize = 200;
pub const WORDS_PER_GENE: usize = 4; // 55 base pairs = 110 bits, need 4 u32 words (128 bits)
pub const GENOME_WORD_COUNT: usize = MAX_GENES_PER_GENOME * WORDS_PER_GENE;
pub const MAX_GENOME_EVENTS: usize = 65_536;

pub const MAX_SPECIES_CAPACITY: usize = 1024;

/// Cell structure for GPU processing
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cell {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub random_force: [f32; 2],
    pub radius: f32,
    pub energy: f32,
    pub cell_wall_thickness: f32,
    pub is_alive: u32,
    pub lifeform_slot: u32,
    pub generation: u32,
    pub color: [f32; 4],
    pub link_count: u32,
    pub link_indices: [u32; 6],
    // Perlin noise permutation values for cell wall perturbation (20 values)
    pub noise_permutations: [u32; CELL_WALL_SAMPLES],
    // Organelle positions in unit circle (5 coordinates: nucleus, 3 small white blobs, 1 large dark blob)
    pub organelles: [f32; 10],
    pub angle: f32, // Rotation angle in radians
    // Random offset for sampling perlin noise texture (ensures cell stays within texture bounds)
    pub noise_texture_offset: [f32; 2], // 8 bytes
    // Padding to maintain 16-byte alignment (total size should be multiple of 16)// 12 bytes padding for 16-byte alignment
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenomeEntry {
    pub gene_ids: [u32; MAX_GENES_PER_GENOME],
    pub gene_sequences: [u32; GENOME_WORD_COUNT],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenomeEvent {
    pub dst_genome_slot: u32,
    pub src_genome_slot: u32,
    pub seed: u32,
    pub lifeform_slot: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenomeEventBuffer {
    pub counter: u32,
    pub _pad: u32,
    pub events: [GenomeEvent; MAX_GENOME_EVENTS],
}

impl GenomeEntry {
    pub fn inactive() -> Self {
        Self {
            gene_ids: [0; MAX_GENES_PER_GENOME],
            gene_sequences: [0; GENOME_WORD_COUNT],
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Input {
    pub weight: f32,
    pub index: u32,
    pub promoter_type: u32,
    pub _pad: u32,
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompiledRegulatoryUnit {
    pub input_count: u32,
    pub output_index: u32,
    pub flags: u32,
    pub _padding: u32,
    pub inputs: [Input; MAX_GRN_INPUTS_PER_UNIT],
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrnDescriptor {
    pub receptor_count: u32,
    pub unit_count: u32,
    pub state_stride: u32,
    pub unit_offset: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
    pub _pad3: u32,
}


#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Zeroable)]
pub struct Lifeform {
    pub lifeform_id: u32,
    pub species_slot: u32,
    pub species_id: u32,
    pub gene_count: u32,
    pub rng_state: u32,
    pub first_cell_slot: u32,
    pub cell_count: u32,
    pub grn_descriptor_slot: u32,
    pub grn_unit_offset: u32,
    pub grn_timer: u32,
    pub flags: u32,
    pub _pad: u32,
    pub _pad2: [u32; 2],
    pub grn_state: [f32; MAX_GRN_STATE_SIZE],
}

unsafe impl bytemuck::Pod for Lifeform {}

/// Link that connects two cells together.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Link {
    pub a: u32,
    pub b: u32,
    pub flags: u32,
    pub generation_a: u32,
    pub rest_length: f32,
    pub stiffness: f32,
    pub energy_transfer_rate: f32,
    pub generation_b: u32,
}

impl Link {
    pub const FLAG_ALIVE: u32 = 1 << 0;
    pub const FLAG_ADHESIVE: u32 = 1 << 1;

    pub fn new(a: u32, b: u32, rest_length: f32, stiffness: f32, energy_transfer_rate: f32) -> Self {
        Self {
            a,
            b,
            flags: Self::FLAG_ALIVE,
            generation_a: 0,
            rest_length,
            stiffness,
            energy_transfer_rate,
            generation_b: 0,
        }
    }
}

const _: [(); 32] = [(); std::mem::size_of::<Link>()];
const _: [(); 16] = [(); std::mem::align_of::<Link>()];

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpeciesEntry {
    pub species_id: u32,
    pub mascot_lifeform_slot: u32,
    pub member_count: u32, // Note: atomic on GPU side, but regular u32 in Rust
    pub flags: u32,
    pub mascot_genome: GenomeEntry,
}

impl SpeciesEntry {
    pub const FLAG_ACTIVE: u32 = 1 << 0;

    pub fn inactive() -> Self {
        Self {
            species_id: 0,
            mascot_lifeform_slot: 0,
            member_count: 0,
            flags: 0,
            mascot_genome: GenomeEntry::inactive(),
        }
    }
}

/// Buffer for accumulating position changes from links to cells
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PositionChangeEntry {
    pub delta_x: u32, // Fixed-point: divide by POSITION_CHANGE_SCALE for actual value
    pub delta_y: u32, // Fixed-point: divide by POSITION_CHANGE_SCALE for actual value
    pub num_changes: u32,
    pub _pad: u32,
}

impl PositionChangeEntry {
    pub fn zero() -> Self {
        Self {
            delta_x: 0,
            delta_y: 0,
            num_changes: 0,
            _pad: 0,
        }
    }
}

