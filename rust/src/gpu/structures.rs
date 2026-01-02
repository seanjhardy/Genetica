// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck::{self, Pod, Zeroable};

pub const CELL_WALL_SAMPLES: usize = 20;

pub const MAX_GRN_RECEPTOR_INPUTS: usize = 2;
pub const MAX_GRN_REGULATORY_UNITS: usize = 2;
pub const MAX_GRN_INPUTS_PER_UNIT: usize = 8;
pub const MAX_GRN_STATE_SIZE: usize = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;

pub const MAX_GENES_PER_GENOME: usize = 200;
pub const WORDS_PER_GENE: usize = 4;
pub const GENOME_WORD_COUNT: usize = MAX_GENES_PER_GENOME * WORDS_PER_GENE;
pub const MAX_GENOME_EVENTS: usize = 65_536;

pub const MAX_SPECIES_CAPACITY: usize = 1024;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct VerletPoint {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub accel: [f32; 2],
    pub angle: f32,
    pub radius: f32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Cell {
    pub point_idx: u32,
    pub lifeform_id: u32,
    pub generation: u32,
    pub energy: f32,
    pub cell_wall_thickness: f32,
    pub noise_permutations: [u32; CELL_WALL_SAMPLES],
    pub noise_texture_offset: [f32; 2],
    pub color: [f32; 4],
    pub flags: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Link {
    pub a_cell: u32,
    pub a_generation: u32,
    pub b_cell: u32,
    pub b_generation: u32,
    pub rest_length: f32,
    pub stiffness: f32,
    pub flags: u32,
}

impl Link {
    pub const FLAG_ALIVE: u32 = 1 << 0;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Lifeform {
    pub lifeform_id: u32,
    pub species_id: u32,
    pub first_cell: u32,
    pub cell_count: u32,
    pub flags: u32,
    pub _pad: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SpeciesEntry {
    pub species_id: u32,
    pub ancestor_species_id: u32,
    pub member_count: u32,
    pub flags: u32,
    pub mascot_genome: GenomeEntry,
}

impl SpeciesEntry {
    pub fn inactive() -> Self {
        Self {
            species_id: 0,
            ancestor_species_id: 0,
            member_count: 0,
            flags: 0,
            mascot_genome: GenomeEntry::inactive(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GenomeEntry {
    pub gene_ids: [u32; MAX_GENES_PER_GENOME],
    pub gene_sequences: [u32; GENOME_WORD_COUNT],
}

impl GenomeEntry {
    pub fn inactive() -> Self {
        Self {
            gene_ids: [0; MAX_GENES_PER_GENOME],
            gene_sequences: [0; GENOME_WORD_COUNT],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GenomeEvent {
    pub dst_genome_slot: u32,
    pub src_genome_slot: u32,
    pub seed: u32,
    pub lifeform_slot: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GenomeEventBuffer {
    pub counter: u32,
    pub _pad: u32,
    pub events: [GenomeEvent; MAX_GENOME_EVENTS],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PositionChangeEntry {
    pub delta_x: i32,
    pub delta_y: i32,
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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SpawnRequest {
    pub pos: [f32; 2],
    pub radius: f32,
    pub energy: f32,
    pub lifeform_id: u32,
    pub parent_cell: u32,
    pub _pad: u32,
}


#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Input {
    pub weight: f32,
    pub index: u32,
    pub promoter_type: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompiledRegulatoryUnit {
    pub input_count: u32,
    pub output_index: u32,
    pub flags: u32,
    pub _padding: u32,
    pub inputs: [Input; MAX_GRN_INPUTS_PER_UNIT],
}

#[repr(C)]
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


