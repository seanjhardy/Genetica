// GPU structures for lifeforms and cells - optimized for parallel processing
use bytemuck::{self, Pod, Zeroable};

pub const CELL_WALL_SAMPLES: usize = 20;
pub const MAX_CELL_LINKS: usize = 8;
pub const LINK_CORRECTION_STRIDE: usize = 3;

pub const MAX_GRN_RECEPTOR_INPUTS: usize = 2;
pub const MAX_GRN_REGULATORY_UNITS: usize = 100;
pub const MAX_GRN_INPUTS_PER_UNIT: usize = 8;
pub const MAX_GRN_STATE_SIZE: usize = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;
pub const CELL_INPUT_ARRAY_SIZE: usize = 32; // Fixed size input array per cell

pub const MAX_GENES_PER_GENOME: usize = 200;
pub const WORDS_PER_GENE: usize = 4;
pub const GENOME_WORD_COUNT: usize = MAX_GENES_PER_GENOME * WORDS_PER_GENE;

pub const CREATE_LIFEFORM_FLAG: u32 = 1;
pub const ADD_CELL_TO_LIFEFORM_FLAG: u32 = 2;
pub const REMOVE_CELL_FROM_LIFEFORM_FLAG: u32 = 3;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Point {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub accel: [f32; 2],
    pub angle: f32,
    pub radius: f32,
    pub flags: u32,
    pub _pad0: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Cell {
    pub point_idx: u32,
    pub lifeform_id: u32,
    pub generation: u32,
    pub energy: f32,
    pub cell_wall_thickness: f32,
    pub _pad0: [u32; 3],
    pub color: [f32; 4],
    pub flags: u32,
    pub link_count: u32,
    pub link_indices: [u32; MAX_CELL_LINKS],
    pub noise_permutations: [f32; CELL_WALL_SAMPLES],
    pub noise_texture_offset: [f32; 2],
    //pub inputs: [f32; CELL_INPUT_ARRAY_SIZE],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Link {
    pub a_cell: u32,
    pub a_generation: u32,
    pub b_cell: u32,
    pub b_generation: u32,
    pub angle_from_a: f32,
    pub angle_from_b: f32,
    pub stiffness: f32,
    pub flags: u32,
    pub _pad: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LinkCorrection {
    pub dx: i32,
    pub dy: i32,
    pub d_angle: i32,
}

impl Link {
    pub const FLAG_ALIVE: u32 = 1 << 0;
}
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Species {
    pub species_id: u32,
    pub ancestor_species_id: u32,
    pub member_count: u32,
    pub flags: u32,
    pub mascot_genome: Genome,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Genome {
    pub gene_ids: [u32; MAX_GENES_PER_GENOME],
    pub gene_sequences: [u32; GENOME_WORD_COUNT],
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

pub type CompiledGrn = Vec<CompiledRegulatoryUnit>;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrnDescriptor {
    pub unit_count: u32,
    pub unit_start_index: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Event {
  pub event_type: u32,
  pub parent_lifeform_id: u32,
  pub lifeform_id: u32,
  _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DivisionRequest {
    pub parent_cell_idx: u32,
    pub generation: u32,
    pub energy: f32,
    pub angle: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PickParams {
    pub mouse_pos: [f32; 2],
    pub _pad: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct PickResult {
    pub best_dist: u32,
    pub cell_index: u32,
}

impl PickResult {
    pub fn reset() -> Self {
        Self {
            best_dist: f32::MAX.to_bits(),
            cell_index: u32::MAX,
        }
    }
}
