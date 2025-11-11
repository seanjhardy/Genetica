// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck;

use crate::genetic_algorithm::systems::GeneRegulatoryNetwork;
use crate::genetic_algorithm::genome::Genome;

/// Cell structure for GPU processing
#[repr(C, align(16))]
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
    pub _padding: u32,
}

impl Cell {
    pub fn new(pos: [f32; 2], radius: f32, lifeform_slot: u32, initial_energy: f32) -> Self {
        Self {
            pos,
            prev_pos: pos,
            random_force: [0.0, 0.0],
            radius,
            energy: initial_energy,
            cell_wall_thickness: 0.1,
            is_alive: 1,
            lifeform_slot,
            _padding: 0,
        }
    }
}

const _: [(); 48] = [(); std::mem::size_of::<Cell>()];
const _: [(); 16] = [(); std::mem::align_of::<Cell>()];


/// Lifeform structure for GPU processing
/// Stores metadata about lifeforms for efficient parallel access
#[repr(C)]
pub struct Lifeform {
    pub lifeform_id: usize,
    pub species_id: usize,
    pub cell_idxs: Vec<u32>,
    pub is_alive: bool,
    pub genome: Genome,
    pub grn: GeneRegulatoryNetwork,
}

impl Lifeform {
    pub fn new(
        lifeform_id: usize,
        species_id: usize,
        cell_idxs: Vec<u32>,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
    ) -> Self {
        Self {
            lifeform_id,
            species_id,
            cell_idxs,
            is_alive: true,
            genome,
            grn,
        }
    }
}


/// Division request emitted by GPU compute when a cell divides
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DivisionRequest {
    pub parent_lifeform_slot: u32,
    pub cell_index: u32,
    pub pos: [f32; 2],
    pub radius: f32,
    pub energy: f32,
}



