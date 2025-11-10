// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck;

use crate::genetic_algorithm::systems::GeneRegulatoryNetwork;
use crate::genetic_algorithm::genome::Genome;

/// Cell structure for GPU processing
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Cell {
    pub pos: [f32; 2],           // offset 0, size 8
    pub prev_pos: [f32; 2],      // offset 8, size 8
    pub radius: f32,             // offset 16, size 4
    pub energy: f32,             // offset 20, size 4
    pub cell_wall_thickness: f32, // offset 24, size 4
    pub lifeform_idx: u32,       // offset 28, size 4
    pub random_force: [f32; 2],  // offset 40, size 8 (8-byte aligned)
}

impl Cell {
    pub fn new(pos: [f32; 2], radius: f32, lifeform_idx: u32, initial_energy: f32) -> Self {
        Self {
            pos,
            prev_pos: pos,
            radius,
            energy: initial_energy,
            cell_wall_thickness: 0.1,
            lifeform_idx,
            random_force: [0.0, 0.0],
        }
    }
}


/// Lifeform structure for GPU processing
/// Stores metadata about lifeforms for efficient parallel access
#[repr(C)]
pub struct Lifeform {
    pub lifeform_id: u32,
    pub cell_idxs: Vec<u32>,
    pub is_alive: u32,       // 1 if alive, 0 if dead (using u32 for GPU alignment)
    pub genome: Genome,
    pub grn: GeneRegulatoryNetwork,
}

impl Lifeform {
    pub fn new(lifeform_id: u32, cell_idxs: Vec<u32>, genome: Genome, grn: GeneRegulatoryNetwork) -> Self {
        Self {
            lifeform_id,
            cell_idxs,
            is_alive: 1,
            genome,
            grn,
        }
    }
}


