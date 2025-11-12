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
    pub metadata: u32,
    pub color: [f32; 4],
}

impl Cell {
    pub fn new(pos: [f32; 2], radius: f32, lifeform_slot: u32, initial_energy: f32) -> Self {
        let mut cell = Self {
            pos,
            prev_pos: pos,
            random_force: [0.0, 0.0],
            radius,
            energy: initial_energy,
            cell_wall_thickness: 0.1,
            is_alive: 1,
            lifeform_slot,
            metadata: 0,
            color: [0.0; 4],
        };
        cell.update_color_from_energy();
        cell
    }

    #[inline]
    fn energy_to_color(energy: f32) -> [f32; 4] {
        let energy_normalized = (energy / 100.0).clamp(0.0, 1.0);
        let brightness = 0.1 + energy_normalized * 0.9;
        let r = (1.0 - brightness) * 0.5;
        let g = brightness;
        let b = brightness;
        [r, g, b, 1.0]
    }

    pub fn update_color_from_energy(&mut self) {
        self.color = Self::energy_to_color(self.energy);
    }
}

const _: [(); 64] = [(); std::mem::size_of::<Cell>()];
const _: [(); 16] = [(); std::mem::align_of::<Cell>()];

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

/// Event emitted by GPU about cell lifecycle changes.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CellEvent {
    /// Event kind (see constants below).
    pub kind: u32,
    /// Cell index that emitted the event (usually the parent).
    pub parent_cell_index: u32,
    /// Lifeform slot the parent belongs to.
    pub parent_lifeform_slot: u32,
    /// Misc flags (e.g. adhesion request).
    pub flags: u32,
    pub position: [f32; 2],
    pub radius: f32,
    pub energy: f32,
}

impl CellEvent {
    pub const KIND_DIVISION: u32 = 1;
    pub const KIND_DEATH: u32 = 2;

    pub const FLAG_ADHESIVE: u32 = 1 << 0;
}

const _: [(); 32] = [(); std::mem::size_of::<CellEvent>()];
const _: [(); 16] = [(); std::mem::align_of::<CellEvent>()];

/// Event describing mutations to the link graph that require CPU involvement.
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LinkEvent {
    pub kind: u32,
    pub link_index: u32,
    pub cell_a: u32,
    pub cell_b: u32,
    pub rest_length: f32,
    pub stiffness: f32,
    pub energy_transfer_rate: f32,
    pub _padding: f32,
}

impl LinkEvent {
    pub const KIND_CREATE: u32 = 1;
    pub const KIND_REMOVE: u32 = 2;
}

const _: [(); 32] = [(); std::mem::size_of::<LinkEvent>()];
const _: [(); 16] = [(); std::mem::align_of::<LinkEvent>()];


/// Lifeform structure for GPU processing
/// Stores metadata about lifeforms for efficient parallel access
#[repr(C)]
pub struct Lifeform {
    pub lifeform_id: usize,
    pub species_id: usize,
    pub is_alive: bool,
    pub genome: Genome,
    pub grn: GeneRegulatoryNetwork,
}

impl Lifeform {
    pub fn new(
        lifeform_id: usize,
        species_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
    ) -> Self {
        Self {
            lifeform_id,
            species_id,
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



