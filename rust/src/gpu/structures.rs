// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck::{self, Zeroable};

use crate::genetic_algorithm::systems::GeneRegulatoryNetwork;
use crate::genetic_algorithm::genome::Genome;

pub const MAX_GRN_RECEPTOR_INPUTS: usize = 16;
pub const MAX_GRN_REGULATORY_UNITS: usize = 16;
pub const MAX_GRN_INPUTS_PER_UNIT: usize = 8;
pub const MAX_GRN_STATE_SIZE: usize = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;
pub const GRN_EVALUATION_INTERVAL: u32 = 8;

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
    pub grn_receptor_count: u32,
    pub grn_unit_count: u32,
    pub grn_timer: u32,
    pub _grn_padding: u32,
    pub grn_state: [f32; MAX_GRN_STATE_SIZE],
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
            grn_receptor_count: 0,
            grn_unit_count: 0,
            grn_timer: 0,
            _grn_padding: 0,
            grn_state: [0.0; MAX_GRN_STATE_SIZE],
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

    pub fn reset_grn_state(&mut self) {
        self.grn_receptor_count = 0;
        self.grn_unit_count = 0;
        self.grn_timer = 0;
        self.grn_state.fill(0.0);
    }
}

const _: [(); 208] = [(); std::mem::size_of::<Cell>()];
const _: [(); 16] = [(); std::mem::align_of::<Cell>()];

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
    pub evaluation_interval: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

#[derive(Clone, Debug)]
pub struct CompiledGrn {
    pub descriptor: GrnDescriptor,
    pub units: Vec<CompiledRegulatoryUnit>,
}

impl CompiledGrn {
    pub fn empty() -> Self {
        Self {
            descriptor: GrnDescriptor {
                receptor_count: 0,
                unit_count: 0,
                state_stride: 0,
                unit_offset: 0,
                evaluation_interval: GRN_EVALUATION_INTERVAL,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            units: Vec::new(),
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LifeformState {
    pub lifeform_id: u32,
    pub first_cell_slot: u32,
    pub cell_count: u32,
    pub grn_descriptor_slot: u32,
    pub grn_unit_offset: u32,
    pub grn_unit_count: u32,
    pub flags: u32,
    pub _pad: u32,
}

impl LifeformState {
    pub const FLAG_ACTIVE: u32 = 1;

    pub fn inactive() -> Self {
        Self::zeroed()
    }

    pub fn from_descriptor(
        lifeform_id: u32,
        grn_descriptor_slot: u32,
        descriptor: &GrnDescriptor,
    ) -> Self {
        Self {
            lifeform_id,
            first_cell_slot: 0,
            cell_count: 0,
            grn_descriptor_slot,
            grn_unit_offset: descriptor.unit_offset,
            grn_unit_count: descriptor.unit_count,
            flags: Self::FLAG_ACTIVE,
            _pad: 0,
        }
    }
}

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
    pub compiled_grn: CompiledGrn,
}

impl Lifeform {
    pub fn new(
        lifeform_id: usize,
        species_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
        compiled_grn: CompiledGrn,
    ) -> Self {
        Self {
            lifeform_id,
            species_id,
            is_alive: true,
            genome,
            grn,
            compiled_grn,
        }
    }

    pub fn compiled_grn(&self) -> &CompiledGrn {
        &self.compiled_grn
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



