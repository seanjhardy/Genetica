// GPU structures for lifeforms and cells - optimized for parallel processing

use bytemuck::{self, Zeroable};

pub const MAX_GRN_RECEPTOR_INPUTS: usize = 2;
pub const MAX_GRN_REGULATORY_UNITS: usize = 2;
pub const MAX_GRN_INPUTS_PER_UNIT: usize = 8;
pub const MAX_GRN_STATE_SIZE: usize = MAX_GRN_RECEPTOR_INPUTS + MAX_GRN_REGULATORY_UNITS;
pub const GRN_EVALUATION_INTERVAL: u32 = 8;

pub const MAX_GENES_PER_GENOME: usize = 20;
pub const BASE_PAIRS_PER_GENE: usize = 20;
pub const BASE_PAIRS_PER_GENOME: usize = MAX_GENES_PER_GENOME * BASE_PAIRS_PER_GENE;
pub const GENOME_WORD_COUNT: usize = (BASE_PAIRS_PER_GENOME + 3) / 4;

pub const MAX_SPECIES_CAPACITY: usize = 1024;
pub const MAX_LIFEFORM_EVENTS: usize = 1024;
pub const MAX_SPECIES_EVENTS: usize = 256;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GenomeEntry {
    pub gene_count: u32,
    pub base_pairs: [u32; GENOME_WORD_COUNT],
}

impl GenomeEntry {
    pub fn inactive() -> Self {
        Self {
            gene_count: 0,
            base_pairs: [0; GENOME_WORD_COUNT],
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
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            },
            units: Vec::new(),
        }
    }
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

impl Lifeform {
    pub const FLAG_ACTIVE: u32 = 1 << 0;
    pub const FLAG_PRESERVED: u32 = 1 << 1;

    pub fn inactive() -> Self {
        Self::zeroed()
    }

    pub fn from_descriptor(
        lifeform_id: u32,
        grn_descriptor_slot: u32,
        descriptor: &GrnDescriptor,
    ) -> Self {
        let mut lifeform = Self::zeroed();
        lifeform.lifeform_id = lifeform_id;
        lifeform.species_slot = 0;
        lifeform.species_id = 0;
        lifeform.gene_count = 0;
        lifeform.rng_state = 0;
        lifeform.first_cell_slot = 0;
        lifeform.cell_count = 0;
        lifeform.grn_descriptor_slot = grn_descriptor_slot;
        lifeform.grn_unit_offset = descriptor.unit_offset;
        lifeform.grn_timer = 0;
        lifeform.flags = Self::FLAG_ACTIVE;
        lifeform._pad = 0;
        lifeform._pad2 = [0; 2];
        lifeform
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


#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpeciesEntry {
    pub species_id: u32,
    pub mascot_lifeform_slot: u32,
    pub member_count: u32,
    pub flags: u32,
}

impl SpeciesEntry {
    pub const FLAG_ACTIVE: u32 = 1 << 0;

    pub fn inactive() -> Self {
        Self {
            species_id: 0,
            mascot_lifeform_slot: 0,
            member_count: 0,
            flags: 0,
        }
    }
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LifeformEvent {
    pub kind: u32,
    pub lifeform_id: u32,
    pub species_id: u32,
    pub lifeform_slot: u32,
}

impl LifeformEvent {
    pub const KIND_CREATE: u32 = 1;
    pub const KIND_DESTROY: u32 = 2;
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpeciesEvent {
    pub kind: u32,
    pub species_id: u32,
    pub species_slot: u32,
    pub member_count: u32,
}

impl SpeciesEvent {
    pub const KIND_CREATE: u32 = 1;
    pub const KIND_EXTINCT: u32 = 2;
}
