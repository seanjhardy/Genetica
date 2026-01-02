use std::sync::atomic::{AtomicU32, Ordering};
use crate::genetic_algorithm::systems::GeneRegulatoryNetwork;
use crate::genetic_algorithm::Genome;
use crate::genetic_algorithm::systems::morphology::compile_grn::CompiledRegulatoryUnit;

const IS_ALIVE_FLAG: u32 = 1 << 0;

pub struct Lifeform {
    pub birth_time: usize,
    pub lifeform_id: usize,
    pub species_id: usize,
    pub first_cell: u32,
    pub flags: u32,
    pub genome: Genome,
    pub grn: GeneRegulatoryNetwork,
    pub compiled_grn: Vec<CompiledRegulatoryUnit>,
    pub cell_count: AtomicU32,
}

impl Lifeform {
    pub fn new(
        birth_time: usize,
        lifeform_id: usize,
        species_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
        compiled_grn: Vec<CompiledRegulatoryUnit>,
    ) -> Self {
        Self {
            birth_time,
            lifeform_id,
            species_id,
            first_cell: 0,
            flags: 0,
            genome,
            grn,
            compiled_grn,
            cell_count: AtomicU32::new(0),
        }
    }

    pub fn is_alive(&self) -> bool {
        self.flags & IS_ALIVE_FLAG == 1
    }

    pub fn compiled_grn(&self) -> &[CompiledRegulatoryUnit] {
        &self.compiled_grn
    }


    pub fn cell_count(&self) -> u32 {
        self.cell_count.load(Ordering::Relaxed)
    }

    pub fn increment_cell_count(&self) {
        self.cell_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn decrement_cell_count(&self) -> bool {
        let old_count = self.cell_count.fetch_sub(1, Ordering::Relaxed);
        old_count > 0
    }

    pub fn has_cells(&self) -> bool {
        self.cell_count.load(Ordering::Relaxed) > 0
    }
}
