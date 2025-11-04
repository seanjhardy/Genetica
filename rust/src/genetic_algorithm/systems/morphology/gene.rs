// Gene - represents a factor in the gene regulatory network

use super::genetic_unit::GeneticUnit;

/// Factor type determines what the gene represents
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorType {
    // External Factors
    Constant,      // A value of 1
    Time,          // A value that increases over time
    Generation,    // The generation of a particular cell
    Energy,        // The energy of the lifeform
    MaternalFactor, // A factor from the genome
    Crowding,      // The number of nearby cells

    // Genetic factors
    InternalProduct, // Morphogen that acts inside the cell
    ExternalProduct, // Morphogen that can interact with other cells
    Receptor,       // Binds to morphogens to control divide direction
}

/// Gene - a genetic unit that represents a factor in the GRN
#[derive(Debug, Clone, Copy)]
pub struct Gene {
    pub genetic_unit: GeneticUnit,
    pub factor_type: FactorType,
    pub extra: [f32; 2], // Extra parameters for some factor types
}

impl Gene {
    pub fn new(
        factor_type: FactorType,
        sign: bool,
        modifier: f32,
        embedding: [f32; 3],
        extra: Option<[f32; 2]>,
    ) -> Self {
        Self {
            genetic_unit: GeneticUnit::new(sign, modifier, embedding),
            factor_type,
            extra: extra.unwrap_or([0.0, 0.0]),
        }
    }
}

