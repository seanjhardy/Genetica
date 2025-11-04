// Effector - output genes that produce effects

use super::genetic_unit::GeneticUnit;

/// Effector type determines what effect is produced
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectorType {
    // Actions
    Divide,
    Die,
    Freeze,

    // Parameters
    Distance,
    Radius,
    Red,
    Green,
    Blue,

    // Proteins
    Chloroplast,
    TouchSensor,
    
    // Marker for array length
    EffectorLength,
}

/// Effector - an output gene that produces an effect
#[derive(Debug, Clone, Copy)]
pub struct Effector {
    pub genetic_unit: GeneticUnit,
    pub effector_type: EffectorType,
}

impl Effector {
    pub fn new(
        effector_type: EffectorType,
        sign: bool,
        modifier: f32,
        embedding: [f32; 3],
    ) -> Self {
        Self {
            genetic_unit: GeneticUnit::new(sign, modifier, embedding),
            effector_type,
        }
    }
}

