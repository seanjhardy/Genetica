// Genetic unit - base structure for all genetic elements

/// Genetic unit - base structure for all genetic elements in the GRN
/// Contains common properties: sign, modifier, and embedding vector
#[derive(Debug, Clone, Copy)]
pub struct GeneticUnit {
    pub sign: bool,        // Whether this unit is positive or negative
    pub modifier: f32,     // Modifier value
    pub embedding: [f32; 3], // 3D embedding vector for affinity calculations
}

impl GeneticUnit {
    pub const DISTANCE_THRESHOLD: f32 = 0.5;

    pub fn new(sign: bool, modifier: f32, embedding: [f32; 3]) -> Self {
        Self {
            sign,
            modifier,
            embedding,
        }
    }
}

