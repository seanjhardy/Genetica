// Promoter - promotes the level of other elements

use super::genetic_unit::GeneticUnit;

/// Promoter type determines how promoters combine
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromoterType {
    Additive,      // Add promoter activities
    Multiplicative, // Multiply promoter activities
}

/// Promoter - an element that promotes the level of other elements
#[derive(Debug, Clone, Copy)]
pub struct Promoter {
    pub genetic_unit: GeneticUnit,
    pub promoter_type: PromoterType,
}

impl Promoter {
    pub fn new(
        promoter_type: PromoterType,
        sign: bool,
        modifier: f32,
        embedding: [f32; 3],
    ) -> Self {
        Self {
            genetic_unit: GeneticUnit::new(sign, modifier, embedding),
            promoter_type,
        }
    }
}

