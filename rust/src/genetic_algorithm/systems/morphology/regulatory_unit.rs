// Regulatory unit - a node in the gene regulatory network

/// Regulatory unit - composed of promoters and factors
/// Promoters get activated based on binding distance to other factors,
/// which causes the factors in the regulatory unit to get produced
#[derive(Debug, Clone)]
pub struct RegulatoryUnit {
    /// Indices into the promoters array
    pub promoters: Vec<usize>,
    /// Indices into the factors array
    pub factors: Vec<usize>,
}

impl RegulatoryUnit {
    pub const W: f32 = 10.0;

    pub fn new() -> Self {
        Self {
            promoters: Vec::new(),
            factors: Vec::new(),
        }
    }

    pub fn add_promoter(&mut self, promoter_idx: usize) {
        self.promoters.push(promoter_idx);
    }

    pub fn add_factor(&mut self, factor_idx: usize) {
        self.factors.push(factor_idx);
    }
}

impl Default for RegulatoryUnit {
    fn default() -> Self {
        Self::new()
    }
}

