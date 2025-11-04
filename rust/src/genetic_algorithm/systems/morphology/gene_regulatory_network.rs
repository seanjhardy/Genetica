// Gene regulatory network - network of genes that activate and deactivate one another

use super::{Gene, Promoter, Effector, RegulatoryUnit};

/// Gene Regulatory Network (GRN)
/// A network where:
/// - Factors represent levels of proteins in the cell (inputs)
/// - Effectors produce effects (outputs)
/// - Promoters bind to factors and promote production
/// - Regulatory units combine promoters and produce factors
#[derive(Clone)]
pub struct GeneRegulatoryNetwork {
    /// Genes which represent levels of proteins
    pub factors: Vec<Gene>,
    /// Promoters that bind to factors
    pub promoters: Vec<Promoter>,
    /// Output genes that produce effects
    pub effectors: Vec<Effector>,
    /// Regulatory units (nodes in the network)
    pub regulatory_units: Vec<RegulatoryUnit>,
    
    /// Affinities between promoters and factors (flat matrix)
    pub promoter_factor_affinities: Vec<f32>,
    /// Affinities between factors and effectors (flat matrix)
    pub factor_effector_affinities: Vec<f32>,
    /// Affinities between factors and receptors (flat matrix)
    pub factor_receptor_affinities: Vec<f32>,
    
    /// Distances between each pair of cells (triangular matrix)
    pub cell_distances: Vec<f32>,
}

impl GeneRegulatoryNetwork {
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            promoters: Vec::new(),
            effectors: Vec::new(),
            regulatory_units: Vec::new(),
            promoter_factor_affinities: Vec::new(),
            factor_effector_affinities: Vec::new(),
            factor_receptor_affinities: Vec::new(),
            cell_distances: Vec::new(),
        }
    }

    pub fn calculate_affinities(&mut self) {
        // Calculate promoter-factor affinities based on embedding distances
        self.promoter_factor_affinities.clear();
        
        for promoter in &self.promoters {
            for factor in &self.factors {
                let dist = self.embedding_distance(promoter.genetic_unit.embedding, factor.genetic_unit.embedding);
                let affinity = 1.0 / (1.0 + dist);
                self.promoter_factor_affinities.push(affinity);
            }
        }
    }

    fn embedding_distance(&self, a: [f32; 3], b: [f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl Default for GeneRegulatoryNetwork {
    fn default() -> Self {
        Self::new()
    }
}

