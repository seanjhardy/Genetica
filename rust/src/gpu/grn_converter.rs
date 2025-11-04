// GRN converter - converts CPU GeneRegulatoryNetwork to GPU format

use crate::gpu::structures::{
    GpuGene, GpuPromoter, GpuEffector, GpuRegulatoryUnit, GpuGrnMetadata,
};
use crate::genetic_algorithm::systems::morphology::GeneRegulatoryNetwork;

/// Convert CPU GRNs to GPU format with all constituent parts flattened
/// Returns all components needed to populate GPU buffers
pub struct GrnGpuData {
    pub factors: Vec<GpuGene>,
    pub promoters: Vec<GpuPromoter>,
    pub effectors: Vec<GpuEffector>,
    pub regulatory_units: Vec<GpuRegulatoryUnit>,
    pub promoter_indices: Vec<u32>,  // Flattened promoter indices for regulatory units
    pub factor_indices: Vec<u32>,    // Flattened factor indices for regulatory units
    pub promoter_factor_affinities: Vec<f32>,
    pub factor_effector_affinities: Vec<f32>,
    pub factor_receptor_affinities: Vec<f32>,
    pub grn_metadata: Vec<GpuGrnMetadata>,  // One per lifeform
}

impl GrnGpuData {
    /// Convert a list of CPU GRNs to GPU format
    /// All constituent parts are flattened into separate arrays
    pub fn from_cpu_grns(grns: &[GeneRegulatoryNetwork]) -> Self {
        let mut factors = Vec::new();
        let mut promoters = Vec::new();
        let mut effectors = Vec::new();
        let mut regulatory_units = Vec::new();
        let mut promoter_indices = Vec::new();
        let mut factor_indices = Vec::new();
        let mut promoter_factor_affinities = Vec::new();
        let mut factor_effector_affinities = Vec::new();
        let mut factor_receptor_affinities = Vec::new();
        let mut grn_metadata = Vec::new();
        
        let mut global_factor_offset = 0u32;
        let mut global_promoter_offset = 0u32;
        let mut global_effector_offset = 0u32;
        let mut global_regulatory_unit_offset = 0u32;
        let mut global_affinity_offset = 0u32;
        
        for grn in grns {
            // Calculate affinities if not already done
            // Clone the GRN to get mutable access for calculating affinities
            let mut grn_mut = (*grn).clone();
            if grn.promoter_factor_affinities.is_empty() && !grn.promoters.is_empty() && !grn.factors.is_empty() {
                grn_mut.calculate_affinities();
            }
            
            let factor_offset = global_factor_offset;
            let promoter_offset = global_promoter_offset;
            let effector_offset = global_effector_offset;
            let regulatory_unit_offset = global_regulatory_unit_offset;
            let promoter_index_offset = promoter_indices.len() as u32;
            let factor_index_offset = factor_indices.len() as u32;
            let affinity_offset = global_affinity_offset;
            
            // Convert factors
            for factor in &grn_mut.factors {
                factors.push(GpuGene::from_cpu(factor));
            }
            
            // Convert promoters
            for promoter in &grn_mut.promoters {
                promoters.push(GpuPromoter::from_cpu(promoter));
            }
            
            // Convert effectors
            for effector in &grn_mut.effectors {
                effectors.push(GpuEffector::from_cpu(effector));
            }
            
            // Convert regulatory units (flatten Vec<usize> indices)
            for reg_unit in &grn_mut.regulatory_units {
                let promoter_count = reg_unit.promoters.len() as u32;
                let factor_count = reg_unit.factors.len() as u32;
                
                // Store promoter indices (remember start offset before pushing)
                let promoter_idx_start = promoter_indices.len() as u32;
                for &promoter_idx in &reg_unit.promoters {
                    promoter_indices.push((promoter_offset + promoter_idx as u32) as u32);
                }
                
                // Store factor indices (remember start offset before pushing)
                let factor_idx_start = factor_indices.len() as u32;
                for &factor_idx in &reg_unit.factors {
                    factor_indices.push((factor_offset + factor_idx as u32) as u32);
                }
                
                regulatory_units.push(GpuRegulatoryUnit {
                    promoter_count,
                    promoter_offset: promoter_idx_start,
                    factor_count,
                    factor_offset: factor_idx_start,
                });
            }
            
            // Store affinity matrices
            let affinity_size = grn_mut.promoters.len() * grn_mut.factors.len();
            promoter_factor_affinities.extend_from_slice(&grn_mut.promoter_factor_affinities);
            
            // Pad if affinity matrix is incomplete
            while promoter_factor_affinities.len() < affinity_offset as usize + affinity_size {
                promoter_factor_affinities.push(0.0);
            }
            
            // Factor-effector affinities (if not empty)
            if !grn_mut.factor_effector_affinities.is_empty() {
                factor_effector_affinities.extend_from_slice(&grn_mut.factor_effector_affinities);
            }
            
            // Factor-receptor affinities (if not empty)
            if !grn_mut.factor_receptor_affinities.is_empty() {
                factor_receptor_affinities.extend_from_slice(&grn_mut.factor_receptor_affinities);
            }
            
            // Get final index offsets after processing all regulatory units
            let final_promoter_index_offset = promoter_indices.len() as u32;
            let final_factor_index_offset = factor_indices.len() as u32;
            
            // Create GRN metadata for this lifeform
            grn_metadata.push(GpuGrnMetadata {
                factor_count: grn_mut.factors.len() as u32,
                factor_offset,
                promoter_count: grn_mut.promoters.len() as u32,
                promoter_offset,
                effector_count: grn_mut.effectors.len() as u32,
                effector_offset,
                regulatory_unit_count: grn_mut.regulatory_units.len() as u32,
                regulatory_unit_offset,
                promoter_index_offset,
                factor_index_offset,
                affinity_matrix_offset: affinity_offset,
                affinity_matrix_size: affinity_size as u32,
            });
            
            // Update global offsets
            global_factor_offset += grn_mut.factors.len() as u32;
            global_promoter_offset += grn_mut.promoters.len() as u32;
            global_effector_offset += grn_mut.effectors.len() as u32;
            global_regulatory_unit_offset += grn_mut.regulatory_units.len() as u32;
            global_affinity_offset += affinity_size as u32;
        }
        
        Self {
            factors,
            promoters,
            effectors,
            regulatory_units,
            promoter_indices,
            factor_indices,
            promoter_factor_affinities,
            factor_effector_affinities,
            factor_receptor_affinities,
            grn_metadata,
        }
    }
}

