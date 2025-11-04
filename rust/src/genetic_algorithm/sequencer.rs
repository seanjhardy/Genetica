// Sequencer - reads genome and builds gene regulatory network

use crate::genetic_algorithm::genome::Genome;
use crate::genetic_algorithm::utils::{read_base, read_base_range, read_unique_base_range};
use crate::genetic_algorithm::systems::morphology::{
    Gene, FactorType, Promoter, PromoterType, Effector, EffectorType, 
    RegulatoryUnit, GeneRegulatoryNetwork,
};

/// Sequence a genome to build a gene regulatory network
/// Returns the GRN even if some genes fail to parse
pub fn sequence_grn(genome: &Genome) -> GeneRegulatoryNetwork {
    let mut grn = GeneRegulatoryNetwork::new();
    
    let mut factors: Vec<Gene> = Vec::new();
    let mut promoters: Vec<Promoter> = Vec::new();
    let mut effectors: Vec<Effector> = Vec::new();
    let mut regulatory_units: Vec<RegulatoryUnit> = Vec::new();
    
    let mut regulatory_unit = RegulatoryUnit::new();
    let mut regulatory_promoters: Vec<usize> = Vec::new();
    let mut regulatory_factors: Vec<usize> = Vec::new();
    let mut reading_promoters = true;
    
    // Process each gene in the genome
    for (_gene_id, sequence) in genome.hox_gene_order.iter()
        .filter_map(|id| genome.hox_genes.get(id).map(|seq| (*id, seq.clone())))
    {
        let mut rna = sequence.clone();
        
        // Process genes until RNA is exhausted or we can't read more
        while rna.len() >= 20 { // Minimum required bases for a gene (2 + 1 + 1 + 8 + 8*3 = 20)
            // Read gene type (2 bases -> 0-16, mod 5 -> 0-4)
            let type_val = match read_base_range(&mut rna, 2) {
                Ok(val) => (val * 16.0) as usize % 5,
                Err(_) => break,
            };
            let sign = match read_base(&mut rna) {
                Ok(b) => b >= 2,
                Err(_) => break,
            };
            let active = match read_base(&mut rna) {
                Ok(b) => b >= 1,
                Err(_) => break,
            };
            
            if !active {
                continue;
            }
            
            // Read modifier (8 bases)
            let modifier = match read_unique_base_range(&mut rna, 8) {
                Ok(m) => m,
                Err(_) => break,
            };
            
            // Read embedding (8 bases each for x, y, z)
            let embedding = match (
                read_unique_base_range(&mut rna, 8),
                read_unique_base_range(&mut rna, 8),
                read_unique_base_range(&mut rna, 8),
            ) {
                (Ok(x), Ok(y), Ok(z)) => [x, y, z],
                _ => break,
            };
            
            // Process based on type
            match type_val {
                0 => {
                    // External factors
                    let external_factor_types = [
                        FactorType::MaternalFactor,
                        FactorType::Crowding,
                        FactorType::Constant,
                        FactorType::Generation,
                        FactorType::Energy,
                        FactorType::Time,
                    ];
                    let sub_type = match read_unique_base_range(&mut rna, 3) {
                        Ok(val) => ((val * 64.0) as usize) % 6,
                        Err(_) => break,
                    };
                    let factor_type = external_factor_types[sub_type];
                    
                    let extra = match (
                        read_unique_base_range(&mut rna, 8),
                        read_unique_base_range(&mut rna, 8),
                    ) {
                        (Ok(x), Ok(y)) => [x, y],
                        _ => break,
                    };
                    
                    let gene = Gene::new(factor_type, sign, modifier, embedding, Some(extra));
                    factors.push(gene);
                }
                1 => {
                    // Effectors
                    let effector_types = [
                        EffectorType::Divide, EffectorType::Die, EffectorType::Freeze,
                        EffectorType::Distance, EffectorType::Radius,
                        EffectorType::Red, EffectorType::Green, EffectorType::Blue,
                        EffectorType::Chloroplast, EffectorType::TouchSensor,
                        EffectorType::EffectorLength,
                    ];
                    let sub_type = match read_unique_base_range(&mut rna, 4) {
                        Ok(val) => ((val * 256.0) as usize) % effector_types.len(),
                        Err(_) => break,
                    };
                    let effector_type = effector_types[sub_type];
                    
                    // Skip if this effector type already exists
                    if effectors.iter().any(|e| e.effector_type == effector_type) {
                        continue;
                    }
                    
                    let effector = Effector::new(effector_type, sign, modifier, embedding);
                    effectors.push(effector);
                }
                2 => {
                    // Promoters
                    let additive = match read_base(&mut rna) {
                        Ok(b) => b >= 1,
                        Err(_) => break,
                    };
                    let promoter_type = if additive {
                        PromoterType::Additive
                    } else {
                        PromoterType::Multiplicative
                    };
                    
                    let promoter = Promoter::new(promoter_type, sign, modifier, embedding);
                    
                    if !reading_promoters {
                        // Finish current regulatory unit
                        regulatory_unit.promoters = regulatory_promoters.clone();
                        regulatory_unit.factors = regulatory_factors.clone();
                        regulatory_units.push(regulatory_unit);
                        regulatory_unit = RegulatoryUnit::new();
                        regulatory_promoters.clear();
                        regulatory_factors.clear();
                        reading_promoters = true;
                    }
                    
                    regulatory_promoters.push(promoters.len());
                    promoters.push(promoter);
                }
                3 | 4 | 5 => {
                    // Genes: internal product, external product, receptor
                    let gene_types = [
                        FactorType::ExternalProduct,
                        FactorType::InternalProduct,
                        FactorType::Receptor,
                    ];
                    
                    if regulatory_promoters.is_empty() {
                        break;
                    }
                    
                    let factor_type = gene_types[type_val - 3];
                    let gene = Gene::new(factor_type, sign, modifier, embedding, None);
                    
                    reading_promoters = false;
                    regulatory_factors.push(factors.len());
                    factors.push(gene);
                }
                _ => {}
            }
        }
    }
    
    // Add final regulatory unit if it exists
    if !regulatory_promoters.is_empty() {
        regulatory_unit.promoters = regulatory_promoters;
        regulatory_unit.factors = regulatory_factors;
        regulatory_units.push(regulatory_unit);
    }
    
    grn.factors = factors;
    grn.promoters = promoters;
    grn.effectors = effectors;
    grn.regulatory_units = regulatory_units;
    grn.calculate_affinities();
    
    grn
}

