// Sequencer - reads genome and builds gene regulatory network

use std::collections::VecDeque;

use crate::genetic_algorithm::genome::Genome;
use crate::genetic_algorithm::systems::{GeneRegulatoryNetwork, Receptor, Factor, Promoter, Effector,
     RegulatoryUnit, ReceptorType, FactorType, PromoterType, EffectorType, EMBEDDING_DIMENSIONS};
use crate::genetic_algorithm::utils::{read_base, read_base_range, read_unique_base_range};

/// Sequence a genome to build a gene regulatory network
/// Returns the GRN even if some genes fail to parse
pub fn sequence_grn(genome: &Genome) -> GeneRegulatoryNetwork {
    let mut grn = GeneRegulatoryNetwork::new();
    // Inputs
    let mut receptors: Vec<Receptor> = Vec::new();
    // hidden units
    let mut regulatory_units: Vec<RegulatoryUnit> = Vec::new();
    // Outputs
    let mut effectors: Vec<Effector> = Vec::new();
    
    let mut regulatory_unit = RegulatoryUnit::new();
    let mut regulatory_promoters: Vec<Promoter> = Vec::new();
    let mut regulatory_factors: Vec<Factor> = Vec::new();
    let mut reading_promoters = true;
    
    // Process each gene in the genome
    for (_gene_id, sequence) in genome
        .hox_gene_order
        .iter()
        .filter_map(|id| genome.hox_genes.get(id).map(|seq| (*id, seq.clone())))
    {
        let mut rna: VecDeque<u8> = VecDeque::from(sequence.clone());
        
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
            
            // Read embedding (8 bases each for x, y, z ... of EMBEDDING_DIMENSIONS)
            let mut embedding: [f32; EMBEDDING_DIMENSIONS] = [0.0f32; EMBEDDING_DIMENSIONS];
            for i in 0..EMBEDDING_DIMENSIONS {
                embedding[i] = match read_unique_base_range(&mut rna, 8) {
                    Ok(m) => m,
                    Err(_) => break,
                };
            }
            
            // Process based on type
            match type_val {
                0 => {
                    // Receptor
                    let receptor_types = [
                        ReceptorType::MaternalFactor,
                        ReceptorType::Crowding,
                        ReceptorType::Constant,
                        ReceptorType::Generation,
                        ReceptorType::Energy,
                        ReceptorType::Time,
                    ];
                    let sub_type = match read_unique_base_range(&mut rna, 3) {
                        Ok(val) => ((val * 64.0) as usize) % 6,
                        Err(_) => break,
                    };
                    let receptor_type: ReceptorType = receptor_types[sub_type];
                    
                    let extra = match (
                        read_unique_base_range(&mut rna, 8),
                        read_unique_base_range(&mut rna, 8),
                    ) {
                        (Ok(x), Ok(y)) => [x, y],
                        _ => break,
                    };
                    
                    let receptor = Receptor::new(receptor_type, sign, modifier, embedding, extra);
                    receptors.push(receptor);
                }
                1 => {
                    // Effectors
                    let effector_types = [
                        EffectorType::Divide, EffectorType::Die, EffectorType::Freeze,
                        EffectorType::Distance, EffectorType::Radius,
                        EffectorType::Red, EffectorType::Green, EffectorType::Blue,
                    ];
                    let sub_type = match read_unique_base_range(&mut rna, 4) {
                        Ok(val) => ((val * 256.0) as usize) % effector_types.len(),
                        Err(_) => break,
                    };
                    let effector_type = effector_types[sub_type];

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
                    
                    regulatory_promoters.push(promoter);
                }
                3 | 4 | 5 => {
                    // Genes: internal product, external product, receptor
                    let gene_types = [
                        FactorType::ExternalMorphogen,
                        FactorType::InternalMorphogen,
                        FactorType::Orientant,
                    ];
                    
                    if regulatory_promoters.is_empty() {
                        break;
                    }
                    
                    let factor_type = gene_types[type_val - 3];
                    let gene = Factor::new(factor_type, sign, modifier, embedding);
                    
                    reading_promoters = false;
                    regulatory_factors.push(gene);
                }
                _ => {}
            }
        }
    }
    
    // Add final regulatory unit if it exists
    if !regulatory_promoters.is_empty() && !regulatory_factors.is_empty() {
        regulatory_unit.promoters = regulatory_promoters;
        regulatory_unit.factors = regulatory_factors;
        regulatory_units.push(regulatory_unit);
    }
    
    grn.receptors = receptors;
    grn.effectors = effectors;
    grn.regulatory_units = regulatory_units;
    grn
}

