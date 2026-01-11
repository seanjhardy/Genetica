// Sequencer - reads genome and builds gene regulatory network

use std::mem;

use puffin::profile_scope;

use crate::genetic_algorithm::genome::Genome;
use crate::genetic_algorithm::systems::{GeneRegulatoryNetwork, Receptor, Factor, Promoter, Effector,
     RegulatoryUnit, ReceptorType, FactorType, PromoterType, EffectorType, EMBEDDING_DIMENSIONS};

const MIN_GENE_BASES: usize = 20;
const RECEPTOR_TYPES: [ReceptorType; 6] = [
    ReceptorType::MaternalFactor,
    ReceptorType::Crowding,
    ReceptorType::Constant,
    ReceptorType::Generation,
    ReceptorType::Energy,
    ReceptorType::Time,
];
const EFFECTOR_TYPES: [EffectorType; 8] = [
    EffectorType::Divide,
    EffectorType::Die,
    EffectorType::Freeze,
    EffectorType::Distance,
    EffectorType::Radius,
    EffectorType::Red,
    EffectorType::Green,
    EffectorType::Blue,
];
const FACTOR_TYPES: [FactorType; 3] = [
    FactorType::ExternalMorphogen,
    FactorType::InternalMorphogen,
    FactorType::Orientant,
];

struct SequenceReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> SequenceReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    #[inline]
    fn read_base(&mut self) -> Option<u8> {
        let value = *self.data.get(self.position)?;
        self.position += 1;
        Some(value)
    }

    fn read_base_range(&mut self, length: usize) -> Option<f32> {
        let mut result = 0.0f32;
        for _ in 0..length {
            result += self.read_base()? as f32;
        }
        Some(result / (3.0 * length as f32))
    }

    fn read_unique_base_range(&mut self, length: usize) -> Option<f32> {
        let mut result = 0.0f32;
        let mut weight = 0.25f32;
        for _ in 0..length {
            let base = self.read_base()? as f32;
            result += base * weight;
            weight *= 0.25f32;
        }
        Some(result)
    }
}

/// Sequence a genome to build a gene regulatory network
/// Returns the GRN even if some genes fail to parse
pub fn sequence_grn(genome: &Genome) -> GeneRegulatoryNetwork {
    profile_scope!("Sequence GRN");
    let mut grn = GeneRegulatoryNetwork::new();
    let gene_count = genome.hox_gene_order.len();
    // Inputs
    let mut receptors: Vec<Receptor> = Vec::with_capacity(gene_count);
    // hidden units
    let mut regulatory_units: Vec<RegulatoryUnit> = Vec::with_capacity(gene_count);
    // Outputs
    let mut effectors: Vec<Effector> = Vec::with_capacity(gene_count);
    
    let mut regulatory_unit = RegulatoryUnit::new();
    let mut regulatory_promoters: Vec<Promoter> = Vec::new();
    let mut regulatory_factors: Vec<Factor> = Vec::new();
    let mut reading_promoters = true;
    
    // Process each gene in the genome
    for gene_id in genome.hox_gene_order.iter() {
        let sequence = match genome.hox_genes.get(gene_id) {
            Some(seq) => seq.as_slice(),
            None => continue,
        };

        let mut reader = SequenceReader::new(sequence);

        while reader.remaining() >= MIN_GENE_BASES {
            // Read gene type (2 bases -> 0-16, mod 5 -> 0-4)
            let type_val = match reader.read_base_range(2) {
                Some(val) => (val * 16.0) as usize % 5,
                None => break,
            };
            let sign = match reader.read_base() {
                Some(b) => b >= 2,
                None => break,
            };
            let active = match reader.read_base() {
                Some(b) => b >= 1,
                None => break,
            };
            
            if !active {
                continue;
            }
            
            // Read modifier (8 bases)
            let modifier = match reader.read_unique_base_range(8) {
                Some(m) => m,
                None => break,
            };
            
            // Read embedding (8 bases each for x, y, z ... of EMBEDDING_DIMENSIONS)
            let mut embedding: [f32; EMBEDDING_DIMENSIONS] = [0.0f32; EMBEDDING_DIMENSIONS];
            let mut embedding_valid = true;
            for value in embedding.iter_mut() {
                match reader.read_unique_base_range(8) {
                    Some(m) => *value = m,
                    None => {
                        embedding_valid = false;
                        break;
                    }
                }
            }
            if !embedding_valid {
                break;
            }
            
            // Process based on type
            match type_val {
                0 => {
                    // Receptor
                    let sub_type = match reader.read_unique_base_range(3) {
                        Some(val) => ((val * 64.0) as usize) % 6,
                        None => break,
                    };
                    let receptor_type: ReceptorType = RECEPTOR_TYPES[sub_type];
                    
                    let extra = match (
                        reader.read_unique_base_range(8),
                        reader.read_unique_base_range(8),
                    ) {
                        (Some(x), Some(y)) => [x, y],
                        _ => break,
                    };
                    
                    let receptor = Receptor::new(receptor_type, sign, modifier, embedding, extra);
                    receptors.push(receptor);
                }
                1 => {
                    // Effectors
                    let sub_type = match reader.read_unique_base_range(4) {
                        Some(val) => ((val * 256.0) as usize) % EFFECTOR_TYPES.len(),
                        None => break,
                    };
                    let effector_type = EFFECTOR_TYPES[sub_type];

                    let effector = Effector::new(effector_type, sign, modifier, embedding);
                    effectors.push(effector);
                }
                2 => {
                    // Promoters
                    let additive = match reader.read_base() {
                        Some(b) => b >= 1,
                        None => break,
                    };
                    let promoter_type = if additive {
                        PromoterType::Additive
                    } else {
                        PromoterType::Multiplicative
                    };
                    
                    let promoter = Promoter::new(promoter_type, sign, modifier, embedding);
                    
                    if !reading_promoters {
                        // Finish current regulatory unit
                        regulatory_unit.promoters = mem::take(&mut regulatory_promoters);
                        regulatory_unit.factors = mem::take(&mut regulatory_factors);
                        regulatory_units.push(regulatory_unit);
                        regulatory_unit = RegulatoryUnit::new();
                        reading_promoters = true;
                    }
                    
                    regulatory_promoters.push(promoter);
                }
                3 | 4 | 5 => {
                    // Genes: internal product, external product, receptor
                    if regulatory_promoters.is_empty() {
                        break;
                    }
                    
                    let factor_type = FACTOR_TYPES[type_val - 3];
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
        regulatory_unit.promoters = mem::take(&mut regulatory_promoters);
        regulatory_unit.factors = mem::take(&mut regulatory_factors);
        regulatory_units.push(regulatory_unit);
    }
    
    grn.receptors = receptors;
    grn.effectors = effectors;
    grn.regulatory_units = regulatory_units;
    grn
}
