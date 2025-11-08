// Genome module - stores genetic information as base pairs

use std::collections::HashMap;

use puffin::profile_scope;
use rand::Rng;
use crate::genetic_algorithm::GeneticAlgorithm;
use crate::utils::{genetic_algorithm::gene_similarity_score};

/// Base pair type (0, 1, 2, or 3)
pub type BasePair = u8;

/// Genome stores genetic information as a collection of genes
/// Each gene is a string of base pairs (0-3)
pub struct Genome {
    /// Map of gene ID to base pair sequence
    pub hox_genes: HashMap<u32, Vec<BasePair>>,
    /// Order of genes in the genome
    pub hox_gene_order: Vec<u32>,
}

impl Genome {
    pub const INITIAL_HOX_SIZE: usize = 20;

    /// Create a new empty genome
    pub fn new() -> Self {
        Self {
            hox_genes: HashMap::new(),
            hox_gene_order: Vec::new(),
        }
    }

    /// Initialize a random genome
    pub fn init_random<R: rand::Rng>(rng: &mut R, num_genes: usize, gene_length: usize) -> Self {
        let mut genome = Self::new();
        
        for _ in 0..num_genes {
            let gene = (0..gene_length)
                .map(|_| rng.gen::<u8>() % 4)
                .collect::<Vec<BasePair>>();
            let gene_id = GeneticAlgorithm::next_gene_id();
            genome.hox_genes.insert(gene_id, gene);
            genome.hox_gene_order.push(gene_id);
        }
        
        genome
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();

        let mut new_genes: HashMap<u32, Vec<BasePair>> = HashMap::new();
        let mut new_gene_order: Vec<u32> = Vec::new();
        let mut inserted_genes: Vec<u32> = Vec::new();

        // Traverse the genome in order and mutate
        for gene_id in self.hox_gene_order.iter() {
            let hox_gene = self.hox_genes.get(gene_id).unwrap();
            if rng.gen::<f32>() < GeneticAlgorithm::CLONE_GENE_CHANCE {
                new_genes.insert(*gene_id,hox_gene.clone());
                new_gene_order.push(*gene_id);
            }

            if rng.gen::<f32>() < GeneticAlgorithm::INSERT_GENE_CHANCE {
                let new_gene_id = GeneticAlgorithm::next_gene_id();
                let new_gene = (0..Genome::INITIAL_HOX_SIZE)
                    .map(|_| rng.gen::<u8>() % 4)
                    .collect::<Vec<BasePair>>();
                new_genes.insert(new_gene_id, new_gene);
                inserted_genes.push(new_gene_id);
            }

            let mut gene_copy: Vec<BasePair> = Vec::new();
            // Mutate gene
            for &base_pair in hox_gene.iter() {
                let mut base: BasePair = base_pair;
                if rng.gen::<f32>() < GeneticAlgorithm::MUTATE_BASE_CHANCE {
                    base = rng.gen::<u8>() % 4;
                }
                
                if rng.gen::<f32>() < GeneticAlgorithm::INSERT_BASE_CHANCE {
                    gene_copy.push(rng.gen::<u8>() % 4);
                }

                if rng.gen::<f32>() > GeneticAlgorithm::DELETE_BASE_CHANCE {
                    gene_copy.push(base);
                }
            }

            // If we're NOT deleting, then add this gene to the new genome
            if rng.gen::<f32>() > GeneticAlgorithm::DELETE_GENE_CHANCE {
                new_genes.insert(*gene_id, gene_copy);
                new_gene_order.push(*gene_id);
            }
        }

        // Add inserted genes randomly throughout the genome
        for gene_id in inserted_genes {
            let random_position = rng.gen_range(0..=new_gene_order.len());
            new_gene_order.insert(random_position, gene_id);
        }

        self.hox_genes = new_genes;
        self.hox_gene_order = new_gene_order;
    }

    pub fn crossover(&self, other: &Genome) -> Genome {
        let mut new_genes: HashMap<u32, Vec<BasePair>> = HashMap::new();
        let mut new_gene_order: Vec<u32> = Vec::new();

        // Crossover genes from both genomes
        for gene_id in self.hox_gene_order.iter() {
            let hox_gene = self.hox_genes.get(gene_id).unwrap();
            if other.hox_genes.contains_key(gene_id) {
                let other_gene = other.hox_genes.get(gene_id).unwrap();
                new_genes.insert(*gene_id, self.crossover_gene(hox_gene, other_gene));
                new_gene_order.push(*gene_id);
            } else {
                new_genes.insert(*gene_id, hox_gene.clone());
                new_gene_order.push(*gene_id);
            }
        }
        
        for (idx, gene_id) in other.hox_gene_order.iter().enumerate() {
            let hox_gene = other.hox_genes.get(gene_id).unwrap();
            if !self.hox_genes.contains_key(gene_id) {
                new_genes.insert(*gene_id, hox_gene.clone());
                // Insert gene where the parent gene is
                new_gene_order.insert(idx, *gene_id);
            }
        }

        let mut new_genome = Genome::new();
        new_genome.hox_genes = new_genes;
        new_genome.hox_gene_order = new_gene_order;
        new_genome
    }
    
    pub fn crossover_gene(&self, gene1: &Vec<BasePair>, gene2: &Vec<BasePair>) -> Vec<BasePair> {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut new_gene: Vec<BasePair> = Vec::new();

        let crossover = rng.gen::<f32>() < GeneticAlgorithm::CROSSOVER_GENE_CHANCE;
        let parent = rng.gen::<bool>();

        let gene_length = gene1.len().max(gene2.len());

        for i in 0..gene_length {
            let base1 = gene1.get(i).copied();
            let base2 = gene2.get(i).copied();

            let c = if crossover {
                match (base1, base2) {
                    (Some(b1), Some(b2)) => if rng.gen::<bool>() { b1 } else { b2 },
                    (Some(b1), None) => b1,
                    (None, Some(b2)) => b2,
                    (None, None) => continue,
                }
            } else {
                match if parent { base1 } else { base2 } {
                    Some(b) => b,
                    None => continue,
                }
            };

            new_gene.push(c);
        }

        new_gene
    }

    pub fn compare(&self, other: &Genome) -> f32 {
        profile_scope!("Compare Genomes");
        let mut gene_difference: u32 = 0;
        let mut base_difference: f32 = 0.0f32;

        for gene_id in self.hox_gene_order.iter() {
            let gene = self.hox_genes.get(gene_id).unwrap();
            if !other.hox_genes.contains_key(gene_id) {
                gene_difference += 1;
            } else {
                let other = other.hox_genes.get(gene_id).unwrap();
                base_difference += (gene_similarity_score(gene, other) * 100.0) as f32;
            }
        }

        for gene_id in other.hox_gene_order.iter() {
            if !self.hox_genes.contains_key(gene_id) {
                gene_difference += 1;
            }
        }

        (gene_difference as f32) * GeneticAlgorithm::GENE_DIFFERENCE_SCALAR + base_difference * GeneticAlgorithm::BASE_DIFFERENCE_SCALAR
    }
}

impl Default for Genome {
    fn default() -> Self {
        Self::new()
    }
}

