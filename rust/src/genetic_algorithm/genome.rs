// Genome module - stores genetic information as base pairs

use std::collections::{HashMap, HashSet};

use puffin::profile_scope;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use rand_distr::Poisson;
use smallvec::SmallVec;
use crate::genetic_algorithm::GeneticAlgorithm;

/// Base pair type (0, 1, 2, or 3)
pub type BasePair = u8;

const INLINE_GENE_CAPACITY: usize = 64;
type GeneSequence = SmallVec<[BasePair; INLINE_GENE_CAPACITY]>;

/// Genome stores genetic information as a collection of genes
/// Each gene is a string of base pairs (0-3)
#[derive(Clone)]
pub struct Genome {
    /// Map of gene ID to base pair sequence
    pub hox_genes: HashMap<usize, GeneSequence>,
    /// Order of genes in the genome
    pub hox_gene_order: Vec<usize>,
}

impl Genome {
    #[allow(dead_code)]
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
        genome.hox_genes.reserve(num_genes);
        genome.hox_gene_order.reserve(num_genes);
        
        for _ in 0..num_genes {
            let gene = Self::generate_random_gene(rng, gene_length);
            let gene_id = GeneticAlgorithm::next_gene_id();
            genome.hox_genes.insert(gene_id, gene);
            genome.hox_gene_order.push(gene_id);
        }
        
        genome
    }

    #[allow(dead_code)]
    pub fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        self.mutate_with_rng(&mut rng);
    }

    pub fn mutate_with_rng<R: Rng>(&mut self, rng: &mut R) {
        profile_scope!("Genome Mutate");
        let clone_gene_threshold =
            Self::probability_to_threshold(GeneticAlgorithm::CLONE_GENE_CHANCE);
        let insert_gene_threshold =
            Self::probability_to_threshold(GeneticAlgorithm::INSERT_GENE_CHANCE);
        let delete_gene_threshold =
            Self::probability_to_threshold(GeneticAlgorithm::DELETE_GENE_CHANCE);

        let mut index = 0usize;
        while index < self.hox_gene_order.len() {
            profile_scope!("Mutate Gene");
            let gene_id = self.hox_gene_order[index];

            let mut cloned_gene: Option<GeneSequence> = None;
            let mut inserted_gene: Option<GeneSequence> = None;
            let delete_gene = {
                let gene_entry = self
                    .hox_genes
                    .get_mut(&gene_id)
                    .expect("gene id missing during mutation");

                if rng.gen::<u32>() < clone_gene_threshold {
                    cloned_gene = Some(gene_entry.clone());
                }
                if rng.gen::<u32>() < insert_gene_threshold {
                    inserted_gene = Some(Self::generate_random_gene(rng, Self::INITIAL_HOX_SIZE));
                }

                profile_scope!("Apply Base Mutations");
                let original_len = gene_entry.len();

                if original_len > 0 {
                    let mutate_avg =
                        GeneticAlgorithm::MUTATE_BASE_CHANCE as f64 * original_len as f64;
                    let delete_avg =
                        GeneticAlgorithm::DELETE_BASE_CHANCE as f64 * original_len as f64;

                    let mutation_count =
                        Self::sample_poisson(mutate_avg, rng).min(original_len as u64) as usize;
                    if mutation_count > 0 {
                        profile_scope!("Apply Mutations");
                        let mutation_positions =
                            Self::sample_unique_positions(mutation_count, gene_entry.len(), rng);
                        for idx in mutation_positions {
                            profile_scope!("Mutate Base");
                            gene_entry[idx] = Self::random_base(rng);
                        }
                    }

                    let deletion_count =
                        Self::sample_poisson(delete_avg, rng).min(gene_entry.len() as u64)
                            as usize;
                    if deletion_count > 0 {
                        profile_scope!("Apply Deletions");
                        let mut deletion_positions =
                            Self::sample_unique_positions(deletion_count, gene_entry.len(), rng);
                        deletion_positions.sort_unstable_by(|a, b| b.cmp(a));
                        for idx in deletion_positions {
                            profile_scope!("Delete Base");
                            gene_entry.remove(idx);
                        }
                    }
                }

                let insert_candidates = gene_entry.len() + 1;
                let insert_avg =
                    GeneticAlgorithm::INSERT_BASE_CHANCE as f64 * insert_candidates as f64;
                let insertion_count = Self::sample_poisson(insert_avg, rng) as usize;
                if insertion_count > 0 {
                    profile_scope!("Apply Insertions");
                    let mut insert_positions =
                        Self::sample_positions(insertion_count, insert_candidates, rng);
                    insert_positions.sort_unstable();
                    for (offset, position) in insert_positions.into_iter().enumerate() {
                        profile_scope!("Insert Base");
                        let value = Self::random_base(rng);
                        gene_entry.insert(position + offset, value);
                    }
                }

                let delete_gene =
                    gene_entry.is_empty() || rng.gen::<u32>() < delete_gene_threshold;

                delete_gene
            };

            if delete_gene {
                profile_scope!("Delete Gene");
                self.hox_genes.remove(&gene_id);
                self.hox_gene_order.remove(index);
                continue;
            }

            if let Some(mut clone_seq) = cloned_gene {
                profile_scope!("Insert Cloned Gene");
                let new_id = GeneticAlgorithm::next_gene_id();
                if clone_seq.is_empty() {
                    clone_seq.push(rng.gen::<u8>() & 0b11);
                }
                self.hox_genes.insert(new_id, clone_seq);
                let insert_pos = rng.gen_range(0..=self.hox_gene_order.len());
                self.hox_gene_order.insert(insert_pos, new_id);
            }

            if let Some(mut new_gene_seq) = inserted_gene {
                profile_scope!("Insert New Gene");
                if new_gene_seq.is_empty() {
                    new_gene_seq.push(rng.gen::<u8>() & 0b11);
                }
                let new_id = GeneticAlgorithm::next_gene_id();
                self.hox_genes.insert(new_id, new_gene_seq);
                let insert_pos = rng.gen_range(0..=self.hox_gene_order.len());
                self.hox_gene_order.insert(insert_pos, new_id);
            }

            index += 1;
        }
    }

    #[allow(dead_code)]
    pub fn crossover(&self, other: &Genome) -> Genome {
        let mut new_genes: HashMap<usize, GeneSequence> = HashMap::new();
        let mut new_gene_order: Vec<usize> = Vec::new();

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
    
    #[allow(dead_code)]
    pub fn crossover_gene(&self, gene1: &GeneSequence, gene2: &GeneSequence) -> GeneSequence {
        use rand::Rng;

        let mut rng = rand::thread_rng();
        let mut new_gene: GeneSequence = GeneSequence::with_capacity(gene1.len().max(gene2.len()));

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

    #[allow(dead_code)]
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
                base_difference += self.compare_genes(gene, other);
            }
        }

        for gene_id in other.hox_gene_order.iter() {
            if !self.hox_genes.contains_key(gene_id) {
                gene_difference += 1;
            }
        }

        (gene_difference as f32) * GeneticAlgorithm::GENE_DIFFERENCE_SCALAR + base_difference * GeneticAlgorithm::BASE_DIFFERENCE_SCALAR
    }

    pub fn compare_genes(&self, gene1: &GeneSequence, gene2: &GeneSequence) -> f32 {
        let min_len = gene1.len().min(gene2.len());
        let mut diff = gene1.iter().zip(gene2).filter(|(x, y)| x != y).count();
        diff += gene1.len().max(gene2.len()) - min_len;
        diff as f32
    }

    #[inline]
    fn probability_to_threshold(probability: f32) -> u32 {
        let clamped = probability.clamp(0.0, 1.0);
        (clamped * (u32::MAX as f32)) as u32
    }

    fn generate_random_gene<R: Rng>(rng: &mut R, length: usize) -> GeneSequence {
        let capacity = length.max(Self::INITIAL_HOX_SIZE);
        let mut gene = GeneSequence::with_capacity(capacity);
        if length == 0 {
            return gene;
        }

        if length <= INLINE_GENE_CAPACITY {
            let mut buffer = SmallVec::<[u8; INLINE_GENE_CAPACITY]>::with_capacity(length);
            buffer.resize(length, 0);
            rng.fill_bytes(&mut buffer);
            gene.extend(buffer.into_iter().map(|byte| byte & 0b11));
        } else {
            let mut buffer = vec![0u8; length];
            rng.fill_bytes(&mut buffer);
            gene.extend(buffer.into_iter().map(|byte| byte & 0b11));
        }

        gene
    }

    #[inline]
    fn random_base<R: Rng>(rng: &mut R) -> BasePair {
        rng.gen::<u8>() & 0b11
    }

    fn sample_unique_positions<R: Rng>(
        count: usize,
        len: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        if count >= len {
            return (0..len).collect();
        }
        let mut positions = HashSet::with_capacity(count);
        let distribution = Uniform::from(0..len);
        while positions.len() < count {
            positions.insert(distribution.sample(rng));
        }
        positions.into_iter().collect()
    }

    fn sample_positions<R: Rng>(
        count: usize,
        upper_bound: usize,
        rng: &mut R,
    ) -> Vec<usize> {
        if upper_bound == 0 || count == 0 {
            return Vec::new();
        }
        let distribution = Uniform::from(0..upper_bound);
        (0..count).map(|_| distribution.sample(rng)).collect()
    }

    fn sample_poisson<R: Rng>(lambda: f64, rng: &mut R) -> u64 {
        if lambda <= 0.0 {
            0
        } else if lambda < 1e-6 {
            if rng.gen::<f64>() < lambda {
                1
            } else {
                0
            }
        } else {
            let dist = Poisson::new(lambda).unwrap_or_else(|_| Poisson::new(1e-6).unwrap());
            dist.sample(rng) as u64
        }
    }
}

impl Default for Genome {
    fn default() -> Self {
        Self::new()
    }
}

