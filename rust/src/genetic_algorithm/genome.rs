// Genome module - stores genetic information as base pairs

use std::collections::HashMap;

/// Base pair type (0, 1, 2, or 3)
pub type BasePair = u8;

/// Genome stores genetic information as a collection of genes
/// Each gene is a string of base pairs (0-3)
pub struct Genome {
    /// Map of gene ID to base pair sequence
    pub hox_genes: HashMap<usize, String>,
    /// Order of genes in the genome
    pub hox_gene_order: Vec<usize>,
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
        
        for i in 0..num_genes {
            let mut gene = String::with_capacity(gene_length);
            for _ in 0..gene_length {
                let base = (rng.gen::<u8>() % 4) as u8;
                gene.push(char::from(b'0' + base));
            }
            
            let gene_id = i;
            genome.hox_genes.insert(gene_id, gene);
            genome.hox_gene_order.push(gene_id);
        }
        
        genome
    }

    /// Add a gene to the genome
    pub fn add_hox_gene(&mut self, gene_id: usize, sequence: String, position: Option<usize>) {
        self.hox_genes.insert(gene_id, sequence);
        if let Some(pos) = position {
            self.hox_gene_order.insert(pos, gene_id);
        } else {
            self.hox_gene_order.push(gene_id);
        }
    }

    /// Get a gene sequence by ID
    pub fn get_gene(&self, gene_id: usize) -> Option<&String> {
        self.hox_genes.get(&gene_id)
    }

    /// Get all genes
    pub fn get_genes(&self) -> &HashMap<usize, String> {
        &self.hox_genes
    }

    /// Check if genome contains a gene
    pub fn contains(&self, gene_id: usize) -> bool {
        self.hox_genes.contains_key(&gene_id)
    }
}

impl Default for Genome {
    fn default() -> Self {
        Self::new()
    }
}

