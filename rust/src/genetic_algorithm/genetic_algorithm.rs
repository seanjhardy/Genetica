use std::sync::atomic::{AtomicU32, Ordering};

// Genetic algorithm module - manages genomes, lifeforms, and gene regulatory networks
use crate::gpu::structures::{Cell, Lifeform};
use crate::utils::math::Rect;
use crate::genetic_algorithm::{Genome, Species, sequence_grn};

static GENE_ID: AtomicU32 = AtomicU32::new(0);
static LIFEFORM_ID: AtomicU32 = AtomicU32::new(0);
static SPECIES_ID: AtomicU32 = AtomicU32::new(0);
pub struct GeneticAlgorithm {
    //lifeform_id -> (genome, grn)
    lifeforms: Vec<Lifeform>,
    // Other data
    species: Vec<Species>,
}

impl GeneticAlgorithm {
    pub const GENE_DIFFERENCE_SCALAR: f32 = 0.5;
    pub const BASE_DIFFERENCE_SCALAR: f32 = 0.1;
    pub const COMPATABILITY_DISTANCE_THRESHOLD: f32 = 5.0;

    pub const INSERT_GENE_CHANCE: f32 = 0.0005;
    pub const CLONE_GENE_CHANCE: f32 = 0.0001;
    pub const DELETE_GENE_CHANCE: f32 = 0.0005;
    pub const CROSSOVER_GENE_CHANCE: f32 = 0.2;

    pub const MUTATE_BASE_CHANCE: f32 = 0.0005;
    pub const INSERT_BASE_CHANCE: f32 = 0.00003;
    pub const DELETE_BASE_CHANCE: f32 = 0.00005;

    pub fn new() -> Self {
        Self {
            lifeforms: Vec::new(),
            species: Vec::new(),
        }
    }

    pub fn next_gene_id() -> u32 {
        GENE_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn next_lifeform_id() -> u32 {
        LIFEFORM_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn next_species_id() -> u32 {
        SPECIES_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn init(&mut self, num_lifeforms: u32, bounds: Rect) -> Vec<Cell>{
        use rand::Rng;
        
        LIFEFORM_ID.store(0, Ordering::Relaxed);
        SPECIES_ID.store(0, Ordering::Relaxed);
        GENE_ID.store(0, Ordering::Relaxed);
        self.lifeforms.clear();

        let mut rng = rand::thread_rng();
        let mut cells = Vec::new();

        for i in 0..num_lifeforms {
            // Create random genome for this lifeformss
            let lifeform_id = GeneticAlgorithm::next_lifeform_id();
            let num_genes = rng.gen_range(20..100);
            let gene_length = 100;
            let genome = Genome::init_random(&mut rng, num_genes, gene_length);
            
            // Sequence genome to create GRN (stored on CPU for now)
            let grn = sequence_grn(&genome);
            
            // Random position throughout bounds
            let pos = [
                bounds.left + rng.gen::<f32>() * bounds.width,
                bounds.top + rng.gen::<f32>() * bounds.height,
            ];
            
            let cell_idx = cells.len() as u32;
            let cell = Cell::new(pos, i as u32, 100.0);
            cells.push(cell);
            
            // Create lifeform with 1 cell
            let lifeform = Lifeform::new(lifeform_id, Vec::from([cell_idx]), genome, grn);
            self.lifeforms.push(lifeform);
        }

        cells
    }

    /// Get number of lifeforms (from genomes/GRNs count)
    pub fn num_lifeforms(&self) -> usize {
        self.lifeforms.len()
    }
}