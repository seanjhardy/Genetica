use std::collections::{HashMap, HashSet};

use crate::genetic_algorithm::lifeform::Lifeform;
use crate::gpu::structures::{Species};

pub struct GeneticAlgorithm {
    pub lifeforms: HashMap<usize, Lifeform>,
    pub species: HashMap<usize, Species>,
    pub living_species: HashSet<usize>,
}

impl GeneticAlgorithm {
    pub fn new() -> Self {
        Self {
            lifeforms: HashMap::new(),
            species: HashMap::new(),
            living_species: HashSet::new(),
        }
    }
}