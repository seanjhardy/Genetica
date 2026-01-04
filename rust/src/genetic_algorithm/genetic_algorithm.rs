use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use puffin::profile_scope;
use rand::Rng;

// Genetic algorithm module - manages genomes, lifeforms, and gene regulatory networks
use crate::genetic_algorithm::sequence_grn;
use crate::genetic_algorithm::systems::morphology::compile_grn::compile_grn;
use crate::genetic_algorithm::Lifeform;
use crate::genetic_algorithm::{Genome, Species};

static GENE_ID: AtomicUsize = AtomicUsize::new(0);
static SPECIES_ID: AtomicUsize = AtomicUsize::new(0);

pub struct GeneticAlgorithm {
    lifeforms: HashMap<usize, Lifeform>,
    species: HashMap<usize, Species>,
    living_species: HashSet<usize>
}

impl GeneticAlgorithm {
    #[allow(dead_code)]
    pub const GENE_DIFFERENCE_SCALAR: f32 = 0.5;
    #[allow(dead_code)]
    pub const BASE_DIFFERENCE_SCALAR: f32 = 0.1;
    pub const COMPATABILITY_DISTANCE_THRESHOLD: f32 = 200.0;

    #[allow(dead_code)]
    pub const INSERT_GENE_CHANCE: f32 = 0.0005;
    #[allow(dead_code)]
    pub const CLONE_GENE_CHANCE: f32 = 0.0001;
    #[allow(dead_code)]
    pub const DELETE_GENE_CHANCE: f32 = 0.0005;
    #[allow(dead_code)]
    pub const CROSSOVER_GENE_CHANCE: f32 = 0.2;

    #[allow(dead_code)]
    pub const MUTATE_BASE_CHANCE: f32 = 0.0005;
    #[allow(dead_code)]
    pub const INSERT_BASE_CHANCE: f32 = 0.00003;
    #[allow(dead_code)]
    pub const DELETE_BASE_CHANCE: f32 = 0.00005;

    pub fn new() -> Self {
        Self {
            lifeforms: HashMap::new(),
            species: HashMap::new(),
            living_species: HashSet::new(),
        }
    }

    pub fn next_gene_id() -> usize {
        GENE_ID.fetch_add(1, Ordering::Relaxed)
    }


    pub fn next_species_id() -> usize {
        SPECIES_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn num_lifeforms(&self) -> usize {
        self.lifeforms.len()
    }

    pub fn num_species(&self) -> usize {
        self.living_species.len()
    }

    pub fn process_events(&mut self, step: usize ) {
        profile_scope!("Process Genetic Events");
/* 
        while let Ok(event) = event_queue.try_recv() {
            match event.event_type {
                CREATE_LIFEFORM_FLAG => {
                    self.create_lifeform(step, event.parent_lifeform_id, event.lifeform_id);
                }
                ADD_CELL_TO_LIFEFORM_FLAG => {
                    self.add_cell_to_lifeform(event.lifeform_id);
                }
                REMOVE_CELL_FROM_LIFEFORM_FLAG => {
                    self.remove_cell_from_lifeform(step, event.lifeform_id);
                }
                _ => {}
            }
        }*/
    }

    fn create_lifeform(&mut self, step: usize, parent_lifeform_id: Option<usize>, lifeform_id: usize) {
        let alive_parent = parent_lifeform_id
        .and_then(|id| self.lifeforms.get(&id))
        .filter(|p| p.is_alive());

        let (genome, species_id) = match alive_parent {
            Some(parent) => {
                let mut genome = parent.genome.clone();
                genome.mutate();
                let child_species_id = self.allocate_species(parent.lifeform_id, lifeform_id, &genome);
                (genome, child_species_id)
            }
            None => {
                let genome = Self::generate_random_genome(&mut rand::thread_rng());
                let species_id = self.create_species(lifeform_id, &genome, 0);
                (genome, species_id)
            }
        };
        let grn = sequence_grn(&genome);
        let compiled = compile_grn(lifeform_id as u32, grn.clone());
        let lifeform = Lifeform::new(step, lifeform_id, species_id, genome.clone(), grn, compiled);
        lifeform.cell_count.store(1, Ordering::Relaxed);
        self.lifeforms.insert(lifeform_id, lifeform);
    }

    pub fn add_cell_to_lifeform(&mut self, lifeform_id: usize) {
        if let Some(lifeform) = self.lifeforms.get_mut(&lifeform_id) {
            lifeform.increment_cell_count();
        }
    }

    pub fn remove_cell_from_lifeform(&mut self, step: usize, lifeform_id: usize) {
        if let Some(lifeform) = self.lifeforms.get_mut(&lifeform_id) {
            lifeform.decrement_cell_count();
            if !lifeform.has_cells() {
                self.remove_lifeform(lifeform_id, step);
            }
        }
    }

    pub fn remove_lifeform(&mut self, lifeform_id: usize, death_time: usize) {
        if let Some(lifeform) = self.lifeforms.remove(&lifeform_id) {
            if let Some(species) = self.species.get_mut(&lifeform.species_id) {
                let extinct = species.deregister_member(death_time);
                if extinct {
                    self.living_species.remove(&lifeform.species_id);
                }
            }
        }
    }

    fn generate_random_genome<R: Rng>(rng: &mut R) -> Genome {
        profile_scope!("GA Generate Random Genome");
        let num_genes = rng.gen_range(2..20);
        let gene_length = 30;
        Genome::init_random(rng, num_genes, gene_length)
    }

    fn create_species(&mut self, mascot_lifeform_id: usize, mascot_genome: &Genome, origin_time: usize) -> usize {
        let species_id = GeneticAlgorithm::next_species_id();
        let species = Species::new(species_id, mascot_lifeform_id, mascot_genome.clone(), origin_time);
        self.species.insert(species_id, species);
        species_id
    }

    fn allocate_species(&mut self, parent_lifeform_id: usize, lifeform_id: usize, genome: &Genome) -> usize {
        let parent_lifeform = self.lifeforms.get(&parent_lifeform_id).unwrap();
        let difference = parent_lifeform.genome.compare(&genome);
        if difference < Self::COMPATABILITY_DISTANCE_THRESHOLD {
            return parent_lifeform.species_id;
        }
        let species_id = self.create_species(lifeform_id, &genome, 0);
        species_id
    }
}