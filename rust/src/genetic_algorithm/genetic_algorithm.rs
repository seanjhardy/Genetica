use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use puffin::profile_scope;
use rand::Rng;

// Genetic algorithm module - manages genomes, lifeforms, and gene regulatory networks
use crate::genetic_algorithm::sequence_grn;
use crate::genetic_algorithm::systems::GeneRegulatoryNetwork;
use crate::genetic_algorithm::systems::morphology::compile_grn::compile_grn;
use crate::gpu::structures::{Cell, Lifeform};
use crate::utils::math::Rect;
use crate::genetic_algorithm::{Genome, Species};

static GENE_ID: AtomicUsize = AtomicUsize::new(0);
static LIFEFORM_ID: AtomicUsize = AtomicUsize::new(0);
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

    pub fn new(_device: &wgpu::Device) -> Self {
        Self {
            lifeforms: HashMap::new(),
            species: HashMap::new(),
            living_species: HashSet::new(),
        }
    }

    pub fn next_gene_id() -> usize {
        GENE_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn next_lifeform_id() -> usize {
        LIFEFORM_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn next_species_id() -> usize {
        SPECIES_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn init(
        &mut self,
        num_lifeforms: u32,
        bounds: Rect,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Vec<Cell> {
        LIFEFORM_ID.store(0, Ordering::Relaxed);
        SPECIES_ID.store(0, Ordering::Relaxed);
        GENE_ID.store(0, Ordering::Relaxed);
        self.lifeforms.clear();
        self.species.clear();
        self.living_species.clear();

        let mut rng = rand::thread_rng();
        let mut cells = Vec::with_capacity(num_lifeforms as usize);

        for _ in 0..num_lifeforms {
            let lifeform_id = GeneticAlgorithm::next_lifeform_id();
            let genome = Self::generate_random_genome(&mut rng);
            let grn = sequence_grn(&genome);
            let species_id = self.create_species(lifeform_id, &genome, 0);
            self.attach_lifeform(lifeform_id, species_id, genome.clone(), grn.clone());

            let cell = Self::create_seed_cell(&mut rng, &bounds, lifeform_id);
            cells.push(cell);
        }

        cells
    }

    pub fn num_species(&self) -> usize {
        self.living_species.len()
    }

    fn register_new_lifeform_internal(
        &mut self,
        lifeform_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
        birth_time: usize,
    ) -> usize {
        let species_id = self.create_species(lifeform_id, &genome, birth_time);
        self.attach_lifeform(lifeform_id, species_id,  genome, grn);
        species_id
    }

    pub fn spawn_random_lifeform<R: Rng>(
        &mut self,
        rng: &mut R,
        birth_time: usize,
    ) -> (usize, usize) {
        profile_scope!("GA Spawn Random Lifeform");
        let lifeform_id = GeneticAlgorithm::next_lifeform_id();
        let genome = Self::generate_random_genome(rng);
        let grn = sequence_grn(&genome);
        let species_id = self.register_new_lifeform_internal(lifeform_id, genome, grn, birth_time);
        (lifeform_id, species_id)
    }

    pub fn register_child_lifeform(
        &mut self,
        lifeform_id: usize,
        parent_lifeform_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
        birth_time: usize,
    ) -> usize {
        let parent_species_entry = self
            .lifeforms
            .get(&parent_lifeform_id)
            .and_then(|lf| {
                let species_id = lf.species_id;
                self.species
                    .get(&species_id)
                    .map(|species| (species_id, species))
            });

        let (parent_species_id, needs_new_species) = if let Some((_, species)) = parent_species_entry
        {
            let needs_new =
                genome.compare(&species.mascot_genome) > Self::COMPATABILITY_DISTANCE_THRESHOLD;
            (Some(species.id), needs_new)
        } else {
            (None, true)
        };

        let species_id = if needs_new_species {
            self.create_species(lifeform_id, &genome, birth_time)
        } else {
            parent_species_id.unwrap()
        };

        self.attach_lifeform(lifeform_id, species_id,  genome, grn);
        species_id
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

    pub fn list_active_lifeforms(&self) -> Vec<(usize, usize)> {
        self.lifeforms
            .iter()
            .map(|(&id, lf)| (id, lf.species_id))
            .collect()
    }

    pub fn compiled_grn(&self, lifeform_id: usize) -> Option<&crate::gpu::structures::CompiledGrn> {
        self.lifeforms.get(&lifeform_id).map(|lf| lf.compiled_grn())
    }

    pub fn register_division_offspring<R: Rng>(
        &mut self,
        parent_lifeform_id: usize,
        birth_time: usize,
        rng: &mut R,
    ) -> Option<(usize, usize)> {
        profile_scope!("GA Register Division Offspring");
        let mut child_genome = self.lifeforms.get(&parent_lifeform_id)?.genome.clone();
        child_genome.mutate();
        let child_grn = sequence_grn(&child_genome);
        let child_lifeform_id = GeneticAlgorithm::next_lifeform_id();
        let species_id = self.register_child_lifeform(
            child_lifeform_id,
            parent_lifeform_id,
            child_genome,
            child_grn,
            birth_time,
        );
        Some((child_lifeform_id, species_id))
    }

    fn generate_random_genome<R: Rng>(rng: &mut R) -> Genome {
        profile_scope!("GA Generate Random Genome");
        let num_genes = rng.gen_range(2..20);
        let gene_length = 30;
        Genome::init_random(rng, num_genes, gene_length)
    }

    fn create_seed_cell<R: Rng>(rng: &mut R, bounds: &Rect, lifeform_id: usize) -> Cell {
        let pos = [
            bounds.left + rng.gen::<f32>() * bounds.width,
            bounds.top + rng.gen::<f32>() * bounds.height,
        ];
        let random_radius = rng.gen_range(0.5..4.0);
        let energy = rng.gen_range(80.0..140.0);
        Cell::new(pos, random_radius, lifeform_id as u32, energy)
    }

    fn create_species(&mut self, mascot_lifeform_id: usize, mascot_genome: &Genome, origin_time: usize) -> usize {
        let species_id = GeneticAlgorithm::next_species_id();
        let species = Species::new(species_id, mascot_lifeform_id, mascot_genome.clone(), origin_time);
        self.species.insert(species_id, species);
        species_id
    }

    fn attach_lifeform(
        &mut self,
        lifeform_id: usize,
        species_id: usize,
        genome: Genome,
        grn: GeneRegulatoryNetwork,
    ) {
         if let Some(species) = self.species.get_mut(&species_id) {
            let first_member = species.register_member();
            if first_member {
                self.living_species.insert(species_id);
            }
        }
        let compiled = compile_grn(lifeform_id as u32, &grn);
        let lifeform = Lifeform::new(lifeform_id, species_id, genome, grn, compiled);
        self.lifeforms.insert(lifeform_id, lifeform);
    }
}