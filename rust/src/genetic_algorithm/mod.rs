// Genetic algorithm module - manages genomes, lifeforms, and gene regulatory networks

pub mod genome;
pub mod systems;
pub mod sequencer;
pub mod utils;
pub mod species;
pub mod genetic_algorithm;

pub use genome::Genome;
pub use sequencer::sequence_grn;
pub use species::Species;
pub use genetic_algorithm::GeneticAlgorithm;
