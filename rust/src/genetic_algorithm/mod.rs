// Genetic algorithm module - manages genomes, lifeforms, and gene regulatory networks

pub mod genome;
pub mod systems;
pub mod sequencer;
pub mod utils;

pub use genome::Genome;
pub use sequencer::sequence_grn;

