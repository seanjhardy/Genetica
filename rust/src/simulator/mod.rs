// Simulator module - manages the overall simulation, window, camera, and environment

pub mod simulator;
pub mod environment;
pub mod renderer;
pub mod planet;
pub mod genetic_algorithm;

pub use simulator::Simulator;
pub use genetic_algorithm::GeneticAlgorithm;