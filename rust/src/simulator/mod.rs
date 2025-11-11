// Simulator module - manages the overall simulation, window, camera, and environment

pub mod simulator;
pub mod environment;
pub mod renderer;
pub mod planet;
pub mod lifeform_registry;
pub mod state;

#[allow(unused_imports)]
pub use simulator::Simulation;
pub use simulator::Simulator;