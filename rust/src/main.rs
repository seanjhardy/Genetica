// Main entry point for the Genetica Rust simulation
// This file should be minimal - just initialization and event loop setup

mod utils;
mod simulator;
mod gpu;
mod ui;

use simulator::Simulator;

fn main() {
    env_logger::init();
    
    // Initialize puffin profiler
    puffin::set_scopes_on(true);
    let server_addr = "127.0.0.1:8587";
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    println!("Puffin profiler available at http://{} - view in puffin-viewer", server_addr);
    
    // Create and run the simulator
    let simulator = Simulator::new();
    simulator.run();
}
