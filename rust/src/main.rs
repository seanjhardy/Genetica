mod utils;
mod simulator;
mod gpu;
mod ui;
mod genetic_algorithm;

fn main() {
    env_logger::init();
    
    // Initialize puffin profiler
    puffin::set_scopes_on(true);
    let server_addr = "127.0.0.1:8587";
    let _puffin_server = puffin_http::Server::new(&server_addr).unwrap();
    println!("Puffin profiler available at http://{} - view in puffin-viewer", server_addr);
    
    if let Err(e) = simulator::run() {
        eprintln!("Error: {:?}", e);
    }
}
