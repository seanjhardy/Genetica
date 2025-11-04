// Environment module - manages bounds, lifeforms, cells, and physics simulation

use crate::modules::math::{Rect, Vec2};
use crate::modules::ui::DragHandler;
use crate::gpu::structures::{Cell, Lifeform};
use crate::genetic_algorithm::{Genome, sequence_grn};
use crate::genetic_algorithm::systems::morphology::GeneRegulatoryNetwork;

/// Environment manages the simulation bounds and genetic data (genomes/GRNs)
/// NOTE: Cells and lifeforms are stored on GPU in GpuBuffers, not here
/// GRNs are stored here on CPU but can be uploaded to GPU for parallel simulation
/// The GRN is essentially a recurrent neural network - we'll use GPU compute kernels
/// to simulate the nodes in parallel
pub struct Environment {
    bounds: Rect,
    drag_handler: DragHandler,
    /// Store GRNs for each lifeform (index matches GPU lifeforms array)
    /// These can be uploaded to GPU for parallel node simulation
    grns: Vec<GeneRegulatoryNetwork>,
    /// Store genomes for each lifeform (index matches GPU lifeforms array)
    /// Only used during initialization
    genomes: Vec<Genome>,
}

impl Environment {
    pub fn new(_initial_bounds: Rect, _num_lifeforms: usize) -> Self {
        Self {
            bounds: _initial_bounds,
            drag_handler: DragHandler::new(),
            grns: Vec::new(),
            genomes: Vec::new(),
        }
    }

    /// Initialize genomes and GRNs for lifeforms
    /// Returns vectors of cells and lifeforms to initialize the GPU buffers
    /// GRNs are stored on CPU here and can be uploaded to GPU for parallel simulation
    pub fn initialize_genomes(&mut self, num_lifeforms: usize) -> (Vec<Cell>, Vec<Lifeform>) {
        use rand::Rng;
        
        let initial_energy = 100.0; // Starting energy for each cell
        self.grns.clear();
        self.genomes.clear();

        let mut rng = rand::thread_rng();
        let mut cells = Vec::new();
        let mut lifeforms = Vec::new();

        for i in 0..num_lifeforms {
            // Create random genome for this lifeform
            let num_genes = rng.gen_range(20..100);
            let gene_length = 100;
            let genome = Genome::init_random(&mut rng, num_genes, gene_length);
            
            // Sequence genome to create GRN (stored on CPU for now)
            let grn = sequence_grn(&genome);
            
            // Random position throughout bounds
            let pos = [
                self.bounds.left + rng.gen::<f32>() * self.bounds.width,
                self.bounds.top + rng.gen::<f32>() * self.bounds.height,
            ];
            
            let cell_idx = cells.len() as u32;
            let cell = Cell::new(pos, i as u32, initial_energy);
            cells.push(cell);
            
            // Create lifeform with 1 cell
            let lifeform = Lifeform::new(cell_idx, 1);
            lifeforms.push(lifeform);
            
            // Store genome and GRN on CPU (GRN can be uploaded to GPU for parallel simulation)
            self.genomes.push(genome);
            self.grns.push(grn);
        }

        (cells, lifeforms)
    }

    /// Get number of lifeforms (from genomes/GRNs count)
    pub fn num_lifeforms(&self) -> usize {
        self.grns.len()
    }

    /// Update environment state (bounds dragging, etc.)
    pub fn update(&mut self, mouse_world_pos: Vec2, zoom: f32, ui_hovered: bool) {
        if !ui_hovered {
            let sensitivity = 15.0 / zoom;
            let delta_bounds = self.drag_handler.update(mouse_world_pos, self.bounds, sensitivity);

            if delta_bounds.width != 0.0
                || delta_bounds.height != 0.0
                || delta_bounds.left != 0.0
                || delta_bounds.top != 0.0
            {
                // Round bounds to grid (20px grid)
                let new_left = (self.bounds.left + delta_bounds.left) / 20.0;
                let new_top = (self.bounds.top + delta_bounds.top) / 20.0;
                let new_width = (self.bounds.width + delta_bounds.width) / 20.0;
                let new_height = (self.bounds.height + delta_bounds.height) / 20.0;

                self.bounds = Rect::new(
                    new_left.round() * 20.0,
                    new_top.round() * 20.0,
                    new_width.round() * 20.0,
                    new_height.round() * 20.0,
                );
            }
        } else {
            self.drag_handler.reset();
        }
    }

    /// Handle mouse press events
    pub fn handle_mouse_press(&mut self, pressed: bool) {
        self.drag_handler.handle_mouse_press(pressed);
    }

    /// Get current bounds
    pub fn get_bounds(&self) -> Rect {
        self.bounds
    }
    /// Get cursor hint for drag handle
    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        self.drag_handler.get_cursor_hint()
    }

}
