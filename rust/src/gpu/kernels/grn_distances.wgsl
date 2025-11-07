// Compute shader for calculating distances between cells in GRNs

struct CellData {
    pos: vec2<f32>,
    energy: f32,
    generation: u32,
}

@group(0) @binding(0)
var<storage, read> cell_data: array<CellData>;

@group(0) @binding(1)
var<storage, read_write> cell_distances: array<f32>;

// Constants (will be passed as uniforms or push constants in full implementation)
const MAX_CELLS_PER_GRN: u32 = 1000u;
const NUM_GRNS: u32 = 1000u; // Will be dynamic

// Calculate linear index for triangular matrix
// Distance between cell i and cell j (where i < j) is stored at:
// i * num_cells - (i * (i + 1)) / 2 + (j - i - 1)
fn get_distance_index(cell_i: u32, cell_j: u32, num_cells: u32) -> u32 {
    let firstIndex = min(cell_i, cell_j);
    let secondIndex = max(cell_i, cell_j);
    return firstIndex * num_cells - (firstIndex * (firstIndex + 1u)) / 2u + (secondIndex - firstIndex - 1u);
}

fn distance(a: vec2<f32>, b: vec2<f32>) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Calculate: grn_id, cell_i, cell_j
    let num_cells = MAX_CELLS_PER_GRN;
    let num_pairs_per_grn = num_cells * (num_cells - 1u) / 2u;
    let total_pairs = NUM_GRNS * num_pairs_per_grn;
    
    if (idx >= total_pairs) {
        return;
    }
    
    let grn_id = idx / num_pairs_per_grn;
    let pair_idx = idx % num_pairs_per_grn;
    
    // Convert linear index to cell pair (i, j) where i < j
    var cell_i: u32 = 0u;
    var cell_j: u32 = 0u;
    var remaining = pair_idx;
    
    for (var i: u32 = 0u; i < num_cells; i++) {
        let pairs_in_row = num_cells - i - 1u;
        if (remaining < pairs_in_row) {
            cell_i = i;
            cell_j = i + 1u + remaining;
            break;
        }
        remaining -= pairs_in_row;
    }
    
    // Get cell positions (accounting for GRN offset)
    let cell1_idx = grn_id * MAX_CELLS_PER_GRN + cell_i;
    let cell2_idx = grn_id * MAX_CELLS_PER_GRN + cell_j;
    let cell1 = cell_data[cell1_idx];
    let cell2 = cell_data[cell2_idx];
    
    // Calculate distance
    let dist = distance(cell1.pos, cell2.pos);
    
    // Store in triangular matrix (accounting for GRN offset)
    let distance_idx = grn_id * num_pairs_per_grn + get_distance_index(cell_i, cell_j, num_cells);
    cell_distances[distance_idx] = dist;
}

