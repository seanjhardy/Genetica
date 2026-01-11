@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;

// Cells buffer
@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
// GRN descriptors - one per lifeform
@group(0) @binding(1) var<storage, read> grn_descriptors: array<GrnDescriptor>;
// Compiled GRN regulatory units (compacted storage)
@group(0) @binding(2) var<storage, read> grn_units: array<CompiledRegulatoryUnit>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_idx = global_id.x;

    // Bounds check
    if (cell_idx >= arrayLength(&cells)) {
        return;
    }

    let cell = cells[cell_idx];
    let lifeform_id = cell.lifeform_id;

    // Skip cells that don't belong to any lifeform
    if (lifeform_id == 0u) {
        return;
    }

    let grn_desc = grn_descriptors[lifeform_id];

    // Process each regulatory unit in this GRN
    for (var unit_idx = 0u; unit_idx < grn_desc.unit_count; unit_idx++) {
        let unit_offset = grn_desc.unit_start_index + unit_idx;
        let regulatory_unit = grn_units[unit_offset];

        // Calculate activation for this regulatory unit
        var activation = 0.0;

        // Process each input connection
        for (var input_idx = 0u; input_idx < regulatory_unit.input_count; input_idx++) {
            let grn_input = regulatory_unit.inputs[input_idx];

            // Read the input value from the cell's input array
            let input_value = cells[cell_idx].inputs[grn_input.index];

            // Apply weight based on promoter type
            if (grn_input.promoter_type == 0u) {
                // Additive promoter: add weighted input
                activation += input_value * grn_input.weight;
            } else {
                // Multiplicative promoter: modulate activation by weighted input
                // Initialize activation to 1.0 for multiplicative regulation if this is the first input
                if (input_idx == 0u) {
                    activation = 1.0;
                }
                activation *= (1.0 + input_value * grn_input.weight);
            }
        }

        // TODO: Handle activation threshold and regulatory unit output
        // For now, just store the activation value somewhere
        // This will be implemented later as per user request
    }
}
