@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;

// Note: This file accesses cells, links, and link_free_list which must be bound
// in the file that includes this one (e.g., links.wgsl, cells.wgsl)

fn remove_link_from_cell(cell_index: u32, link_index: u32) {
    if cell_index >= arrayLength(&cells) {
        return;
    }
    var cell = cells[cell_index];
    // Find and remove the link
    for (var i: u32 = 0u; i < cell.link_count; i = i + 1u) {
        if cell.link_indices[i] == link_index {
            // Shift remaining links left
            for (var j: u32 = i; j < cell.link_count - 1u; j = j + 1u) {
                cell.link_indices[j] = cell.link_indices[j + 1u];
            }
            cell.link_count = cell.link_count - 1u;
            cell.link_indices[cell.link_count] = 0u; // Clear last slot
            cells[cell_index] = cell;
            return;
        }
    }
}

fn release_link(index: u32) {
    if index >= arrayLength(&links) {
        return;
    }

    let existing = links[index];
    if (existing.flags & LINK_FLAG_ALIVE) == 0u {
        return;
    }

    // Remove link from both cells' link_indices arrays
    remove_link_from_cell(existing.a, index);
    remove_link_from_cell(existing.b, index);

    var cleared = existing;
    cleared.flags = 0u;
    cleared.a = 0u;
    cleared.b = 0u;
    cleared.generation_a = 0u;
    cleared.rest_length = 0.0;
    cleared.stiffness = 0.0;
    cleared.energy_transfer_rate = 0.0;
    cleared.generation_b = 0u;
    links[index] = cleared;

    let free_index = atomicAdd(&link_free_list.count, 1u);
    if free_index < arrayLength(&link_free_list.indices) {
        link_free_list.indices[free_index] = index;
    } else {
        atomicSub(&link_free_list.count, 1u);
    }
}


