import package::constants::{LINK_FLAG_ACTIVE};
import package::types::{Cell, Link, FreeList};

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
    if (existing.flags & LINK_FLAG_ACTIVE) == 0u {
        return;
    }

    // Remove link from both cells' link_indices arrays
    remove_link_from_cell(existing.a_cell, index);
    remove_link_from_cell(existing.b_cell, index);

    var cleared = existing;
    cleared.flags = 0u;
    cleared.a_cell = 0u;
    cleared.b_cell = 0u;
    cleared.a_generation = 0u;
    cleared.angle_from_a = 0.0;
    cleared.angle_from_b = 0.0;
    cleared.stiffness = 0.0;
    cleared.b_generation = 0u;
    links[index] = cleared;

    let free_index = atomicAdd(&link_free_list.count, 1u);
    if free_index < arrayLength(&link_free_list.indices) {
        link_free_list.indices[free_index] = index;
    } else {
        atomicSub(&link_free_list.count, 1u);
    }
}
