@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;


fn release_link(index: u32) {
    if index >= arrayLength(&links) {
        return;
    }

    let existing = links[index];
    if (existing.flags & LINK_FLAG_ALIVE) == 0u {
        return;
    }

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


