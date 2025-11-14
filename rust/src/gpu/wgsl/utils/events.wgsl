@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/types.wgsl;


fn push_link_event(
    kind: u32,
    link_index: u32,
    cell_a: u32,
    cell_b: u32,
    rest_length: f32,
    stiffness: f32,
    energy_transfer_rate: f32,
) {
    let event_index = atomicAdd(&link_events.counter.value, 1u);
    if event_index < arrayLength(&link_events.events) {
        link_events.events[event_index] = LinkEvent(
            kind,
            link_index,
            cell_a,
            cell_b,
            rest_length,
            stiffness,
            energy_transfer_rate,
            0.0,
        );
    } else {
        atomicSub(&link_events.counter.value, 1u);
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

    push_link_event(
        LINK_EVENT_KIND_REMOVE,
        index,
        existing.a,
        existing.b,
        existing.rest_length,
        existing.stiffness,
        existing.energy_transfer_rate,
    );
}



fn push_cell_event(
    kind: u32,
    parent_cell_index: u32,
    parent_lifeform_slot: u32,
    flags: u32,
    position: vec2<f32>,
    radius: f32,
    energy: f32,
) {
    let event_index = atomicAdd(&cell_events.counter.value, 1u);
    if event_index < arrayLength(&cell_events.events) {
        cell_events.events[event_index] = CellEvent(
            kind,
            parent_cell_index,
            parent_lifeform_slot,
            flags,
            position,
            radius,
            energy,
        );
    } else {
        atomicSub(&cell_events.counter.value, 1u);
    }
}

