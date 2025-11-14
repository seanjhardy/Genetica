@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;


fn hash_cell_position(pos: vec2<f32>) -> u32 {
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u {
        return 0u;
    }
    let grid = vec2<i32>(floor(pos / HASH_CELL_SIZE));
    let hashed = (grid.x * 73856093) ^ (grid.y * 19349663);
    let mask = bucket_count - 1u;
    return u32(hashed) & mask;
}


fn compute_collision_correction(index: u32, position: vec2<f32>, radius: f32) -> vec2<f32> {
    let bucket_count = arrayLength(&cell_bucket_heads);
    if bucket_count == 0u {
        return vec2<f32>(0.0, 0.0);
    }

    let cell_capacity = arrayLength(&cells);
    let next_length = arrayLength(&cell_hash_next);

    var correction = vec2<f32>(0.0, 0.0);

    var dx: i32 = -1;
    loop {
        if dx > 1 {
            break;
        }

        var dy: i32 = -1;
        loop {
            if dy > 1 {
                break;
            }

            let neighbor_pos = position + vec2<f32>(f32(dx), f32(dy)) * HASH_CELL_SIZE;
            let neighbor_hash = hash_cell_position(neighbor_pos);

            var head = atomicLoad(&cell_bucket_heads[neighbor_hash]);
            loop {
                if head == -1 {
                    break;
                }

                let neighbor_index = u32(head);
                if neighbor_index != index && neighbor_index < cell_capacity {
                    let neighbor = cells[neighbor_index];
                    if neighbor.is_alive != 0u {
                        let delta = position - neighbor.pos;
                        let dist_sq = dot(delta, delta);
                        let min_dist = radius + neighbor.radius;
                        if min_dist > 0.0 && dist_sq < (min_dist * min_dist) {
                            let dist = sqrt(max(dist_sq, COLLISION_EPSILON));
                            var push_dir = vec2<f32>(0.0, 0.0);
                            if dist > 0.0 {
                                push_dir = delta / dist;
                            }
                            if push_dir.x == 0.0 && push_dir.y == 0.0 {
                                if index < neighbor_index {
                                    push_dir = vec2<f32>(1.0, 0.0);
                                } else {
                                    push_dir = vec2<f32>(-1.0, 0.0);
                                }
                            }
                            let overlap = min_dist - dist;
                            if overlap > 0.0 {
                                correction += push_dir * (overlap * 0.5);
                            }
                        }
                    }
                }

                var next_head: i32 = -1;
                if neighbor_index < next_length {
                    next_head = cell_hash_next[neighbor_index];
                }
                head = next_head;
            }

            dy = dy + 1;
        }

        dx = dx + 1;
    }

    return correction;
}