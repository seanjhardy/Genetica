@include src/gpu/wgsl/utils/genetic_algorithm.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;
@group(0) @binding(1)
var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2)
var<storage, read_write> cell_free_list: CellFreeList;
@group(0) @binding(3)
var<storage, read_write> cell_counter: Counter;
@group(0) @binding(4)
var<storage, read_write> spawn_buffer: SpawnBuffer;
@group(0) @binding(5)
var<storage, read_write> nutrient_grid: NutrientGrid;
@group(0) @binding(6)
var<storage, read_write> links: array<Link>;
@group(0) @binding(7)
var<storage, read_write> link_free_list: FreeList;
@group(0) @binding(8)
var<storage, read_write> cell_bucket_heads: array<atomic<i32>>;
@group(0) @binding(11)
var<storage, read_write> cell_hash_next: array<i32>;
@group(0) @binding(12)
var<storage, read_write> grn_descriptors: array<GrnDescriptor>;
@group(0) @binding(13)
var<storage, read_write> grn_units: array<CompiledRegulatoryUnit>;
@group(0) @binding(14)
var<storage, read_write> lifeforms: array<Lifeform>;
@group(0) @binding(15)
var<storage, read_write> lifeform_free: FreeList;
@group(0) @binding(16)
var<storage, read_write> next_lifeform_id: Counter;
@group(0) @binding(17)
var<storage, read_write> genomes: array<GenomeEntry>;
@group(0) @binding(18)
var<storage, read_write> species_entries: array<SpeciesEntry>;
@group(0) @binding(19)
var<storage, read_write> species_free: FreeList;
@group(0) @binding(20)
var<storage, read_write> next_species_id: Counter;
@group(0) @binding(21)
var<storage, read_write> next_gene_id: Counter;
@group(0) @binding(22)
var<storage, read_write> lifeform_counter: Counter;
@group(0) @binding(23)
var<storage, read_write> species_counter: Counter;
@group(0) @binding(24)
var<storage, read_write> position_changes: array<PositionChangeEntry>;

fn rand(seed: vec2<u32>) -> f32 {
    var x = seed.x * 1664525u + 1013904223u;
    var y = seed.y * 22695477u + 1u;
    let n = x ^ y;
    return f32(n & 0x00FFFFFFu) / f32(0x01000000u);
}

// Process deferred genome copy/mutate requests in parallel.
@compute @workgroup_size(64)
fn process_genome_events(@builtin(local_invocation_id) local_id: vec3<u32>) {
    // Pop one event from the queue (LIFO). Threads exit if queue empty.
    var event_index: u32 = 0u;
    loop {
        let prev = atomicLoad(&genome_events.counter.value);
        if prev == 0u {
            return;
        }
        let desired = prev - 1u;
        let exchange = atomicCompareExchangeWeak(&genome_events.counter.value, prev, desired);
        if exchange.old_value == prev && exchange.exchanged {
            event_index = desired;
            break;
        }
    };

    if event_index >= MAX_GENOME_EVENTS {
        return;
    }

    let event = genome_events.events[event_index];
    if event.dst_genome_slot >= arrayLength(&genomes) || event.src_genome_slot >= arrayLength(&genomes) {
        return;
    }

    // Strided copy of gene ids
    let stride = 64u;
    var idx = local_id.x;
    while idx < MAX_GENES_PER_GENOME {
        genomes[event.dst_genome_slot].gene_ids[idx] = genomes[event.src_genome_slot].gene_ids[idx];
        idx = idx + stride;
    }

    // Strided copy of gene sequences
    idx = local_id.x;
    while idx < GENOME_WORD_COUNT {
        genomes[event.dst_genome_slot].gene_sequences[idx] = genomes[event.src_genome_slot].gene_sequences[idx];
        idx = idx + stride;
    }

    // Apply mutation after copy
    mutate_genome(event.dst_genome_slot, event.seed);

    // Mark lifeform as having an updated genome slot (for sequencing later)
    if event.lifeform_slot < arrayLength(&lifeforms) {
        lifeforms[event.lifeform_slot].grn_descriptor_slot = event.dst_genome_slot;
        lifeforms[event.lifeform_slot].flags = lifeforms[event.lifeform_slot].flags | LIFEFORM_FLAG_ACTIVE;
    }
}
