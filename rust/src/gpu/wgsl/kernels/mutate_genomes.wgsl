@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/genetic_algorithm.wgsl;

// Parallel genome mutation shader - optimizes mutate_genome operations
// Each thread handles one genome's mutation, allowing parallel processing

@group(0) @binding(17)
var<storage, read_write> genomes: array<GenomeEntry>;

@group(0) @binding(23)
var<storage, read_write> next_gene_id: Counter;

// Buffer containing mutation operations: (genome_slot, seed) pairs
struct MutationOperation {
    genome_slot: u32,
    seed: u32,
}

@group(0) @binding(24)
var<storage, read> mutation_operations: array<MutationOperation>;

// Parallel mutation: each thread handles one genome
@compute @workgroup_size(64)
fn mutate_genomes_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let operation_index = global_id.x;

    if operation_index >= arrayLength(&mutation_operations) {
        return;
    }

    let op = mutation_operations[operation_index];
    if op.genome_slot >= arrayLength(&genomes) {
        return;
    }
    
    // Call the mutate_genome function for this genome
    // Note: This still uses the sequential implementation, but now runs in parallel
    // across multiple genomes. For further optimization, the mutation logic itself
    // could be parallelized within each genome.
    mutate_genome(op.genome_slot, op.seed);
}

