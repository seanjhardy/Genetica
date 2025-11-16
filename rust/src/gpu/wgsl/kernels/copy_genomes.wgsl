@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;

// Parallel genome copying shader - optimizes copy_genome operations
// Each thread copies a portion of one genome, allowing parallel processing

@group(0) @binding(17)
var<storage, read_write> genomes: array<GenomeEntry>;

// Buffer containing copy operations: (dest_slot, src_slot) pairs
struct CopyOperation {
    dest_slot: u32,
    src_slot: u32,
}

@group(0) @binding(24)
var<storage, read> copy_operations: array<CopyOperation>;

// Parallel copy: each thread handles a portion of the genome
// Workgroup size should be chosen to divide GENOME_WORD_COUNT evenly
// For 800 words (200 genes * 4 words), workgroup_size(64) works well (800/64 = 12.5, round up to 13 workgroups per genome)
@compute @workgroup_size(64)
fn copy_genomes_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let operation_index = global_id.x / 13u; // 13 workgroups per genome (800 words / 64 threads = 12.5, round up)
    let word_offset = (global_id.x % 13u) * 64u;

    if operation_index >= arrayLength(&copy_operations) {
        return;
    }

    let op = copy_operations[operation_index];
    if op.dest_slot >= arrayLength(&genomes) || op.src_slot >= arrayLength(&genomes) {
        return;
    }
    
    // Copy gene IDs (200 genes, can be done by first 4 threads)
    if global_id.x < 4u && word_offset < MAX_GENES_PER_GENOME {
        let start_idx = word_offset;
        let end_idx = min(start_idx + 64u, MAX_GENES_PER_GENOME);
        for (var i: u32 = start_idx; i < end_idx; i = i + 1u) {
            genomes[op.dest_slot].gene_ids[i] = genomes[op.src_slot].gene_ids[i];
        }
    }
    
    // Copy gene sequences (800 words, distributed across threads)
    let seq_start = word_offset;
    let seq_end = min(seq_start + 64u, GENOME_WORD_COUNT);
    for (var i: u32 = seq_start; i < seq_end; i = i + 1u) {
        genomes[op.dest_slot].gene_sequences[i] = genomes[op.src_slot].gene_sequences[i];
    }
}

