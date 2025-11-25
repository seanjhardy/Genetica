@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;
@include src/gpu/wgsl/utils/events.wgsl;

// ============================================
// Free List Management
// ============================================

fn allocate_lifeform_slot() -> u32 {
    loop {
        let prev = atomicLoad(&lifeform_free.count);
        if prev == 0u {
            break;
        }
        let desired = prev - 1u;
        let exchange = atomicCompareExchangeWeak(&lifeform_free.count, prev, desired);
        if exchange.old_value == prev && exchange.exchanged {
            if desired < arrayLength(&lifeform_free.indices) {
                return lifeform_free.indices[desired];
            }
            break;
        }
    }
    return LIFEFORM_CAPACITY;
}

fn recycle_lifeform_slot(slot: u32) {
    let index = atomicAdd(&lifeform_free.count, 1u);
    if index < arrayLength(&lifeform_free.indices) {
        lifeform_free.indices[index] = slot;
    }
}

fn allocate_species_slot() -> u32 {
    loop {
        let prev = atomicLoad(&species_free.count);
        if prev == 0u {
            break;
        }
        let desired = prev - 1u;
        let exchange = atomicCompareExchangeWeak(&species_free.count, prev, desired);
        if exchange.old_value == prev && exchange.exchanged {
            if desired < arrayLength(&species_free.indices) {
                return species_free.indices[desired];
            }
            break;
        }
    }
    return MAX_SPECIES_CAPACITY;
}

fn recycle_species_slot(slot: u32) {
    let index = atomicAdd(&species_free.count, 1u);
    if index < arrayLength(&species_free.indices) {
        species_free.indices[index] = slot;
    }
}

fn queue_spawn_cell(new_cell: Cell) -> bool {
    let index = atomicAdd(&spawn_buffer.counter.value, 1u);
    if index < arrayLength(&spawn_buffer.requests) {
        spawn_buffer.requests[index] = new_cell;
        return true;
    }
    atomicSub(&spawn_buffer.counter.value, 1u);
    return false;
}

// ============================================
// Cell-to-Link Mapping Functions
// ============================================

// Add a link index to a cell's link_indices array
fn add_link_to_cell(cell_index: u32, link_index: u32) {
    if cell_index >= arrayLength(&cells) {
        return;
    }
    var cell = cells[cell_index];
    if cell.link_count >= 6u {
        return; // Maximum 6 links per cell
    }
    // Check if link already exists
    for (var i: u32 = 0u; i < cell.link_count; i = i + 1u) {
        if cell.link_indices[i] == link_index {
            return; // Link already exists
        }
    }
    // Add the link
    cell.link_indices[cell.link_count] = link_index;
    cell.link_count = cell.link_count + 1u;
    cells[cell_index] = cell;
}

// Redistribute links from a parent cell to either parent or child based on proximity
fn redistribute_cell_links(parent_index: u32, child_index: u32) {
    if parent_index >= arrayLength(&cells) || child_index >= arrayLength(&cells) {
        return;
    }

    var parent_cell = cells[parent_index];
    var child_cell = cells[child_index];

    if parent_cell.is_alive == 0u || child_cell.is_alive == 0u {
        return;
    }

    let parent_pos = parent_cell.pos;
    let child_pos = child_cell.pos;
    
    // Iterate through all links connected to the parent
    // We need to iterate through the parent's link_indices
    let link_count = parent_cell.link_count;
    
    // Since we might modify the array while iterating, we'll collect link indices first
    var links_to_redistribute: array<u32, 6>;
    var links_count: u32 = 0u;
    
    // Collect links that need redistribution (exclude the direct parent-child link)
    for (var i: u32 = 0u; i < link_count; i = i + 1u) {
        let link_index = parent_cell.link_indices[i];
        if link_index >= arrayLength(&links) {
            continue;
        }
        let link = links[link_index];
        if (link.flags & LINK_FLAG_ALIVE) == 0u {
            continue;
        }
        
        // Skip the direct parent-child link
        if (link.a == parent_index && link.b == child_index) || (link.a == child_index && link.b == parent_index) {
            continue;
        }

        links_to_redistribute[links_count] = link_index;
        links_count = links_count + 1u;
    }
    
    // Redistribute each link
    for (var i: u32 = 0u; i < links_count; i = i + 1u) {
        let link_index = links_to_redistribute[i];
        if link_index >= arrayLength(&links) {
            continue;
        }

        var link = links[link_index];
        if (link.flags & LINK_FLAG_ALIVE) == 0u {
            continue;
        }
        
        // Determine which cell this link connects to (the other end, not the parent)
        var other_cell_index: u32;
        if link.a == parent_index {
            other_cell_index = link.b;
        } else if link.b == parent_index {
            other_cell_index = link.a;
        } else {
            continue; // This link doesn't connect to parent - should not happen
        }

        if other_cell_index >= arrayLength(&cells) {
            continue;
        }

        var other_cell = cells[other_cell_index];
        if other_cell.is_alive == 0u {
            continue;
        }

        let other_pos = other_cell.pos;
        
        // Calculate distances to parent and child
        let dist_to_parent_sq = dot(parent_pos - other_pos, parent_pos - other_pos);
        let dist_to_child_sq = dot(child_pos - other_pos, child_pos - other_pos);
        
        // Choose the closer cell (parent or child)
        let target_cell_index = select(parent_index, child_index, dist_to_child_sq < dist_to_parent_sq);
        
        // If target is different from parent, update the link
        if target_cell_index != parent_index {
            // Remove link from parent
            remove_link_from_cell(parent_index, link_index);
            
            // Re-read parent cell after removal to get fresh state
            // This prevents issues where the link_indices array has shifted
            parent_cell = cells[parent_index];
            
            // Update link to point to child instead of parent
            if link.a == parent_index {
                link.a = child_index;
                link.generation_a = child_cell.generation;
            } else {
                link.b = child_index;
                link.generation_b = child_cell.generation;
            }
            links[link_index] = link;
            
            // Add link to both cells' link_indices arrays to ensure symmetry
            add_link_to_cell(child_index, link_index);
            add_link_to_cell(other_cell_index, link_index);
        }
        // If target is parent, keep the link as is
    }
}

// ============================================
// Genome Access Functions
// ============================================
// GenomeEntry structure:
// - gene_ids[0..199]: Array of gene IDs (which genes are present, 0 = empty slot)
// - gene_sequences[0..799]: Packed base pair data
//   Each gene uses 4 consecutive u32 words (16 base pairs per word, 55 bases = 4 words)
//   Gene at index i uses words: gene_sequences[i*4 .. i*4+3]

// Read a base pair from a specific gene in a genome
// genome_slot: index in genomes array
// gene_index: which gene slot (0-199)
// base_index: which base within the gene (0-54)
fn read_gene_base(genome_slot: u32, gene_index: u32, base_index: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME || base_index >= BASE_PAIRS_PER_GENE {
        return 0u;
    }
    // Calculate word index: each gene uses 4 words, base pairs are packed 16 per word
    let word_within_gene = base_index / 16u; // Which of the 4 words (0-3)
    let word_index = gene_index * WORDS_PER_GENE + word_within_gene;
    let bit_offset = (base_index & 15u) * 2u; // Position within word (0-30, step 2)
    if word_index >= GENOME_WORD_COUNT {
        return 0u;
    }
    let word = genomes[genome_slot].gene_sequences[word_index];
    return (word >> bit_offset) & 0x3u; // Extract 2-bit base pair
}

// Write a base pair to a specific gene in a genome
fn write_gene_base(genome_slot: u32, gene_index: u32, base_index: u32, value: u32) {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME || base_index >= BASE_PAIRS_PER_GENE {
        return;
    }
    let word_within_gene = base_index / 16u;
    let word_index = gene_index * WORDS_PER_GENE + word_within_gene;
    let bit_offset = (base_index & 15u) * 2u;
    if word_index >= GENOME_WORD_COUNT {
        return;
    }
    let mask = ~(0x3u << bit_offset); // Clear the 2 bits
    let word = genomes[genome_slot].gene_sequences[word_index];
    genomes[genome_slot].gene_sequences[word_index] = (word & mask) | ((value & 0x3u) << bit_offset);
}

// Get the gene ID at a specific index
fn get_gene_id(genome_slot: u32, gene_index: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME {
        return 0u;
    }
    return genomes[genome_slot].gene_ids[gene_index];
}

// Set the gene ID at a specific index
fn set_gene_id(genome_slot: u32, gene_index: u32, gene_id: u32) {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME {
        return;
    }
    genomes[genome_slot].gene_ids[gene_index] = gene_id;
}

// Find the index of a gene with a specific ID, returns MAX_GENES_PER_GENOME if not found
fn find_gene_index(genome_slot: u32, gene_id: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) {
        return MAX_GENES_PER_GENOME;
    }
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        if genomes[genome_slot].gene_ids[i] == gene_id {
            return i;
        }
    }
    return MAX_GENES_PER_GENOME;
}

// Count the number of active genes (non-zero gene IDs)
fn count_active_genes(genome_slot: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) {
        return 0u;
    }
    var count: u32 = 0u;
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        if genomes[genome_slot].gene_ids[i] != 0u {
            count = count + 1u;
        }
    }
    return count;
}

// ============================================
// Genome Operations
// ============================================

// TODO: PERFORMANCE OPTIMIZATION - This function is slow when called many times
// Consider creating a separate parallel compute shader (copy_genomes.wgsl) that processes
// multiple genome copies in parallel using workgroup_size(64) or similar.
// Each thread can copy a portion of the genome (e.g., thread 0 copies gene_ids[0-63], etc.)
fn copy_genome(dest_slot: u32, src_slot: u32) {
    if dest_slot >= arrayLength(&genomes) || src_slot >= arrayLength(&genomes) {
        return;
    }
    // Copy gene IDs
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        genomes[dest_slot].gene_ids[i] = genomes[src_slot].gene_ids[i];
    }
    // Copy gene sequences (all base pair data)
    for (var i: u32 = 0u; i < GENOME_WORD_COUNT; i = i + 1u) {
        genomes[dest_slot].gene_sequences[i] = genomes[src_slot].gene_sequences[i];
    }
}

// Compare two genes by their base pairs
fn compare_genes(genome_slot_a: u32, gene_index_a: u32, genome_slot_b: u32, gene_index_b: u32) -> f32 {
    var diff: u32 = 0u;
    for (var i: u32 = 0u; i < BASE_PAIRS_PER_GENE; i = i + 1u) {
        let base_a = read_gene_base(genome_slot_a, gene_index_a, i);
        let base_b = read_gene_base(genome_slot_b, gene_index_b, i);
        if base_a != base_b {
            diff = diff + 1u;
        }
    }
    return f32(diff);
}

// Compare two genomes - matches CPU logic exactly
// Returns compatibility distance (lower = more similar)
fn compare_genomes(genome_slot_a: u32, genome_slot_b: u32) -> f32 {
    if genome_slot_a >= arrayLength(&genomes) || genome_slot_b >= arrayLength(&genomes) {
        return 10000.0; // Very different if invalid
    }

    var gene_difference: u32 = 0u;
    var base_difference: f32 = 0.0;
    
    // Check genes in genome A
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        let gene_id_a = get_gene_id(genome_slot_a, i);
        if gene_id_a == 0u {
            continue; // Skip empty slots
        }
        let gene_index_b = find_gene_index(genome_slot_b, gene_id_a);
        if gene_index_b >= MAX_GENES_PER_GENOME {
            // Gene exists in A but not in B
            gene_difference = gene_difference + 1u;
        } else {
            // Both have the gene, compare base pairs
            base_difference = base_difference + compare_genes(genome_slot_a, i, genome_slot_b, gene_index_b);
        }
    }
    
    // Check genes in genome B that don't exist in A
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        let gene_id_b = get_gene_id(genome_slot_b, i);
        if gene_id_b == 0u {
            continue; // Skip empty slots
        }
        let gene_index_a = find_gene_index(genome_slot_a, gene_id_b);
        if gene_index_a >= MAX_GENES_PER_GENOME {
            // Gene exists in B but not in A
            gene_difference = gene_difference + 1u;
        }
    }

    return f32(gene_difference) * GENE_DIFFERENCE_SCALAR + base_difference * BASE_DIFFERENCE_SCALAR;
}

// Generate a random genome
fn random_genome(genome_slot: u32, seed: u32, num_genes: u32) {
    if genome_slot >= arrayLength(&genomes) {
        return;
    }
    
    // Clear genome
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        genomes[genome_slot].gene_ids[i] = 0u;
    }
    for (var i: u32 = 0u; i < GENOME_WORD_COUNT; i = i + 1u) {
        genomes[genome_slot].gene_sequences[i] = 0u;
    }
    
    // Generate random genes
    let actual_num_genes = min(num_genes, MAX_GENES_PER_GENOME);
    let next_gene_id_start = atomicAdd(&next_gene_id.value, actual_num_genes);

    for (var gene_idx: u32 = 0u; gene_idx < actual_num_genes; gene_idx = gene_idx + 1u) {
        let gene_id = next_gene_id_start + gene_idx + 1u;
        set_gene_id(genome_slot, gene_idx, gene_id);
        
        // Generate random base pairs for this gene
        for (var base_idx: u32 = 0u; base_idx < BASE_PAIRS_PER_GENE; base_idx = base_idx + 1u) {
            let random_value = rand(vec2<u32>(seed + gene_idx * 17u + base_idx * 31u, genome_slot * 97u + base_idx * 13u));
            let base = u32(random_value * 4.0);
            write_gene_base(genome_slot, gene_idx, base_idx, base);
        }
    }
}

// Convert probability to threshold (0.0-1.0 -> 0 to u32::MAX)
fn probability_to_threshold(probability: f32) -> u32 {
    let clamped = clamp(probability, 0.0, 1.0);
    return u32(clamped * 4294967295.0); // u32::MAX as f32
}

// Mutate a genome - matches CPU logic
// TODO: PERFORMANCE OPTIMIZATION - This function can be slow for many genomes
// Consider creating a separate parallel compute shader (mutate_genomes.wgsl) that processes
// multiple genomes in parallel. Each thread handles one genome's mutation.
fn mutate_genome(genome_slot: u32, seed: u32) {
    if genome_slot >= arrayLength(&genomes) {
        return;
    }

    let clone_gene_threshold = probability_to_threshold(CLONE_GENE_CHANCE);
    let insert_gene_threshold = probability_to_threshold(INSERT_GENE_CHANCE);
    let delete_gene_threshold = probability_to_threshold(DELETE_GENE_CHANCE);
    
    // Track genes to clone/insert (we'll process these after)
    var genes_to_clone: array<u32, MAX_GENES_PER_GENOME>;
    var clone_count: u32 = 0u;
    var genes_to_insert: array<u32, MAX_GENES_PER_GENOME>;
    var insert_count: u32 = 0u;
    
    // First pass: process existing genes
    var gene_idx: u32 = 0u;
    while gene_idx < MAX_GENES_PER_GENOME {
        let gene_id = get_gene_id(genome_slot, gene_idx);
        if gene_id == 0u {
            gene_idx = gene_idx + 1u;
            continue;
        }
        
        // Check for gene cloning
        let clone_rand = u32(rand(vec2<u32>(seed + gene_idx * 17u, genome_slot * 97u + gene_idx * 13u)) * 4294967295.0);
        var should_clone = false;
        if clone_rand < clone_gene_threshold {
            should_clone = true;
        }
        
        // Check for random gene insertion
        let insert_rand = u32(rand(vec2<u32>(seed + gene_idx * 19u, genome_slot * 101u + gene_idx * 17u)) * 4294967295.0);
        var should_insert_random = false;
        if insert_rand < insert_gene_threshold {
            should_insert_random = true;
        }
        
        // Mutate base pairs in this gene
        // Use simplified Poisson approximation: average mutations per gene
        let mutate_avg = MUTATE_BASE_CHANCE * f32(BASE_PAIRS_PER_GENE);
        let delete_avg = DELETE_BASE_CHANCE * f32(BASE_PAIRS_PER_GENE);
        let insert_avg = INSERT_BASE_CHANCE * f32(BASE_PAIRS_PER_GENE + 1u);
        
        // Approximate Poisson with binomial for small probabilities
        // For each base, check if it should mutate
        for (var base_idx: u32 = 0u; base_idx < BASE_PAIRS_PER_GENE; base_idx = base_idx + 1u) {
            let base_seed = vec2<u32>(seed + gene_idx * 31u + base_idx * 17u, genome_slot * 131u + base_idx);
            
            // Mutate base
            let mutate_rand = rand(base_seed);
            if mutate_rand < MUTATE_BASE_CHANCE {
                let new_base = u32(rand(vec2<u32>(seed * 3u + base_idx * 7u, genome_slot * 211u + gene_idx)) * 4.0);
                write_gene_base(genome_slot, gene_idx, base_idx, new_base);
            }
            
            // Delete base (swap with random base to simulate deletion)
            let delete_rand = rand(vec2<u32>(base_seed.x + 1u, base_seed.y + 1u));
            if delete_rand < DELETE_BASE_CHANCE && base_idx < BASE_PAIRS_PER_GENE - 1u {
                // Swap with next base to simulate deletion effect
                let next_base = read_gene_base(genome_slot, gene_idx, base_idx + 1u);
                write_gene_base(genome_slot, gene_idx, base_idx, next_base);
            }
            
            // Insert base (shift and insert)
            let insert_rand = rand(vec2<u32>(base_seed.x + 2u, base_seed.y + 2u));
            if insert_rand < INSERT_BASE_CHANCE && base_idx < BASE_PAIRS_PER_GENE - 1u {
                // Shift bases right and insert random base
                let new_base = u32(rand(vec2<u32>(seed * 5u + base_idx * 11u, genome_slot * 311u + gene_idx)) * 4.0);
                // Shift remaining bases (simplified - just overwrite next position)
                write_gene_base(genome_slot, gene_idx, base_idx + 1u, new_base);
            }
        }
        
        // Check if gene should be deleted
        let delete_rand = u32(rand(vec2<u32>(seed + gene_idx * 23u, genome_slot * 137u + gene_idx * 19u)) * 4294967295.0);
        var should_delete = false;
        if delete_rand < delete_gene_threshold {
            should_delete = true;
        }

        if should_delete {
            // Delete gene by clearing its ID
            set_gene_id(genome_slot, gene_idx, 0u);
            // Clear gene sequence
            for (var base_idx: u32 = 0u; base_idx < BASE_PAIRS_PER_GENE; base_idx = base_idx + 1u) {
                write_gene_base(genome_slot, gene_idx, base_idx, 0u);
            }
            gene_idx = gene_idx + 1u;
            continue;
        }
        
        // Store gene for cloning if needed
        if should_clone && clone_count < MAX_GENES_PER_GENOME {
            genes_to_clone[clone_count] = gene_idx;
            clone_count = clone_count + 1u;
        }
        
        // Store for random insertion if needed
        if should_insert_random && insert_count < MAX_GENES_PER_GENOME {
            genes_to_insert[insert_count] = gene_idx; // Store position hint
            insert_count = insert_count + 1u;
        }

        gene_idx = gene_idx + 1u;
    }
    
    // Second pass: insert cloned genes
    for (var i: u32 = 0u; i < clone_count; i = i + 1u) {
        let src_gene_idx = genes_to_clone[i];
        let src_gene_id = get_gene_id(genome_slot, src_gene_idx);
        if src_gene_id == 0u {
            continue;
        }
        
        // Find empty slot
        var empty_slot = MAX_GENES_PER_GENOME;
        for (var j: u32 = 0u; j < MAX_GENES_PER_GENOME; j = j + 1u) {
            if get_gene_id(genome_slot, j) == 0u {
                empty_slot = j;
                break;
            }
        }

        if empty_slot < MAX_GENES_PER_GENOME {
            // Generate new gene ID
            let new_gene_id = atomicAdd(&next_gene_id.value, 1u) + 1u;
            set_gene_id(genome_slot, empty_slot, new_gene_id);
            
            // Copy gene sequence
            for (var base_idx: u32 = 0u; base_idx < BASE_PAIRS_PER_GENE; base_idx = base_idx + 1u) {
                let base = read_gene_base(genome_slot, src_gene_idx, base_idx);
                write_gene_base(genome_slot, empty_slot, base_idx, base);
            }
        }
    }
    
    // Third pass: insert random new genes
    for (var i: u32 = 0u; i < insert_count; i = i + 1u) {
        // Find empty slot
        var empty_slot = MAX_GENES_PER_GENOME;
        for (var j: u32 = 0u; j < MAX_GENES_PER_GENOME; j = j + 1u) {
            if get_gene_id(genome_slot, j) == 0u {
                empty_slot = j;
                break;
            }
        }

        if empty_slot < MAX_GENES_PER_GENOME {
            // Generate new gene ID
            let new_gene_id = atomicAdd(&next_gene_id.value, 1u) + 1u;
            set_gene_id(genome_slot, empty_slot, new_gene_id);
            
            // Generate random gene (55 bases, matching INITIAL_HOX_SIZE from CPU)
            for (var base_idx: u32 = 0u; base_idx < BASE_PAIRS_PER_GENE; base_idx = base_idx + 1u) {
                let random_value = rand(vec2<u32>(seed + empty_slot * 29u + base_idx * 37u, genome_slot * 149u + base_idx * 23u));
                let base = u32(random_value * 4.0);
                write_gene_base(genome_slot, empty_slot, base_idx, base);
            }
        }
    }
}

// ============================================
// Species Management
// ============================================

fn create_species(child_slot: u32) -> vec2<u32> {
    let slot = allocate_species_slot();
    if slot >= MAX_SPECIES_CAPACITY {
        return vec2<u32>(MAX_SPECIES_CAPACITY, 0u);
    }
    let species_id = atomicAdd(&next_species_id.value, 1u);
    if slot < arrayLength(&species_entries) {
        species_entries[slot].species_id = species_id;
        species_entries[slot].mascot_lifeform_slot = child_slot;
        atomicStore(&species_entries[slot].member_count, 1u);
        species_entries[slot].flags = SPECIES_FLAG_ACTIVE;
        
        // Copy mascot genome
        if child_slot < LIFEFORM_CAPACITY && child_slot < arrayLength(&lifeforms) {
            let lifeform_genome_slot = lifeforms[child_slot].grn_descriptor_slot; // Reusing this field for genome slot
            if lifeform_genome_slot < arrayLength(&genomes) {
                copy_genome_to_species(slot, lifeform_genome_slot);
            }
        }
    }
    atomicAdd(&species_counter.value, 1u);
    return vec2<u32>(slot, species_id);
}

fn copy_genome_to_species(species_slot: u32, genome_slot: u32) {
    if species_slot >= arrayLength(&species_entries) || genome_slot >= arrayLength(&genomes) {
        return;
    }
    // Copy gene IDs
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        species_entries[species_slot].mascot_genome.gene_ids[i] = genomes[genome_slot].gene_ids[i];
    }
    // Copy gene sequences
    for (var i: u32 = 0u; i < GENOME_WORD_COUNT; i = i + 1u) {
        species_entries[species_slot].mascot_genome.gene_sequences[i] = genomes[genome_slot].gene_sequences[i];
    }
}

fn release_species(slot: u32, species_id: u32) {
    if slot >= arrayLength(&species_entries) {
        return;
    }
    species_entries[slot].species_id = 0u;
    species_entries[slot].mascot_lifeform_slot = 0u;
    atomicStore(&species_entries[slot].member_count, 0u);
    species_entries[slot].flags = 0u;
    recycle_species_slot(slot);
    loop {
        let current = atomicLoad(&species_counter.value);
        if current == 0u {
            break;
        }
        let exchange = atomicCompareExchangeWeak(&species_counter.value, current, current - 1u);
        if exchange.exchanged {
            break;
        }
    }
}

fn assign_species(child_slot: u32, parent_slot: u32) -> vec2<u32> {
    if parent_slot < LIFEFORM_CAPACITY && parent_slot < arrayLength(&lifeforms) {
        // Don't copy the struct - access fields directly (cell_count is atomic and can't be copied)
        if (lifeforms[parent_slot].flags & LIFEFORM_FLAG_ACTIVE) != 0u {
            let child_genome_slot = lifeforms[child_slot].grn_descriptor_slot;
            
            // Compare with parent's species mascot genome
            let species_slot = lifeforms[parent_slot].species_slot;
            if species_slot < arrayLength(&species_entries) {
                // Compare child genome with species mascot genome
                var compatibility_distance: f32 = 0.0;

                if child_genome_slot < arrayLength(&genomes) {
                    // Compare with mascot genome stored in species entry
                    compatibility_distance = compare_genomes_with_species_mascot(child_genome_slot, species_slot);
                }

                if compatibility_distance >= COMPATABILITY_DISTANCE_THRESHOLD {
                    return create_species(child_slot);
                } else {
                    atomicAdd(&species_entries[species_slot].member_count, 1u);
                    species_entries[species_slot].flags = SPECIES_FLAG_ACTIVE;
                    return vec2<u32>(species_slot, lifeforms[parent_slot].species_id);
                }
            }
        }
    }
    return create_species(child_slot);
}

// Compare a genome with a species mascot genome
fn compare_genomes_with_species_mascot(genome_slot: u32, species_slot: u32) -> f32 {
    if genome_slot >= arrayLength(&genomes) || species_slot >= arrayLength(&species_entries) {
        return 10000.0;
    }

    var gene_difference: u32 = 0u;
    var base_difference: f32 = 0.0;
    
    // Check genes in genome
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        let gene_id = get_gene_id(genome_slot, i);
        if gene_id == 0u {
            continue;
        }
        let mascot_gene_idx = find_gene_index_in_species(species_slot, gene_id);
        if mascot_gene_idx >= MAX_GENES_PER_GENOME {
            // Gene exists in child but not in mascot
            gene_difference = gene_difference + 1u;
        } else {
            // Both have the gene, compare base pairs
            base_difference = base_difference + compare_gene_with_species_mascot(genome_slot, i, species_slot, mascot_gene_idx);
        }
    }
    
    // Check genes in mascot genome that don't exist in child
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        let mascot_gene_id = species_entries[species_slot].mascot_genome.gene_ids[i];
        if mascot_gene_id == 0u {
            continue;
        }
        let child_gene_idx = find_gene_index(genome_slot, mascot_gene_id);
        if child_gene_idx >= MAX_GENES_PER_GENOME {
            // Gene exists in mascot but not in child
            gene_difference = gene_difference + 1u;
        }
    }

    return f32(gene_difference) * GENE_DIFFERENCE_SCALAR + base_difference * BASE_DIFFERENCE_SCALAR;
}

fn compare_gene_with_species_mascot(genome_slot: u32, gene_idx: u32, species_slot: u32, mascot_gene_idx: u32) -> f32 {
    var diff: u32 = 0u;
    for (var i: u32 = 0u; i < BASE_PAIRS_PER_GENE; i = i + 1u) {
        let base_a = read_gene_base(genome_slot, gene_idx, i);
        let base_b = read_gene_base_from_species(species_slot, mascot_gene_idx, i);
        if base_a != base_b {
            diff = diff + 1u;
        }
    }
    return f32(diff);
}

fn read_gene_base_from_species(species_slot: u32, gene_index: u32, base_index: u32) -> u32 {
    if species_slot >= arrayLength(&species_entries) || gene_index >= MAX_GENES_PER_GENOME || base_index >= BASE_PAIRS_PER_GENE {
        return 0u;
    }
    let gene_base_word = base_index / 16u;
    let word_index = gene_index * WORDS_PER_GENE + gene_base_word;
    let offset = (base_index & 15u) * 2u;
    if word_index >= GENOME_WORD_COUNT {
        return 0u;
    }
    let word = species_entries[species_slot].mascot_genome.gene_sequences[word_index];
    return (word >> offset) & 0x3u;
}

fn find_gene_index_in_species(species_slot: u32, gene_id: u32) -> u32 {
    if species_slot >= arrayLength(&species_entries) {
        return MAX_GENES_PER_GENOME;
    }
    for (var i: u32 = 0u; i < MAX_GENES_PER_GENOME; i = i + 1u) {
        if species_entries[species_slot].mascot_genome.gene_ids[i] == gene_id {
            return i;
        }
    }
    return MAX_GENES_PER_GENOME;
}

fn release_lifeform(lifeform_slot: u32) {
    if lifeform_slot >= LIFEFORM_CAPACITY {
        return;
    }
    // Don't copy the struct - access fields directly (cell_count is atomic and can't be copied)
    if (lifeforms[lifeform_slot].flags & LIFEFORM_FLAG_ACTIVE) == 0u {
        return;
    }

    let species_slot = lifeforms[lifeform_slot].species_slot;
    let lifeform_id = lifeforms[lifeform_slot].lifeform_id;

    if species_slot < arrayLength(&species_entries) {
        let previous = atomicSub(&species_entries[species_slot].member_count, 1u);
        if previous <= 1u {
            release_species(species_slot, lifeforms[lifeform_slot].species_id);
        }
    }

    if lifeform_slot < arrayLength(&lifeforms) {
        lifeforms[lifeform_slot].lifeform_id = 0u;
        lifeforms[lifeform_slot].species_slot = 0u;
        lifeforms[lifeform_slot].species_id = 0u;
        lifeforms[lifeform_slot].gene_count = 0u;
        lifeforms[lifeform_slot].rng_state = 0u;
        lifeforms[lifeform_slot].cell_count = 0u;
        lifeforms[lifeform_slot].flags = 0u;
        lifeforms[lifeform_slot].first_cell_slot = 0u;
        lifeforms[lifeform_slot].grn_descriptor_slot = 0u;
        lifeforms[lifeform_slot].grn_unit_offset = 0u;
        lifeforms[lifeform_slot].grn_timer = 0u;
        lifeforms[lifeform_slot]._pad = 0u;
        lifeforms[lifeform_slot]._pad2 = vec2<u32>(0u, 0u);
    }

    recycle_lifeform_slot(lifeform_slot);
    loop {
        let current = atomicLoad(&lifeform_counter.value);
        if current == 0u {
            break;
        }
        let exchange = atomicCompareExchangeWeak(&lifeform_counter.value, current, current - 1u);
        if exchange.exchanged {
            break;
        }
    }
}

// ============================================
// Lifeform Creation
// ============================================

fn random_position(seed: vec2<u32>) -> vec2<f32> {
    let width = uniforms.bounds.z - uniforms.bounds.x;
    let height = uniforms.bounds.w - uniforms.bounds.y;
    let rx = rand(seed);
    let ry = rand(seed.yx);
    return vec2<f32>(
        uniforms.bounds.x + rx * width,
        uniforms.bounds.y + ry * height,
    );
}

fn initialise_lifeform_state(slot: u32, lifeform_id: u32) {
    if slot >= arrayLength(&lifeforms) {
        return;
    }
    lifeforms[slot].lifeform_id = lifeform_id;
    lifeforms[slot].first_cell_slot = 0u;
    // Initialize cell_count to 0 atomically
    loop {
        let current = atomicLoad(&lifeforms[slot].cell_count);
        let exchange = atomicCompareExchangeWeak(&lifeforms[slot].cell_count, current, 0u);
        if exchange.exchanged {
            break;
        }
    }
    lifeforms[slot].grn_descriptor_slot = slot; // Store genome slot here temporarily
    lifeforms[slot].grn_unit_offset = slot * MAX_GRN_REGULATORY_UNITS;
    lifeforms[slot].flags = LIFEFORM_FLAG_ACTIVE;
    lifeforms[slot].grn_timer = 0u;
    lifeforms[slot]._pad = 0u;
    lifeforms[slot]._pad2 = vec2<u32>(0u, 0u);
    for (var i: u32 = 0u; i < MAX_GRN_STATE_SIZE; i = i + 1u) {
        lifeforms[slot].grn_state[i] = 0.0;
    }
}

fn initialise_grn(slot: u32) {
    if slot >= arrayLength(&grn_descriptors) {
        return;
    }
    grn_descriptors[slot].receptor_count = 0u;
    grn_descriptors[slot].unit_count = 0u;
    grn_descriptors[slot].state_stride = 0u;
    grn_descriptors[slot].unit_offset = slot * MAX_GRN_REGULATORY_UNITS;
}

fn create_lifeform_cell(
    parent_index: u32,
    seed: vec2<u32>,
) {
    var lifeform_slot = LIFEFORM_CAPACITY;
    var lifeform_id = 0u;
    var is_new_lifeform = false;
    if parent_index < CELL_CAPACITY {
        lifeform_slot = cells[parent_index].lifeform_slot;
        lifeform_id = lifeforms[lifeform_slot].lifeform_id;
    } else {
        lifeform_slot = allocate_lifeform_slot();
        is_new_lifeform = true;
        // Early return if we can't allocate a lifeform slot
        if lifeform_slot >= LIFEFORM_CAPACITY {
            return;
        }
        lifeform_id = atomicAdd(&next_lifeform_id.value, 1u);
        lifeforms[lifeform_slot].lifeform_id = lifeform_id;
    }

    let genome_slot = lifeform_slot; // Use lifeform slot as genome slot for now

    /*if parent_slot < LIFEFORM_CAPACITY && (lifeforms[parent_slot].flags & LIFEFORM_FLAG_ACTIVE) != 0u {
        let parent_genome_slot = lifeforms[parent_slot].grn_descriptor_slot;
        copy_genome(genome_slot, parent_genome_slot);
        mutate_genome(genome_slot, lifeform_id + 11u);
    } else {
        // Generate random genome
        let num_genes = u32(rand(seed) * 18.0) + 2u; // 2-20 genes
        random_genome(genome_slot, lifeform_id + 1u, num_genes);
    }*/

    //let species_info = assign_species(lifeform_slot, parent_slot);
    let species_info = vec2<u32>(0u, 0u);
    let species_slot = species_info.x;
    let species_id = species_info.y;

    let valid_species = species_slot < MAX_SPECIES_CAPACITY;
    if valid_species {
        lifeforms[lifeform_slot].species_slot = species_slot;
        lifeforms[lifeform_slot].species_id = species_id;
    } else {
        lifeforms[lifeform_slot].species_slot = 0u;
        lifeforms[lifeform_slot].species_id = 0u;
    }
    lifeforms[lifeform_slot].rng_state = lifeform_id * 1664525u + 1013904223u;
    // For newly allocated lifeforms, initialize cell_count and increment lifeform_counter
    if is_new_lifeform {
        // Initialize cell_count to 0 atomically
        loop {
            let current = atomicLoad(&lifeforms[lifeform_slot].cell_count);
            let exchange = atomicCompareExchangeWeak(
                &lifeforms[lifeform_slot].cell_count,
                current,
                0u,
            );
            if exchange.exchanged {
                break;
            }
        }
        atomicAdd(&lifeform_counter.value, 1u);
    }
    lifeforms[lifeform_slot].flags = LIFEFORM_FLAG_ACTIVE;
    lifeforms[lifeform_slot]._pad = 0u;
    lifeforms[lifeform_slot].grn_descriptor_slot = genome_slot; // Store genome slot here

    //initialise_lifeform_state(lifeform_slot, lifeform_id);
    //initialise_grn(lifeform_slot);

    var new_cell: Cell;

    new_cell.is_alive = 1u;
    new_cell.lifeform_slot = lifeform_slot;
    new_cell.link_count = 0u;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        new_cell.link_indices[i] = 0u;
    }
    new_cell._pad = 0u;

    if parent_index < CELL_CAPACITY {
        let parent_cell = cells[parent_index];
        let offset_seed = vec2<u32>(parent_index * 97u + 13u, parent_cell.lifeform_slot * 211u + 17u);
        let angle = rand(offset_seed) * 6.2831853;
        let distance = parent_cell.radius * 1.4;
        let child_position = parent_cell.pos + vec2<f32>(cos(angle), sin(angle)) * distance;
        let cell_wall_thickness = parent_cell.cell_wall_thickness + rand(vec2<u32>(seed.x * 17u + 7u, seed.y * 29u + 3u)) * 0.05;

        new_cell.pos = child_position;
        new_cell.prev_pos = child_position;
        new_cell.random_force = vec2<f32>(0.0, 0.0);
        new_cell.radius = parent_cell.radius;
        new_cell.energy = parent_cell.energy * 0.5;
        new_cell.cell_wall_thickness = clamp(cell_wall_thickness, 0.1, 0.8);
        new_cell.generation = parent_cell.generation + 1u;
    } else {
        let position = random_position(vec2<u32>(seed.x * 97u + 11u, seed.y * 131u + 23u));
        let radius = 0.5 + rand(vec2<u32>(seed.x * 17u + 7u, seed.y * 29u + 3u)) * 5.0;
        let energy = 60.0 + rand(vec2<u32>(seed.x * 53u + 5u, seed.y * 71u + 19u)) * 80.0;
        let cell_wall_thickness = 0.2;

        new_cell.pos = position;
        new_cell.prev_pos = position;
        new_cell.random_force = vec2<f32>(0.0, 0.0);
        new_cell.radius = radius;
        new_cell.energy = energy;
        new_cell.cell_wall_thickness = cell_wall_thickness;
        new_cell.generation = 0u;
    }

    // For division, spawn child immediately and create a link; for other uses, go through the spawn buffer.
    if parent_index < CELL_CAPACITY {
        // Allocate a free cell slot
        loop {
            let free_prev = atomicLoad(&cell_free_list.count);
            if free_prev == 0u {
                // No free slots available - abort this division
                return;
            }
            let free_desired = free_prev - 1u;
            let free_exchange = atomicCompareExchangeWeak(
                &cell_free_list.count,
                free_prev,
                free_desired,
            );
            if free_exchange.old_value == free_prev && free_exchange.exchanged {
                if free_desired >= arrayLength(&cell_free_list.indices) {
                    atomicAdd(&cell_free_list.count, 1u);
                    break;
                }
                let slot_index = cell_free_list.indices[free_desired];
                if slot_index >= arrayLength(&cells) {
                    atomicAdd(&cell_free_list.count, 1u);
                    break;
                }
                var slot_cell = cells[slot_index];
                if slot_cell.is_alive != 0u {
                    atomicAdd(&cell_free_list.count, 1u);
                    break;
                }
                let previous_generation = slot_cell.generation;
                var verify_slot_cell = cells[slot_index];
                if verify_slot_cell.is_alive != 0u || verify_slot_cell.generation != previous_generation {
                    atomicAdd(&cell_free_list.count, 1u);
                    break;
                }

                var spawned_cell = new_cell;
                spawned_cell.generation = previous_generation;
                spawned_cell.is_alive = 1u;
                cells[slot_index] = spawned_cell;

                atomicAdd(&cell_counter.value, 1u);

                let lf_idx = spawned_cell.lifeform_slot;
                if lf_idx < LIFEFORM_CAPACITY {
                    if lf_idx < arrayLength(&lifeforms) {
                        // Atomically increment cell_count
                        let previous = atomicAdd(&lifeforms[lf_idx].cell_count, 1u);
                        if previous == 0u {
                            lifeforms[lf_idx].first_cell_slot = slot_index;
                        }
                    }
                }

                // Create a link between parent and child if possible
                var link_created = false;
                loop {
                    let link_prev = atomicLoad(&link_free_list.count);
                    if link_prev == 0u {
                        break;
                    }
                    let link_desired = link_prev - 1u;
                    let link_exchange = atomicCompareExchangeWeak(
                        &link_free_list.count,
                        link_prev,
                        link_desired,
                    );
                    if link_exchange.old_value == link_prev && link_exchange.exchanged {
                        if link_desired < arrayLength(&link_free_list.indices) {
                            let link_slot = link_free_list.indices[link_desired];
                            if link_slot >= arrayLength(&links) {
                                atomicAdd(&link_free_list.count, 1u);
                                break;
                            }
                            if parent_index >= arrayLength(&cells) || slot_index >= arrayLength(&cells) || parent_index == slot_index {
                                atomicAdd(&link_free_list.count, 1u);
                                break;
                            }
                            let parent_cell_link = cells[parent_index];
                            if parent_cell_link.is_alive == 0u {
                                atomicAdd(&link_free_list.count, 1u);
                                break;
                            }
                            let rest_length = parent_cell_link.radius + spawned_cell.radius;

                            links[link_slot].a = parent_index;
                            links[link_slot].b = slot_index;
                            links[link_slot].flags = LINK_FLAG_ALIVE | LINK_FLAG_ADHESIVE;
                            links[link_slot].generation_a = parent_cell_link.generation;
                            links[link_slot].rest_length = rest_length;
                            links[link_slot].stiffness = 0.6;
                            links[link_slot].energy_transfer_rate = 0.0;
                            links[link_slot].generation_b = spawned_cell.generation;
                            
                            // Add link to both cells' link_indices arrays
                            add_link_to_cell(parent_index, link_slot);
                            add_link_to_cell(slot_index, link_slot);

                            link_created = true;
                            break;
                        }
                        atomicAdd(&link_free_list.count, 1u);
                    }
                }
                
                // Redistribute parent cell's existing links to parent or child based on proximity
                if link_created {
                    redistribute_cell_links(parent_index, slot_index);
                }
                
                break;
            }
        }
    } else {
        if !queue_spawn_cell(new_cell) {
            if is_new_lifeform {
                recycle_lifeform_slot(lifeform_slot);
            }
            return;
        }
    }
}


// Optimized: distribute population maintenance across threads instead of running serially on thread 0
// Each thread checks if it should create a lifeform based on its index
fn ensure_minimum_population_parallel(thread_index: u32, total_threads: u32) {
    let alive = atomicLoad(&cell_counter.value);
    if alive >= MIN_ACTIVE_CELLS {
        return;
    }
    let deficit = MIN_ACTIVE_CELLS - alive;
    
    // Distribute deficit across threads - each thread handles its portion
    let lifeforms_per_thread = (deficit + total_threads - 1u) / total_threads;
    let start_index = thread_index * lifeforms_per_thread;
    let end_index = min(start_index + lifeforms_per_thread, deficit);

    for (var i: u32 = start_index; i < end_index; i = i + 1u) {
        // Check again if we've reached minimum (another thread might have created lifeforms)
        let current_alive = atomicLoad(&cell_counter.value);
        if current_alive >= MIN_ACTIVE_CELLS {
            break;
        }
        create_lifeform_cell(CELL_CAPACITY, vec2<u32>(i + current_alive * 13u + 1u, 0u));
    }
}
