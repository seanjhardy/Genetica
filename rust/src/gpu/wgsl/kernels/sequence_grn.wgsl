@include src/gpu/wgsl/types.wgsl;
@include src/gpu/wgsl/constants.wgsl;

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(14)
var<storage, read_write> lifeforms: array<Lifeform>;

@group(0) @binding(17)
var<storage, read_write> genomes: array<GenomeEntry>;

@group(0) @binding(12)
var<storage, read_write> grn_descriptors: array<GrnDescriptor>;

@group(0) @binding(13)
var<storage, read_write> grn_units: array<CompiledRegulatoryUnit>;

const MIN_GENE_BASES: u32 = 20u;
const EMBEDDING_DIMENSIONS: u32 = 3u;

// Helper functions for reading base pairs
fn read_gene_base(genome_slot: u32, gene_index: u32, base_index: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME || base_index >= BASE_PAIRS_PER_GENE {
        return 0u;
    }
    let gene_base_word = base_index / 16u;
    let word_index = gene_index * WORDS_PER_GENE + gene_base_word;
    let offset = (base_index & 15u) * 2u;
    if word_index >= GENOME_WORD_COUNT {
        return 0u;
    }
    let word = genomes[genome_slot].gene_sequences[word_index];
    return (word >> offset) & 0x3u;
}

fn get_gene_id(genome_slot: u32, gene_index: u32) -> u32 {
    if genome_slot >= arrayLength(&genomes) || gene_index >= MAX_GENES_PER_GENOME {
        return 0u;
    }
    return genomes[genome_slot].gene_ids[gene_index];
}

// Sequence reader state
struct SequenceReader {
    genome_slot: u32,
    gene_index: u32,
    position: u32,
}

fn reader_new(genome_slot: u32, gene_index: u32) -> SequenceReader {
    return SequenceReader(genome_slot, gene_index, 0u);
}

fn reader_read_base(reader: ptr<function, SequenceReader>) -> u32 {
    let pos = (*reader).position;
    if pos >= BASE_PAIRS_PER_GENE {
        return 255u; // Invalid
    }
    let base = read_gene_base((*reader).genome_slot, (*reader).gene_index, pos);
    (*reader).position = pos + 1u;
    return base;
}

fn reader_remaining(reader: SequenceReader) -> u32 {
    return BASE_PAIRS_PER_GENE - reader.position;
}

fn read_base_range(reader: ptr<function, SequenceReader>, length: u32) -> f32 {
    var result: f32 = 0.0;
    for (var i: u32 = 0u; i < length; i = i + 1u) {
        let base = reader_read_base(reader);
        if base >= 4u {
            return 0.0;
        }
        result = result + f32(base);
    }
    return result / (3.0 * f32(length));
}

fn read_unique_base_range(reader: ptr<function, SequenceReader>, length: u32) -> f32 {
    var result: f32 = 0.0;
    for (var i: u32 = 0u; i < length; i = i + 1u) {
        let base = reader_read_base(reader);
        if base >= 4u {
            return 0.0;
        }
        result = result + f32(base) * pow(0.25, f32(i + 1u));
    }
    return result;
}

// GRN component structures (temporary, stored in function scope)
struct ReceptorData {
    receptor_type: u32,
    sign: bool,
    modifier: f32,
    embedding: vec3<f32>,
    extra: vec2<f32>,
}

struct PromoterData {
    promoter_type: u32, // 0 = Additive, 1 = Multiplicative
    sign: bool,
    modifier: f32,
    embedding: vec3<f32>,
}

struct FactorData {
    factor_type: u32,
    sign: bool,
    modifier: f32,
    embedding: vec3<f32>,
}

struct EffectorData {
    effector_type: u32,
    sign: bool,
    modifier: f32,
    embedding: vec3<f32>,
}

// Calculate affinity between two embeddings
fn calculate_affinity(emb1: vec3<f32>, sign1: bool, mod1: f32, emb2: vec3<f32>, sign2: bool, mod2: f32) -> f32 {
    let distance = length(emb1 - emb2);
    if distance > 0.2 { // BINDING_DISTANCE_THRESHOLD
        return 0.0;
    }
    let affinity_sign = select(-1.0, 1.0, sign1 == sign2);
    let affinity = affinity_sign * 
        (2.0 * abs(mod1 * mod2) * (0.2 - distance)) /
        (10.0 * distance + abs(mod1 * mod2));
    return affinity;
}

// Sequence a genome and compile it into GRN
@compute @workgroup_size(64)
fn sequence_and_compile(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lifeform_slot = global_id.x;
    if lifeform_slot >= LIFEFORM_CAPACITY || lifeform_slot >= arrayLength(&lifeforms) {
        return;
    }
    
    let lifeform = lifeforms[lifeform_slot];
    if (lifeform.flags & LIFEFORM_FLAG_ACTIVE) == 0u {
        return;
    }
    
    let genome_slot = lifeform.grn_descriptor_slot; // Reusing this field for genome slot
    if genome_slot >= arrayLength(&genomes) {
        return;
    }
    
    // Temporary storage for GRN components (limited by MAX constants)
    var receptors: array<ReceptorData, MAX_GRN_RECEPTOR_INPUTS>;
    var receptor_count: u32 = 0u;
    var effectors: array<EffectorData, 16u>; // Max effectors
    var effector_count: u32 = 0u;
    
    // Regulatory unit data
    var current_promoters: array<PromoterData, 8u>;
    var promoter_count: u32 = 0u;
    var current_factors: array<FactorData, 8u>;
    var factor_count: u32 = 0u;
    var reading_promoters: bool = true;
    
    var regulatory_units: array<array<PromoterData, 8u>, MAX_GRN_REGULATORY_UNITS>;
    var unit_promoter_counts: array<u32, MAX_GRN_REGULATORY_UNITS>;
    var unit_factor_counts: array<u32, MAX_GRN_REGULATORY_UNITS>;
    var unit_count: u32 = 0u;
    
    // Process each gene in the genome
    for (var gene_idx: u32 = 0u; gene_idx < MAX_GENES_PER_GENOME; gene_idx = gene_idx + 1u) {
        let gene_id = get_gene_id(genome_slot, gene_idx);
        if gene_id == 0u {
            continue;
        }
        
        var reader = reader_new(genome_slot, gene_idx);
        
        while reader_remaining(reader) >= MIN_GENE_BASES {
            // Read gene type (2 bases -> 0-16, mod 5 -> 0-4)
            let type_val_raw = read_base_range(&reader, 2u);
            if type_val_raw == 0.0 {
                break;
            }
            let type_val = u32(type_val_raw * 16.0) % 5u;
            
            let sign_base = reader_read_base(&reader);
            if sign_base >= 4u {
                break;
            }
            let sign = sign_base >= 2u;
            
            let active_base = reader_read_base(&reader);
            if active_base >= 4u {
                break;
            }
            let active = active_base >= 1u;
            
            if !active {
                continue;
            }
            
            // Read modifier (8 bases)
            let modifier = read_unique_base_range(&reader, 8u);
            if modifier == 0.0 {
                break;
            }
            
            // Read embedding (8 bases each for x, y, z)
            var embedding: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
            var embedding_valid: bool = true;
            for (var i: u32 = 0u; i < EMBEDDING_DIMENSIONS; i = i + 1u) {
                let emb_val = read_unique_base_range(&reader, 8u);
                if emb_val == 0.0 {
                    embedding_valid = false;
                    break;
                }
                embedding[i] = emb_val;
            }
            if !embedding_valid {
                break;
            }
            
            // Process based on type
            if type_val == 0u {
                // Receptor
                if receptor_count >= MAX_GRN_RECEPTOR_INPUTS {
                    break;
                }
                let receptor_types = array<u32, 6u>(0u, 1u, 2u, 3u, 4u, 5u);
                let sub_type_raw = read_unique_base_range(&reader, 3u);
                if sub_type_raw == 0.0 {
                    break;
                }
                let sub_type = u32(sub_type_raw * 64.0) % 6u;
                let receptor_type = receptor_types[sub_type];
                
                let extra_x = read_unique_base_range(&reader, 8u);
                let extra_y = read_unique_base_range(&reader, 8u);
                if extra_x == 0.0 || extra_y == 0.0 {
                    break;
                }
                
                receptors[receptor_count] = ReceptorData(
                    receptor_type,
                    sign,
                    modifier,
                    embedding,
                    vec2<f32>(extra_x, extra_y),
                );
                receptor_count = receptor_count + 1u;
            } else if type_val == 1u {
                // Effector
                if effector_count >= 16u {
                    break;
                }
                let effector_types = array<u32, 9u>(1u, 0u, 2u, 3u, 4u, 5u, 6u, 7u, 8u); // Die, Divide, Freeze, etc.
                let sub_type_raw = read_unique_base_range(&reader, 4u);
                if sub_type_raw == 0.0 {
                    break;
                }
                let sub_type = u32(sub_type_raw * 256.0) % 9u;
                let effector_type = effector_types[sub_type];
                
                effectors[effector_count] = EffectorData(
                    effector_type,
                    sign,
                    modifier,
                    embedding,
                );
                effector_count = effector_count + 1u;
            } else if type_val == 2u {
                // Promoter
                let additive_base = reader_read_base(&reader);
                if additive_base >= 4u {
                    break;
                }
                let additive = additive_base >= 1u;
                let promoter_type = select(1u, 0u, additive); // 0 = Additive, 1 = Multiplicative
                
                if !reading_promoters {
                    // Finish current regulatory unit
                    if promoter_count > 0u && factor_count > 0u && unit_count < MAX_GRN_REGULATORY_UNITS {
                        for (var i: u32 = 0u; i < promoter_count; i = i + 1u) {
                            regulatory_units[unit_count][i] = current_promoters[i];
                        }
                        unit_promoter_counts[unit_count] = promoter_count;
                        unit_factor_counts[unit_count] = factor_count;
                        unit_count = unit_count + 1u;
                        promoter_count = 0u;
                        factor_count = 0u;
                    }
                    reading_promoters = true;
                }
                
                if promoter_count < 8u {
                    current_promoters[promoter_count] = PromoterData(
                        promoter_type,
                        sign,
                        modifier,
                        embedding,
                    );
                    promoter_count = promoter_count + 1u;
                }
            } else if type_val >= 3u && type_val <= 5u {
                // Factor (internal product, external product, receptor)
                if promoter_count == 0u {
                    break;
                }
                let factor_types = array<u32, 3u>(1u, 0u, 2u); // ExternalMorphogen, InternalMorphogen, Orientant
                let factor_type = factor_types[type_val - 3u];
                
                if factor_count < 8u {
                    current_factors[factor_count] = FactorData(
                        factor_type,
                        sign,
                        modifier,
                        embedding,
                    );
                    factor_count = factor_count + 1u;
                }
                reading_promoters = false;
            }
        }
    }
    
    // Add final regulatory unit if it exists
    if promoter_count > 0u && factor_count > 0u && unit_count < MAX_GRN_REGULATORY_UNITS {
        for (var i: u32 = 0u; i < promoter_count; i = i + 1u) {
            regulatory_units[unit_count][i] = current_promoters[i];
        }
        unit_promoter_counts[unit_count] = promoter_count;
        unit_factor_counts[unit_count] = factor_count;
        unit_count = unit_count + 1u;
    }
    
    // Now compile the GRN into CompiledRegulatoryUnit structures
    let unit_offset = lifeform.grn_unit_offset;
    let descriptor_slot = lifeform.grn_descriptor_slot;
    
    // Update descriptor
    if descriptor_slot < arrayLength(&grn_descriptors) {
        grn_descriptors[descriptor_slot].receptor_count = receptor_count;
        grn_descriptors[descriptor_slot].unit_count = unit_count;
        grn_descriptors[descriptor_slot].state_stride = receptor_count + unit_count;
        grn_descriptors[descriptor_slot].unit_offset = unit_offset;
    }
    
    // Compile each regulatory unit
    for (var unit_idx: u32 = 0u; unit_idx < unit_count; unit_idx = unit_idx + 1u) {
        let compiled_unit_idx = unit_offset + unit_idx;
        if compiled_unit_idx >= arrayLength(&grn_units) {
            break;
        }
        
        // Find top-k inputs (receptors + factors from other units)
        var input_candidates: array<vec3<f32>, 32u>; // index, weight, promoter_type
        var candidate_count: u32 = 0u;
        
        // Check receptors
        for (var rec_idx: u32 = 0u; rec_idx < receptor_count; rec_idx = rec_idx + 1u) {
            let receptor = receptors[rec_idx];
            var max_affinity: f32 = 0.0;
            var best_promoter_type: u32 = 0u;
            
            // Find best matching promoter
            for (var prom_idx: u32 = 0u; prom_idx < unit_promoter_counts[unit_idx]; prom_idx = prom_idx + 1u) {
                let promoter = regulatory_units[unit_idx][prom_idx];
                let affinity = calculate_affinity(
                    receptor.embedding, receptor.sign, receptor.modifier,
                    promoter.embedding, promoter.sign, promoter.modifier
                );
                if abs(affinity) > abs(max_affinity) {
                    max_affinity = affinity;
                    best_promoter_type = promoter.promoter_type;
                }
            }
            
            if abs(max_affinity) > 0.0 && candidate_count < 32u {
                input_candidates[candidate_count] = vec3<f32>(f32(rec_idx), max_affinity, f32(best_promoter_type));
                candidate_count = candidate_count + 1u;
            }
        }
        
        // Check factors from other regulatory units
        var factor_offset: u32 = receptor_count;
        for (var other_unit_idx: u32 = 0u; other_unit_idx < unit_count; other_unit_idx = other_unit_idx + 1u) {
            if other_unit_idx == unit_idx {
                factor_offset = factor_offset + unit_factor_counts[other_unit_idx];
                continue;
            }
            
            for (var fact_idx: u32 = 0u; fact_idx < unit_factor_counts[other_unit_idx]; fact_idx = fact_idx + 1u) {
                let factor = current_factors[fact_idx]; // Simplified - should get from other unit
                var max_affinity: f32 = 0.0;
                var best_promoter_type: u32 = 0u;
                
                for (var prom_idx: u32 = 0u; prom_idx < unit_promoter_counts[unit_idx]; prom_idx = prom_idx + 1u) {
                    let promoter = regulatory_units[unit_idx][prom_idx];
                    let affinity = calculate_affinity(
                        factor.embedding, factor.sign, factor.modifier,
                        promoter.embedding, promoter.sign, promoter.modifier
                    );
                    if abs(affinity) > abs(max_affinity) {
                        max_affinity = affinity;
                        best_promoter_type = promoter.promoter_type;
                    }
                }
                
                if abs(max_affinity) > 0.0 && candidate_count < 32u {
                    input_candidates[candidate_count] = vec3<f32>(f32(factor_offset + fact_idx), max_affinity, f32(best_promoter_type));
                    candidate_count = candidate_count + 1u;
                }
            }
            factor_offset = factor_offset + unit_factor_counts[other_unit_idx];
        }
        
        // Sort candidates by absolute affinity (simple bubble sort for small arrays)
        for (var i: u32 = 0u; i < candidate_count; i = i + 1u) {
            for (var j: u32 = 0u; j < candidate_count - 1u - i; j = j + 1u) {
                if abs(input_candidates[j].y) < abs(input_candidates[j + 1u].y) {
                    let temp = input_candidates[j];
                    input_candidates[j] = input_candidates[j + 1u];
                    input_candidates[j + 1u] = temp;
                }
            }
        }
        
        // Take top MAX_GRN_INPUTS_PER_UNIT inputs
        let num_inputs = min(candidate_count, MAX_GRN_INPUTS_PER_UNIT);
        grn_units[compiled_unit_idx].input_count = num_inputs;
        
        for (var i: u32 = 0u; i < num_inputs; i = i + 1u) {
            grn_units[compiled_unit_idx].inputs[i].index = u32(input_candidates[i].x);
            grn_units[compiled_unit_idx].inputs[i].weight = input_candidates[i].y;
            grn_units[compiled_unit_idx].inputs[i].promoter_type = u32(input_candidates[i].z);
        }
        
        // Find top effectors for this unit
        var effector_candidates: array<vec2<f32>, 16u>; // index, affinity
        var effector_candidate_count: u32 = 0u;
        
        for (var eff_idx: u32 = 0u; eff_idx < effector_count; eff_idx = eff_idx + 1u) {
            let effector = effectors[eff_idx];
            var total_affinity: f32 = 0.0;
            
            for (var fact_idx: u32 = 0u; fact_idx < unit_factor_counts[unit_idx]; fact_idx = fact_idx + 1u) {
                let factor = current_factors[fact_idx];
                let affinity = calculate_affinity(
                    factor.embedding, factor.sign, factor.modifier,
                    effector.embedding, effector.sign, effector.modifier
                );
                total_affinity = total_affinity + affinity;
            }
            
            if abs(total_affinity) > 0.0 && effector_candidate_count < 16u {
                effector_candidates[effector_candidate_count] = vec2<f32>(f32(eff_idx), total_affinity);
                effector_candidate_count = effector_candidate_count + 1u;
            }
        }
        
        // Sort effectors
        for (var i: u32 = 0u; i < effector_candidate_count; i = i + 1u) {
            for (var j: u32 = 0u; j < effector_candidate_count - 1u - i; j = j + 1u) {
                if abs(effector_candidates[j].y) < abs(effector_candidates[j + 1u].y) {
                    let temp = effector_candidates[j];
                    effector_candidates[j] = effector_candidates[j + 1u];
                    effector_candidates[j + 1u] = temp;
                }
            }
        }
        
        // Store top effector (simplified - just store the best one)
        if effector_candidate_count > 0u {
            grn_units[compiled_unit_idx].output_index = u32(effector_candidates[0].x);
        } else {
            grn_units[compiled_unit_idx].output_index = 0u;
        }
    }
}

