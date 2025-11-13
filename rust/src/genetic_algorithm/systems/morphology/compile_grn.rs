use itertools::Itertools;
use crate::genetic_algorithm::systems::{PromoterType, };
use crate::genetic_algorithm::systems::morphology::gene_regulatory_network::{BINDING_DISTANCE_THRESHOLD, Embedded};
use crate::genetic_algorithm::systems::morphology::gene_regulatory_network::GeneRegulatoryNetwork;
use crate::utils::math::length;
use crate::gpu::structures::{
    CompiledGrn,
    CompiledRegulatoryUnit,
    GrnDescriptor,
    Input,
    MAX_GRN_STATE_SIZE,
    MAX_GRN_INPUTS_PER_UNIT,
    MAX_GRN_REGULATORY_UNITS,
    MAX_GRN_RECEPTOR_INPUTS,
    GRN_EVALUATION_INTERVAL,
};

const MAX_CONNECTIONS: usize = MAX_GRN_INPUTS_PER_UNIT;

#[derive(Clone, Debug, Default)]
struct InputCandidate {
    index: u32,
    weight: f32,
    additive: bool,
}

pub fn compile_grn(_id: u32, grn: &GeneRegulatoryNetwork) -> CompiledGrn {
  let mut regulatory_units: Vec<CompiledRegulatoryUnit> = Vec::new();
  let receptor_count = grn.receptors.len().min(MAX_GRN_RECEPTOR_INPUTS);
  let unit_count = grn.regulatory_units.len().min(MAX_GRN_REGULATORY_UNITS);

  for (unit_idx, unit) in grn.regulatory_units.iter().take(unit_count).enumerate() {
    // Compoute strongest inputs for this regulatory unit
    let mut inputs: Vec<InputCandidate> = Vec::new();

    for promoter in unit.promoters.iter() {
      for (i, receptor) in grn.receptors.iter().take(receptor_count).enumerate() {
        let affinity = calculate_affinity(receptor, promoter);
        inputs.push(InputCandidate {
          index: i as u32,
          weight: affinity,
          additive: promoter.promoter_type == PromoterType::Additive,
        });
      }

      for (i, other_reg_unit) in grn.regulatory_units.iter().take(unit_count).enumerate() {
        if i == unit_idx {
          continue;
        }
        for factor in other_reg_unit.factors.iter() {
          let affinity = calculate_affinity(factor, promoter);
          let source_index = receptor_count + i;
          if source_index < MAX_GRN_STATE_SIZE as usize {
            inputs.push(InputCandidate {
              index: source_index as u32,
              weight: affinity,
              additive: promoter.promoter_type == PromoterType::Additive,
            });
          }
        }
      }
    }

    // Get top-k (MAX_CONNECTIONS) inputs with the highest affinity
    let mut top_inputs: Vec<InputCandidate> = inputs
        .into_iter()
        .filter(|candidate| candidate.weight > 0.0)
        .sorted_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
        .take(MAX_CONNECTIONS)
        .collect();

    // Pad with zeros if needed
    while top_inputs.len() < MAX_CONNECTIONS {
        top_inputs.push(InputCandidate::default());
    }

    let mut unit_entry = CompiledRegulatoryUnit {
        input_count: MAX_CONNECTIONS as u32,
        output_index: (receptor_count + unit_idx) as u32,
        flags: 0,
        _padding: 0,
        inputs: [Input {
            weight: 0.0,
            index: 0,
            promoter_type: 0,
            _pad: 0,
        }; MAX_GRN_INPUTS_PER_UNIT],
    };

    let mut used_inputs = 0u32;
    for (dst, candidate) in unit_entry.inputs.iter_mut().zip(top_inputs.iter()) {
        if candidate.weight == 0.0 && !candidate.additive {
            continue;
        }
        dst.weight = candidate.weight;
        dst.index = candidate.index;
        dst.promoter_type = if candidate.additive { 1 } else { 0 };
        if candidate.weight != 0.0 {
            used_inputs += 1;
        }
    }
    unit_entry.input_count = used_inputs;

    regulatory_units.push(unit_entry);
  }

  CompiledGrn {
      descriptor: GrnDescriptor {
          receptor_count: receptor_count as u32,
          unit_count: unit_count as u32,
          state_stride: (receptor_count + unit_count) as u32,
          unit_offset: 0,
          evaluation_interval: GRN_EVALUATION_INTERVAL,
          _pad0: 0,
          _pad1: 0,
          _pad2: 0,
      },
      units: regulatory_units,
  }
}

fn calculate_affinity(gene1: &impl Embedded, gene2: &impl Embedded) -> f32 {
  let distance = length(&gene1.embedding(), &gene2.embedding());
  if distance > BINDING_DISTANCE_THRESHOLD {
    return 0.0;
  }

  let affinity_sign = if gene1.sign() == gene2.sign() { 1.0 } else { -1.0 };
  let affinity = affinity_sign * 
      (2.0 * (gene1.modifier() * gene2.modifier()).abs() 
      * (BINDING_DISTANCE_THRESHOLD - distance)) /
      (10.0 * distance + (gene1.modifier() * gene2.modifier()).abs());
  affinity
}