use std::cmp::Ordering;
use puffin::profile_scope;
use crate::genetic_algorithm::systems::{PromoterType, };
use crate::genetic_algorithm::systems::morphology::gene_regulatory_network::{BINDING_DISTANCE_THRESHOLD, Embedded};
use crate::genetic_algorithm::systems::morphology::gene_regulatory_network::GeneRegulatoryNetwork;
use crate::utils::math::length;


const MAX_CONNECTIONS: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Input {
  weight: f32,
  index: u16,
  promoter_type: u16,
}

impl Default for Input {
  fn default() -> Self {
    Self {
      weight: 0.0,
      index: 0,
      promoter_type: 0,
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompiledRegulatoryUnit {
  grn_id: u32,
  inputs: [Input; MAX_CONNECTIONS],
  effector_indices: [u16; MAX_CONNECTIONS],
  effector_weights: [f32; MAX_CONNECTIONS],
  num_inputs: u32,
  num_outputs: u32,
}

impl Default for CompiledRegulatoryUnit {
  fn default() -> Self {
    Self {
      grn_id: 0,
      num_inputs: 0,
      num_outputs: 0,
      inputs: [Input::default(); MAX_CONNECTIONS],
      effector_indices: [0; MAX_CONNECTIONS],
      effector_weights: [0.0; MAX_CONNECTIONS],
    }
  }
}

pub fn compile_grn(id: u32, grn: &GeneRegulatoryNetwork) -> Vec<CompiledRegulatoryUnit> {
  puffin::profile_scope!("Compile GRN");
  let mut regulatory_units = Vec::with_capacity(grn.regulatory_units.len());
  let receptors_len = grn.receptors.len();
  let total_factors: usize = grn
    .regulatory_units
    .iter()
    .map(|unit| unit.factors.len())
    .sum();
  let effector_count = grn.effectors.len();

  for unit in grn.regulatory_units.iter() {
    profile_scope!("Compile GRN Unit");
    // Compoute strongest inputs for this regulatory unit
    let mut inputs: Vec<(u16, f32, bool)> = Vec::with_capacity(
      unit.promoters.len().saturating_mul(receptors_len + total_factors),
    );

    for promoter in unit.promoters.iter() {
      profile_scope!("Compile GRN Promoter");
      {
        profile_scope!("Calculate Receptor Affinities");
        for (i, receptor) in grn.receptors.iter().enumerate() {
          let affinity = calculate_affinity(receptor, promoter);
          inputs.push((i as u16, affinity, promoter.promoter_type == PromoterType::Additive));
        }
      }

      {
        profile_scope!("Calculate Factor Affinities");
        for (i, other_reg_unit) in grn.regulatory_units.iter().enumerate() {
          for factor in other_reg_unit.factors.iter() {
            let affinity = calculate_affinity(factor, promoter);
            inputs.push(((grn.receptors.len() + i) as u16, affinity, promoter.promoter_type == PromoterType::Additive));
          }
        }
      }
    }

    let mut outputs: Vec<(u16, f32)> = Vec::with_capacity(effector_count);
    for (i, effector) in grn.effectors.iter().enumerate() {
      profile_scope!("Calculate Effector Affinity");
      let mut affinity = 0.0;
      for factor in unit.factors.iter() {
        affinity += calculate_affinity(factor, effector);
      }
      outputs.push((i as u16, affinity));
    }
  
    let by_affinity = |a: &(u16, f32, bool), b: &(u16, f32, bool)| {
      b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    };
    if inputs.len() > MAX_CONNECTIONS {
      inputs.select_nth_unstable_by(MAX_CONNECTIONS - 1, by_affinity);
      inputs.truncate(MAX_CONNECTIONS);
    }
    inputs.sort_unstable_by(by_affinity);
    let num_inputs = inputs.len().min(MAX_CONNECTIONS) as u32;
    let mut input_array = [Input::default(); MAX_CONNECTIONS];
    for (slot, (index, weight, promoter_type)) in inputs.iter().take(MAX_CONNECTIONS).enumerate() {
      input_array[slot] = Input {
        weight: *weight,
        index: *index,
        promoter_type: if *promoter_type { 1 } else { 0 },
      };
    }

    let by_weight = |a: &(u16, f32), b: &(u16, f32)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    if outputs.len() > MAX_CONNECTIONS {
      outputs.select_nth_unstable_by(MAX_CONNECTIONS - 1, by_weight);
      outputs.truncate(MAX_CONNECTIONS);
    }
    outputs.sort_unstable_by(by_weight);
    let num_outputs = outputs.len().min(MAX_CONNECTIONS) as u32;
    let mut effector_indices = [0u16; MAX_CONNECTIONS];
    let mut effector_weights = [0.0f32; MAX_CONNECTIONS];
    for (slot, (index, weight)) in outputs.iter().take(MAX_CONNECTIONS).enumerate() {
      effector_indices[slot] = *index;
      effector_weights[slot] = *weight;
    }

    let compiled_regulatory_unit = CompiledRegulatoryUnit {
      grn_id: id,
      inputs: input_array,
      effector_indices,
      effector_weights,
      num_inputs,
      num_outputs,
    };
    regulatory_units.push(compiled_regulatory_unit);
  }
  regulatory_units
}

fn calculate_affinity(gene1: &impl Embedded, gene2: &impl Embedded) -> f32 {
  puffin::profile_scope!("Calculate Affinity");
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
