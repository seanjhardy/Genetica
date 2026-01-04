use itertools::Itertools;
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

pub fn compile_grn(id: u32, grn: GeneRegulatoryNetwork) -> Vec<CompiledRegulatoryUnit> {
  puffin::profile_scope!("Compile GRN");
  let mut regulatory_units = Vec::new();

  for unit in grn.regulatory_units.iter() {
    profile_scope!("Compile GRN Unit");
    // Compoute strongest inputs for this regulatory unit
    let mut inputs: Vec<(u16, f32, bool)> = Vec::new();

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

    let mut outputs: Vec<(u16, f32)> = Vec::new();
    for (i, effector) in grn.effectors.iter().enumerate() {
      profile_scope!("Calculate Effector Affinity");
      let mut affinity = 0.0;
      for factor in unit.factors.iter() {
        affinity += calculate_affinity(factor, effector);
      }
      outputs.push((i as u16, affinity));
    }
  
    // Get top-k (MAX_CONNECTIONS) inputs with the highest affinity
    let top_inputs: Vec<(u16, f32, bool)> = inputs
                    .iter()
                    .cloned()
                    .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
                    .take(MAX_CONNECTIONS)
                    .collect();
    let top_inputs: Vec<Input> = top_inputs.iter().map(|(index, weight, promoter_type)| Input {
      weight: *weight,
      index: *index,
      promoter_type: if *promoter_type { 1 } else { 0 },
    }).collect();
    let num_inputs = top_inputs.len() as u32;

    let top_outputs: Vec<(u16, f32)> = outputs
                    .iter()
                    .cloned()
                    .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
                    .take(MAX_CONNECTIONS)
                    .collect();
    let effector_indices: [u16; MAX_CONNECTIONS] = top_outputs.iter().map(|(index, _)| *index).collect::<Vec<u16>>().try_into().unwrap();
    let effector_weights: [f32; MAX_CONNECTIONS] = top_outputs.iter().map(|(_, weight)| *weight).collect::<Vec<f32>>().try_into().unwrap();
    let num_outputs = top_outputs.len() as u32;

    let compiled_regulatory_unit = CompiledRegulatoryUnit {
      grn_id: id,
      inputs: top_inputs.try_into().unwrap(),
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