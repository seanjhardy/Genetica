use wgpu::util::DeviceExt;

use crate::genetic_algorithm::systems::{ReceptorType, morphology::{GeneRegulatoryNetwork, gene_regulatory_network::{Embedded, BINDING_DISTANCE_THRESHOLD}}};
use crate::utils::math::length;

pub struct GRNInput {
    pub input_type: ReceptorType,
    pub extra: [f32; 2],
}


pub struct CompiledGRN {
  // Network architecture paramers
  pub inputs: Vec<GRNInput>,

  // Dimensions
  pub input_size: usize,
  pub hidden_size: usize,
  pub output_size: usize,

  // GPU Buffers for inputs and outputs
  // Previous states hidden nodes are included in input buffer
  pub in_state: wgpu::Buffer,
  pub out_state: wgpu::Buffer,
  // GPU Buffers for weight matrices
  pub w_inh_h: wgpu::Buffer,
  pub w_h_out: wgpu::Buffer,
}

impl CompiledGRN {
  pub fn new(grn: GeneRegulatoryNetwork, device: &wgpu::Device) -> Self {

    // Set up cpu-side input descriptor
    let mut inputs: Vec<GRNInput> = grn.receptors.iter().map(|receptor| GRNInput {
      input_type: receptor.receptor_type,
      extra: receptor.extra,
    }).collect();

    // Create empty states
    let in_state = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("In State Buffer"),
        size: ((inputs.len() + grn.regulatory_units.len()) * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_state = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Out State Buffer"),
        size: (grn.effectors.len() * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Set up gpu-side weight matrices
    let w_inh_h_dim = (inputs.len() + grn.regulatory_units.len(), grn.regulatory_units.len());
    let w_h_out_dim = (grn.regulatory_units.len(), grn.effectors.len());

    // Set up matrices using values from affinities of genes
    // i -> h weight matrix
    let mut w_inh_h_cpu = vec![0.0f32; w_inh_h_dim.0 * w_inh_h_dim.1];
    for (i, receptor) in grn.receptors.iter().enumerate() {
      for (j, regulatory_unit) in grn.regulatory_units.iter().enumerate() {
        for promoter in regulatory_unit.promoters.iter() {
          // affinity between receptor and this promoter in this regulatory unit
          w_inh_h_cpu[i * w_inh_h_dim.1 + j] += calculate_affinity(receptor, promoter);
        }
      }
    }
    // h -> h weight matrix
    for (i, regulatory_unit_1) in grn.regulatory_units.iter().enumerate() {
      for (j, regulatory_unit_2) in grn.regulatory_units.iter().enumerate() {
        for promoter in regulatory_unit_1.promoters.iter() {
          for factor in regulatory_unit_2.factors.iter() {
            w_inh_h_cpu[(inputs.len() + i) * w_inh_h_dim.1 + j] += calculate_affinity(factor, promoter);
          }
        }
      }
    }
    let w_inh_h = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("W In H Buffer"),
      contents: bytemuck::cast_slice(&w_inh_h_cpu),
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });


    // h -> out weight matrix
    let mut w_h_out_cpu = vec![0.0f32; w_h_out_dim.0 * w_h_out_dim.1];
    for (i, regulatory_unit) in grn.regulatory_units.iter().enumerate() {
      for (j, effector) in grn.effectors.iter().enumerate() {
        for factor in regulatory_unit.factors.iter() {
          w_h_out_cpu[i * w_h_out_dim.1 + j] += calculate_affinity(factor, effector);
        }
      }
    }
    let w_h_out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("W H Out Buffer"),
      contents: bytemuck::cast_slice(&w_h_out_cpu),
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    Self {
      inputs,
      in_state,
      out_state,
      input_size: grn.receptors.len(),
      hidden_size: grn.regulatory_units.len(),
      output_size: grn.effectors.len(),
      w_inh_h,
      w_h_out,
    }
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