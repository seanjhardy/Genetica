pub mod gene_regulatory_network;
pub mod compile_grn;

pub use gene_regulatory_network::{Receptor, Factor, Promoter, Effector, RegulatoryUnit, ReceptorType,FactorType, PromoterType, EffectorType, GeneRegulatoryNetwork, EMBEDDING_DIMENSIONS};
pub use compile_grn::CompiledGRN;