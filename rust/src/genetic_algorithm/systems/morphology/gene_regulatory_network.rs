pub const EMBEDDING_DIMENSIONS: usize = 3;
pub const BINDING_DISTANCE_THRESHOLD: f32 = 0.2;

pub struct GeneRegulatoryNetwork {
  // Inputs
  pub receptors: Vec<Receptor>,
  // hidden units
  pub regulatory_units: Vec<RegulatoryUnit>,
  // Outputs
  pub effectors: Vec<Effector>,
}

impl GeneRegulatoryNetwork {
    pub fn new() -> Self {
        Self {
          receptors: Vec::new(),
          regulatory_units: Vec::new(),
          effectors: Vec::new(),
        }
    }
}


pub trait Embedded {
  fn embedding(&self) -> [f32; EMBEDDING_DIMENSIONS];
  fn sign(&self) -> bool;
  fn modifier(&self) -> f32;
}

#[derive(Clone, Copy)]
pub enum ReceptorType {
  MaternalFactor,
  Crowding,
  Constant,
  Generation,
  Energy,
  Time,
}

#[derive(Clone, Copy)]
pub enum FactorType {
  ExternalMorphogen,
  InternalMorphogen,
  Orientant,
}

#[derive(Clone, Copy)]
pub enum PromoterType {
  Additive,
  Multiplicative,
}

#[derive(Clone, Copy)]
pub enum EffectorType {
  Die,
  Divide,
  Freeze,
  Distance,
  Radius,
  Red,
  Green,
  Blue,
}


pub struct Receptor {
    pub receptor_type: ReceptorType,
    pub sign: bool,
    pub modifier: f32,
    pub embedding: [f32; EMBEDDING_DIMENSIONS],
    pub extra: [f32; 2],
}

impl Receptor {
  pub fn new(receptor_type: ReceptorType, sign: bool, modifier: f32, embedding: [f32; EMBEDDING_DIMENSIONS], extra: [f32; 2]) -> Self {
    Self {
      receptor_type,
      sign,
      modifier,
      embedding,
      extra,
    }
  }
}

impl Embedded for Receptor {
  fn embedding(&self) -> [f32; EMBEDDING_DIMENSIONS] {
    self.embedding
  }
  fn sign(&self) -> bool {
    self.sign
  }
  fn modifier(&self) -> f32 {
    self.modifier
  }
}

#[derive(Clone)]
pub struct Factor {
  pub factor_type: FactorType,
  pub sign: bool,
  pub modifier: f32,
  pub embedding: [f32; EMBEDDING_DIMENSIONS],
}

impl Factor {
  pub fn new(factor_type: FactorType, sign: bool, modifier: f32, embedding: [f32; EMBEDDING_DIMENSIONS]) -> Self {
    Self {
      factor_type,
      sign,
      modifier,
      embedding,
    }
  }
}


impl Embedded for Factor {
  fn embedding(&self) -> [f32; EMBEDDING_DIMENSIONS] {
    self.embedding
  }
  fn sign(&self) -> bool {
    self.sign
  }
  fn modifier(&self) -> f32 {
    self.modifier
  }
}

#[derive(Clone)]
pub struct Promoter {
  pub promoter_type: PromoterType,
  pub sign: bool,
  pub modifier: f32,
  pub embedding: [f32; EMBEDDING_DIMENSIONS],
}


impl Promoter {
  pub fn new(promoter_type: PromoterType, sign: bool, modifier: f32, embedding: [f32; EMBEDDING_DIMENSIONS]) -> Self {
    Self {
      promoter_type,
      sign,
      modifier,
      embedding,
    }
  }
}


impl Embedded for Promoter {
  fn embedding(&self) -> [f32; EMBEDDING_DIMENSIONS] {
    self.embedding
  }
  fn sign(&self) -> bool {
    self.sign
  }
  fn modifier(&self) -> f32 {
    self.modifier
  }
}

#[derive(Clone)]
pub struct Effector {
  pub effector_type: EffectorType,
  pub sign: bool,
  pub modifier: f32,
  pub embedding: [f32; EMBEDDING_DIMENSIONS],
}

impl Effector {
  pub fn new(effector_type: EffectorType, sign: bool, modifier: f32, embedding: [f32; EMBEDDING_DIMENSIONS]) -> Self {
    Self {
      effector_type,
      sign,
      modifier,
      embedding,
    }
  }
}

impl Embedded for Effector {
  fn embedding(&self) -> [f32; EMBEDDING_DIMENSIONS] {
    self.embedding
  }
  fn sign(&self) -> bool {
    self.sign
  }
  fn modifier(&self) -> f32 {
    self.modifier
  }
}

pub struct RegulatoryUnit {
  pub promoters: Vec<Promoter>,
  pub factors: Vec<Factor>,
}

impl RegulatoryUnit {
  pub fn new() -> Self {
    Self {
      promoters: Vec::new(),
      factors: Vec::new(),
    }
  }
}