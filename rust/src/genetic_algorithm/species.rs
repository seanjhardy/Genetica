pub struct Species {
  mascotId: usize,
  members: Vec<usize>,
  originTime: usize,
  deathTime: usize,
}

impl Species {
  pub fn new(mascotId: usize, originTime: usize) -> Self {
    Self {
      mascotId,
      members: Vec::new(),
      originTime,
      deathTime: 0,
    }
  }
}