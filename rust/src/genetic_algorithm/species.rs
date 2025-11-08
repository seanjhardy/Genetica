pub struct Species {
  mascot_id: u32,
  members: Vec<u32>,
  origin_time: u32,
  death_time: u32,
}

impl Species {
  pub fn new(mascot_id: u32, origin_time: u32) -> Self {
    Self {
      mascot_id,
      members: Vec::new(),
      origin_time,
      death_time: 0,
    }
  }
}