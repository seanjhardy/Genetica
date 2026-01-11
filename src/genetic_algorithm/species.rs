use std::sync::atomic::{AtomicU32, Ordering};

use crate::genetic_algorithm::Genome;

pub struct Species {
    pub id: usize,
    pub ancestor_species_id: Option<usize>,
    pub descendants: Vec<usize>,
    pub mascot_id: usize,
    pub mascot_genome: Genome,
    pub active_members: AtomicU32,
    pub origin_time: usize,
    pub death_time: Option<usize>,
}

#[allow(dead_code)]
impl Species {
    pub fn new(id: usize, mascot_id: usize, mascot_genome: Genome, origin_time: usize) -> Self {
        Self {
            id,
            ancestor_species_id: None,
            descendants: Vec::new(),
            mascot_id,
            mascot_genome,
            active_members: AtomicU32::new(0),
            origin_time,
            death_time: None,
        }
    }

    /// Mark a lifeform as part of this species. Returns true if the species becomes active.
      pub fn register_member(&mut self) -> bool {
        if self.death_time.is_some() {
            self.death_time = None;
        }
        let previous = self.active_members.fetch_add(1, Ordering::Relaxed);
        previous == 0
    }

    pub fn deregister_member(&mut self, death_time: usize) -> bool {
        let mut current = self.active_members.load(Ordering::Relaxed);
        while current != 0 {
            if self
                .active_members
                .compare_exchange(current, current - 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                if current == 1 {
                    self.death_time = Some(death_time);
                    return true;
                }
                return false;
            }
            current = self.active_members.load(Ordering::Relaxed);
        }
        false
    }
}