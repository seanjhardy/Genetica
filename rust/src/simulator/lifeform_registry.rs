use std::collections::HashMap;

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct LifeformMetadata {
    pub species_id: Option<usize>,
    pub genome_id: Option<usize>,
}

pub struct LifeformRegistry {
    capacity: usize,
    free_slots: Vec<u32>,
    free_slot_listed: Vec<bool>,
    slot_to_id: Vec<Option<usize>>,
    id_to_slot: HashMap<usize, u32>,
    alive_slots: Vec<bool>,
    metadata: HashMap<usize, LifeformMetadata>,
}

pub struct RegistryUpdate {
    pub active_total: u32,
    pub extinct_ids: Vec<usize>,
}

impl LifeformRegistry {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free_slots: (0..capacity as u32).rev().collect(),
            free_slot_listed: vec![true; capacity],
            slot_to_id: vec![None; capacity],
            id_to_slot: HashMap::new(),
            alive_slots: vec![false; capacity],
            metadata: HashMap::new(),
        }
    }

    pub fn bootstrap_slot(
        &mut self,
        slot: u32,
        lifeform_id: usize,
        metadata: LifeformMetadata,
    ) {
        if (slot as usize) >= self.capacity {
            return;
        }
        self.alive_slots[slot as usize] = true;
        self.slot_to_id[slot as usize] = Some(lifeform_id);
        self.id_to_slot.insert(lifeform_id, slot);
        self.metadata.insert(lifeform_id, metadata);
        if self.free_slot_listed.get(slot as usize).copied().unwrap_or(false) {
            if let Some(pos) = self.free_slots.iter().position(|&s| s == slot) {
                self.free_slots.swap_remove(pos);
            }
            if let Some(entry) = self.free_slot_listed.get_mut(slot as usize) {
                *entry = false;
            }
        }
    }

    pub fn reserve_slot(&mut self) -> Option<u32> {
        while let Some(slot) = self.free_slots.pop() {
            if (slot as usize) >= self.capacity {
                continue;
            }
            let listed = self
                .free_slot_listed
                .get_mut(slot as usize)
                .map(|entry| {
                    let was_listed = *entry;
                    *entry = false;
                    was_listed
                })
                .unwrap_or(false);
            if listed {
                self.alive_slots[slot as usize] = true;
                return Some(slot);
            }
        }
        None
    }

    pub fn assign_id_to_slot(
        &mut self,
        slot: u32,
        lifeform_id: usize,
        metadata: LifeformMetadata,
    ) {
        if (slot as usize) >= self.capacity {
            return;
        }
        self.slot_to_id[slot as usize] = Some(lifeform_id);
        self.id_to_slot.insert(lifeform_id, slot);
        self.metadata.insert(lifeform_id, metadata);
        if let Some(entry) = self.free_slot_listed.get_mut(slot as usize) {
            *entry = false;
        }
    }

    pub fn release_slot(&mut self, slot: u32) -> Option<usize> {
        if (slot as usize) >= self.capacity {
            return None;
        }
        self.alive_slots[slot as usize] = false;
        let id = self.slot_to_id[slot as usize].take();
        if let Some(lifeform_id) = id {
            self.id_to_slot.remove(&lifeform_id);
            self.metadata.remove(&lifeform_id);
        }
        if let Some(entry) = self.free_slot_listed.get_mut(slot as usize) {
            if !*entry {
                *entry = true;
                self.free_slots.push(slot);
            }
        } else {
            self.free_slots.push(slot);
        }
        id
    }

    pub fn apply_gpu_flags(&mut self, flags: &[u32]) -> RegistryUpdate {
        let limit = self.alive_slots.len().min(flags.len());
        let mut active_total = 0u32;
        let mut extinct_ids = Vec::new();
        for slot in 0..limit {
            if flags[slot] != 0 {
                self.alive_slots[slot] = true;
                active_total += 1;
                if let Some(entry) = self.free_slot_listed.get_mut(slot) {
                    *entry = false;
                }
            } else if self.alive_slots[slot] {
                self.alive_slots[slot] = false;
                if let Some(id) = self.slot_to_id[slot].take() {
                    self.id_to_slot.remove(&id);
                    self.metadata.remove(&id);
                    extinct_ids.push(id);
                }
                let slot_u32 = slot as u32;
                if let Some(entry) = self.free_slot_listed.get_mut(slot) {
                    if !*entry {
                        *entry = true;
                        self.free_slots.push(slot_u32);
                    }
                } else {
                    self.free_slots.push(slot_u32);
                }
            }
        }
        RegistryUpdate {
            active_total,
            extinct_ids,
        }
    }

    pub fn id_for_slot(&self, slot: u32) -> Option<usize> {
        self.slot_to_id
            .get(slot as usize)
            .and_then(|entry| *entry)
    }

    pub fn is_slot_active(&self, slot: u32) -> bool {
        self.alive_slots
            .get(slot as usize)
            .copied()
            .unwrap_or(false)
    }

    pub fn active_count(&self) -> u32 {
        self.alive_slots.iter().filter(|alive| **alive).count() as u32
    }

    #[allow(dead_code)]
    pub fn metadata(&self, id: usize) -> Option<&LifeformMetadata> {
        self.metadata.get(&id)
    }

    #[allow(dead_code)]
    pub fn metadata_mut(&mut self, id: usize) -> Option<&mut LifeformMetadata> {
        self.metadata.get_mut(&id)
    }

    #[allow(dead_code)]
    pub fn insert_metadata(&mut self, id: usize, metadata: LifeformMetadata) {
        self.metadata.insert(id, metadata);
    }

}

