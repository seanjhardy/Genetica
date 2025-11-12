use std::collections::{HashMap, HashSet};

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
    id_to_slots: HashMap<usize, HashSet<u32>>,
    alive_slots: Vec<bool>,
    metadata: HashMap<usize, LifeformMetadata>,
}

pub struct RegistryUpdate {
    pub active_cells: u32,
    pub active_lifeforms: u32,
    pub extinct_ids: Vec<usize>,
}

impl LifeformRegistry {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free_slots: (0..capacity as u32).rev().collect(),
            free_slot_listed: vec![true; capacity],
            slot_to_id: vec![None; capacity],
            id_to_slots: HashMap::new(),
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
        self.id_to_slots
            .entry(lifeform_id)
            .or_insert_with(HashSet::new)
            .insert(slot);
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
        self.id_to_slots
            .entry(lifeform_id)
            .or_insert_with(HashSet::new)
            .insert(slot);
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
        let mut extinct_id = None;
        if let Some(lifeform_id) = self.slot_to_id[slot as usize].take() {
            if let Some(slots) = self.id_to_slots.get_mut(&lifeform_id) {
                slots.remove(&slot);
                if slots.is_empty() {
                    self.id_to_slots.remove(&lifeform_id);
                    self.metadata.remove(&lifeform_id);
                    extinct_id = Some(lifeform_id);
                }
            }
        } else {
            return None;
        }
        if let Some(entry) = self.free_slot_listed.get_mut(slot as usize) {
            if !*entry {
                *entry = true;
                self.free_slots.push(slot);
            }
        } else {
            self.free_slots.push(slot);
        }
        extinct_id
    }

    pub fn apply_gpu_flags(&mut self, flags: &[u32]) -> RegistryUpdate {
        let limit = self.alive_slots.len().min(flags.len());
        let mut extinct_ids = Vec::new();
        let mut active_cells = 0u32;
        for slot in 0..limit {
            if flags[slot] != 0 {
                self.alive_slots[slot] = true;
                active_cells += 1;
                if let Some(entry) = self.free_slot_listed.get_mut(slot) {
                    *entry = false;
                }
            } else if self.alive_slots[slot] {
                self.alive_slots[slot] = false;
                if let Some(id) = self.slot_to_id[slot].take() {
                    if let Some(slots) = self.id_to_slots.get_mut(&id) {
                        slots.remove(&(slot as u32));
                        if slots.is_empty() {
                            self.id_to_slots.remove(&id);
                            self.metadata.remove(&id);
                            extinct_ids.push(id);
                        }
                    }
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
            active_cells,
            active_lifeforms: self.id_to_slots.len() as u32,
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

    pub fn active_lifeform_count(&self) -> u32 {
        self.id_to_slots.len() as u32
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

