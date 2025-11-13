use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::gpu::buffers::EVENT_STAGING_RING_SIZE;

/// Tracks the current population counters for cells and lifeforms.
#[derive(Debug)]
pub struct PopulationState {
    pub last_alive: u32,
    pub predicted_alive: u32,
    pub last_lifeform: u32,
    pub predicted_lifeform: u32,
    pub last_species: u32,
    pub predicted_species: u32,
}

impl PopulationState {
    pub fn new(initial_alive: u32, initial_lifeforms: u32, initial_species: u32) -> Self {
        Self {
            last_alive: initial_alive,
            predicted_alive: initial_alive,
            last_lifeform: initial_lifeforms,
            predicted_lifeform: initial_lifeforms,
            last_species: initial_species,
            predicted_species: initial_species,
        }
    }

    pub fn sync_alive(&mut self, value: u32) {
        self.last_alive = value;
        self.predicted_alive = value;
    }

    pub fn sync_lifeforms(&mut self, value: u32) {
        self.last_lifeform = value;
        self.predicted_lifeform = value;
    }

    pub fn sync_species(&mut self, value: u32) {
        self.last_species = value;
        self.predicted_species = value;
    }
}

/// Tracks which GPU readbacks and copies are still pending.
#[derive(Debug)]
pub struct GpuTransferState {
    pub alive_counter_pending: bool,
    pub lifeform_flags_pending: bool,
    pub cell_events_pending: [bool; EVENT_STAGING_RING_SIZE],
    pub link_events_pending: [bool; EVENT_STAGING_RING_SIZE],
    pub next_cell_event_staging: usize,
    pub next_link_event_staging: usize,
    pub lifeform_events_pending: [bool; EVENT_STAGING_RING_SIZE],
    pub species_events_pending: [bool; EVENT_STAGING_RING_SIZE],
    pub next_lifeform_event_staging: usize,
    pub next_species_event_staging: usize,
}

impl Default for GpuTransferState {
    fn default() -> Self {
        Self {
            alive_counter_pending: false,
            lifeform_flags_pending: false,
            cell_events_pending: [false; EVENT_STAGING_RING_SIZE],
            link_events_pending: [false; EVENT_STAGING_RING_SIZE],
            next_cell_event_staging: 0,
            next_link_event_staging: 0,
            lifeform_events_pending: [false; EVENT_STAGING_RING_SIZE],
            species_events_pending: [false; EVENT_STAGING_RING_SIZE],
            next_lifeform_event_staging: 0,
            next_species_event_staging: 0,
        }
    }
}

/// Batches command buffer submissions to avoid excessive queue submits.
#[derive(Debug)]
pub struct SubmissionState {
    pub pending_command_buffers: Vec<wgpu::CommandBuffer>,
    pub submission_batch_size: usize,
    pub max_submission_delay: Duration,
    pub last_submission: Instant,
}

impl SubmissionState {
    pub fn new(batch_size: usize, max_delay: Duration) -> Self {
        Self {
            pending_command_buffers: Vec::with_capacity(batch_size),
            submission_batch_size: batch_size,
            max_submission_delay: max_delay,
            last_submission: Instant::now(),
        }
    }

    pub fn record_submission_time(&mut self) {
        self.last_submission = Instant::now();
    }
}

/// Handles paused state for the simulation along with an exposed atomic flag.
#[derive(Debug)]
pub struct PauseState {
    paused: bool,
    flag: Arc<AtomicBool>,
}

impl PauseState {
    pub fn new(initial: bool) -> Self {
        Self {
            paused: initial,
            flag: Arc::new(AtomicBool::new(initial)),
        }
    }

    pub fn set(&mut self, paused: bool) {
        self.paused = paused;
        self.flag.store(paused, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn handle(&self) -> Arc<AtomicBool> {
        self.flag.clone()
    }
}

