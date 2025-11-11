use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::gpu::structures::Cell;

/// Tracks the current population counters for cells and lifeforms.
#[derive(Debug)]
pub struct PopulationState {
    pub last_alive: u32,
    pub predicted_alive: u32,
    pub last_lifeform: u32,
    pub predicted_lifeform: u32,
}

impl PopulationState {
    pub fn new(initial_alive: u32, initial_lifeforms: u32) -> Self {
        Self {
            last_alive: initial_alive,
            predicted_alive: initial_alive,
            last_lifeform: initial_lifeforms,
            predicted_lifeform: initial_lifeforms,
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
}

/// Tracks which GPU readbacks and copies are still pending.
#[derive(Debug, Default)]
pub struct GpuTransferState {
    pub alive_counter_pending: bool,
    pub lifeform_counter_pending: bool,
    pub lifeform_flags_pending: bool,
    pub division_requests_pending: bool,
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

/// Maintains spawn-related scratch buffers and limits.
#[derive(Debug)]
pub struct SpawnState {
    pub capacity: usize,
    pub scratch: Vec<Cell>,
}

impl SpawnState {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            scratch: Vec::with_capacity(capacity),
        }
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

