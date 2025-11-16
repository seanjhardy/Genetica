use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Tracks the current population counters for cells and lifeforms.
#[derive(Debug)]
pub struct PopulationState {
    pub lifeforms: u32,
    pub species: u32,
    pub cells: u32,
}

impl PopulationState {
    pub fn new() -> Self {
        Self {
            lifeforms: 0,
            species: 0,
            cells: 0,
        }
    }
}

/// Tracks which GPU readbacks and copies are still pending.
#[derive(Debug)]
pub struct GpuTransferState {
    pub cell_counter_pending: bool,
    pub lifeform_counter_pending: bool,
    pub species_counter_pending: bool,
}

impl Default for GpuTransferState {
    fn default() -> Self {
        Self {
            cell_counter_pending: false,
            lifeform_counter_pending: false,
            species_counter_pending: false,
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

