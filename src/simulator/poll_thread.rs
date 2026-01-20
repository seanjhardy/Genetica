use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::collections::VecDeque;

use puffin::profile_scope;

use crate::gpu::structures::Event;

/// Commands sent from main thread to poll thread
enum PollCommand {
    /// Request the poll thread to shutdown
    Shutdown,
}

/// Dedicated thread for GPU polling and event processing
pub struct PollThread {
    thread: Option<JoinHandle<()>>,
    command_sender: Sender<PollCommand>,
}

impl PollThread {
    pub fn new(
        device: Arc<wgpu::Device>,
        render_buffers: Arc<parking_lot::Mutex<Arc<crate::gpu::buffers::GpuBuffers>>>,
        paused_state: Arc<parking_lot::Mutex<bool>>,
        genetic_algorithm: Arc<parking_lot::Mutex<crate::genetic_algorithm::GeneticAlgorithm>>,
        step_counter: Arc<std::sync::atomic::AtomicUsize>,
        frame_start_nanos: Arc<AtomicU64>,
        app_start: Arc<std::time::Instant>,
    ) -> Self {
        let (command_sender, command_receiver) = mpsc::channel();

        let thread = thread::Builder::new()
            .name("PollThread".to_string())
            .spawn(move || {
                Self::run_poll_loop(
                    device,
                    render_buffers,
                    paused_state,
                    genetic_algorithm,
                    step_counter,
                    frame_start_nanos,
                    app_start,
                    command_receiver,
                );
            })
            .expect("Failed to spawn poll thread");

        Self {
            thread: Some(thread),
            command_sender,
        }
    }

    pub fn notify_event_scheduled(&self) {
        // No-op for now; the poll thread runs continuously.
    }

    pub fn shutdown(&mut self) {
        let _ = self.command_sender.send(PollCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    fn run_poll_loop(
        device: Arc<wgpu::Device>,
        render_buffers: Arc<parking_lot::Mutex<Arc<crate::gpu::buffers::GpuBuffers>>>,
        paused_state: Arc<parking_lot::Mutex<bool>>,
        genetic_algorithm: Arc<parking_lot::Mutex<crate::genetic_algorithm::GeneticAlgorithm>>,
        step_counter: Arc<std::sync::atomic::AtomicUsize>,
        frame_start_nanos: Arc<AtomicU64>,
        app_start: Arc<std::time::Instant>,
        command_receiver: Receiver<PollCommand>,
    ) {
        puffin::set_scopes_on(true);

        let mut last_frame_start_nanos = 0u64;
        let mut last_read_frame_start_nanos = 0u64;
        let mut frame_deadline_nanos = 0u64;
        let mut pending_events: VecDeque<Event> = VecDeque::new();
        let mut last_debug_log = std::time::Instant::now();
        let mut read_attempts_this_frame = 0u32;
        const MAX_READ_ATTEMPTS_PER_FRAME: u32 = 2;

        loop {
            profile_scope!("Poll Loop");

            while let Ok(cmd) = command_receiver.try_recv() {
                match cmd {
                    PollCommand::Shutdown => return,
                }
            }

            let start_nanos = frame_start_nanos.load(Ordering::Acquire);
            if start_nanos != 0 && start_nanos != last_frame_start_nanos {
                last_frame_start_nanos = start_nanos;
                frame_deadline_nanos = start_nanos.saturating_add(5_000_000);
                read_attempts_this_frame = 0;
            }

            let now_nanos = app_start.elapsed().as_nanos() as u64;
            let should_process = start_nanos == 0 || now_nanos <= frame_deadline_nanos;

            let render_buffers_snapshot = render_buffers.lock().clone();
            let has_pending_readback = render_buffers_snapshot.event_system.has_pending_readback();
            let new_frame = start_nanos != 0 && start_nanos != last_read_frame_start_nanos;
            let should_attempt_read = (new_frame || has_pending_readback)
                && read_attempts_this_frame < MAX_READ_ATTEMPTS_PER_FRAME;

            if should_attempt_read {
                last_read_frame_start_nanos = start_nanos;
                read_attempts_this_frame = read_attempts_this_frame.saturating_add(1);
                profile_scope!("Read Events");
                render_buffers_snapshot.event_system.begin_pending_mappings();
                let _ = device.poll(wgpu::MaintainBase::Poll);
                let events = render_buffers_snapshot.event_system.drain_ready_events();
                if !events.is_empty() {
                    pending_events.extend(events);
                }
                if last_debug_log.elapsed().as_secs_f32() >= 1.0 {
                    let (cpu_idx, gpu_idx, event_count, processed_count) =
                        render_buffers_snapshot.event_system.debug_snapshot();
                    println!(
                        "event readback: cpu_idx={}, gpu_idx={}, event_count={}, processed={}, queued={}",
                        cpu_idx,
                        gpu_idx,
                        event_count,
                        processed_count,
                        pending_events.len()
                    );
                    last_debug_log = std::time::Instant::now();
                }
            } else {
                thread::sleep(Duration::from_micros(200));
            }

            if !pending_events.is_empty() && !*paused_state.lock() {
                let mut ga = genetic_algorithm.lock();
                profile_scope!("Process Events");
                let current_step = step_counter.load(Ordering::Relaxed);
                let batch_size = 500usize;
                for _ in 0..batch_size {
                    if let Some(event) = pending_events.pop_front() {
                        ga.process_event(current_step, event);
                    } else {
                        break;
                    }
                }
            }

            if !should_process {
                profile_scope!("Idle Sleep");
                thread::sleep(Duration::from_micros(200));
                continue;
            }
        }
    }
}
