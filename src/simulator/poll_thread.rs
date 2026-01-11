use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::collections::VecDeque;

use puffin::profile_scope;

use crate::gpu::structures::Event;
use crate::simulator::simulator::Simulation;

/// Commands sent from main thread to poll thread
enum PollCommand {
    /// Notify that event reading was scheduled for the current frame
    EventReadScheduled,
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
        render_buffers: Arc<crate::gpu::buffers::GpuBuffers>,
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
        let _ = self.command_sender.send(PollCommand::EventReadScheduled);
    }

    pub fn shutdown(&mut self) {
        let _ = self.command_sender.send(PollCommand::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }

    fn run_poll_loop(
        device: Arc<wgpu::Device>,
        render_buffers: Arc<crate::gpu::buffers::GpuBuffers>,
        paused_state: Arc<parking_lot::Mutex<bool>>,
        genetic_algorithm: Arc<parking_lot::Mutex<crate::genetic_algorithm::GeneticAlgorithm>>,
        step_counter: Arc<std::sync::atomic::AtomicUsize>,
        frame_start_nanos: Arc<AtomicU64>,
        app_start: Arc<std::time::Instant>,
        command_receiver: Receiver<PollCommand>,
    ) {
        puffin::set_scopes_on(true);

        let mut pending_readback = false;
        let mut last_frame_start_nanos = 0u64;
        let mut frame_deadline_nanos = 0u64;
        let mut pending_events: VecDeque<Event> = VecDeque::new();

        loop {
            profile_scope!("Poll Loop");

            while let Ok(cmd) = command_receiver.try_recv() {
                match cmd {
                    PollCommand::EventReadScheduled => {
                        pending_readback = true;
                    }
                    PollCommand::Shutdown => {
                        return;
                    }
                }
            }

            let start_nanos = frame_start_nanos.load(Ordering::Acquire);
            if start_nanos != 0 && start_nanos != last_frame_start_nanos {
                last_frame_start_nanos = start_nanos;
                frame_deadline_nanos = start_nanos.saturating_add(5_000_000);
            }

            let now_nanos = app_start.elapsed().as_nanos() as u64;
            let should_process = start_nanos == 0 || now_nanos <= frame_deadline_nanos;

            if pending_readback && should_process {
                profile_scope!("Process Pending Readback");
                profile_scope!("Read Events");
                let events = render_buffers.event_system.read_events_blocking(&device);
                if !events.is_empty() {
                    pending_events.extend(events);
                }
                render_buffers.event_system.finish_readback();
                pending_readback = false;
            }

            if should_process && !pending_events.is_empty() && !*paused_state.lock() {
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
                thread::sleep(Duration::from_millis(1));
                continue;
            }

            if !pending_readback {
                profile_scope!("Idle Sleep");
                thread::sleep(Duration::from_millis(1));
            }
        }
    }
}
