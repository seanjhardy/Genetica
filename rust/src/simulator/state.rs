use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use futures::channel::oneshot;
use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::{Mutex};
use std::sync::mpsc;
use bytemuck::Zeroable;

use crate::gpu::structures::Event;


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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Free,
    Completed,
}

pub struct SimSlot {
    pub buffers: Arc<crate::gpu::buffers::GpuBuffers>,
    pub compute_pipelines: crate::gpu::pipelines::ComputePipelines,
    pub completion: Option<oneshot::Receiver<()>>,
    pub step_id: u64,
    pub state: SlotState,
}


enum ReadState {
    Idle,
    CopyQueued,                    // copy submitted last frame; next step is map
    Mapping(mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>),
}

pub struct Counter {
    pub buffer: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    state: Mutex<ReadState>,
    last_value: Mutex<u32>,
}

impl Counter {
    pub fn new(device: &wgpu::Device, label: &str, initial_value: u32) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} Counter")),
            contents: bytemuck::cast_slice(&[initial_value]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Counter Staging")),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            staging,
            state: Mutex::new(ReadState::Idle),
            last_value: Mutex::new(initial_value),
        }
    }

    pub fn get_last(&self) -> u32 {
        let last = *self.last_value.lock();
        last
    }

    /// Call while encoding commands for frame N
    pub fn schedule_copy_if_idle(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut st = self.state.lock();
        match &*st {
            ReadState::Idle => {
                encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.staging, 0, 4);
                *st = ReadState::CopyQueued;
            }
            _ => {}
        }
    }

    /// Call AFTER queue.submit (frame N or N+1), and while you are polling every frame.
    pub fn begin_map_if_ready(&self) {
        let mut st = self.state.lock();
        if !matches!(*st, ReadState::CopyQueued) {
            return;
        }

        let slice = self.staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        *st = ReadState::Mapping(receiver);
    }

    /// Call after device.poll(Maintain::Poll). Returns newest value if ready; otherwise last known.
    pub fn try_read(&self) -> u32 {
        // fast path: if not mapping, just return last
        let mut st = self.state.lock();
        let last = *self.last_value.lock();

        let rx = match &*st {
            ReadState::Mapping(rx) => rx,
            _ => { return last;}
        };

        match rx.try_recv() {
            Ok(Ok(())) => {
                let mapped = self.staging.slice(..).get_mapped_range();
                let val = bytemuck::from_bytes::<u32>(&mapped[..4]);
                let val = *val;
                drop(mapped);
                self.staging.unmap();

                *self.last_value.lock() = val;
                *st = ReadState::Idle;
                val
            }
            Ok(Err(e)) => {
                println!("Counter: try_read - mapping failed: {:?}", e);
                self.staging.unmap();
                *st = ReadState::Idle;
                last
            }
            Err(mpsc::TryRecvError::Empty) => {
                last
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                println!("Counter: try_read - channel disconnected");
                *st = ReadState::Idle;
                last
            }
        }
    }
}

/// Double-buffered event system for reliable GPU->CPU communication
pub struct EventSystem {
    /// GPU event buffers (double buffered)
    event_buffers: [wgpu::Buffer; 2],
    /// GPU atomic counters for events
    counter_buffers: [wgpu::Buffer; 2],
    /// CPU staging buffer for reading events
    staging_events: wgpu::Buffer,
    /// CPU staging buffer for reading counter
    staging_counter: wgpu::Buffer,
    /// Mutable state protected by mutex
    state: Mutex<EventSystemState>,
    /// Prevent overlapping readbacks into shared staging buffers
    readback_in_flight: AtomicBool,
}

/// Mutable state of the EventSystem
struct EventSystemState {
    /// Current buffer index for GPU writing
    gpu_write_index: usize,
    /// Current buffer index for CPU reading
    cpu_read_index: usize,
    /// Events read so far in current buffer
    events_read: usize,
    /// Maximum events per buffer
    max_events: usize,
}

/// Tracks async buffer mapping
pub struct BufferMapping {
    /// Channel receiver for mapping completion
    receiver: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// Number of mapping completions received
    completed: AtomicUsize,
    /// Whether any mapping failed
    failed: AtomicBool,
}

impl EventSystem {
    /// Create a new double-buffered event system
    pub fn new(device: &wgpu::Device) -> Self {
        let max_events = 2000; // Much smaller buffer - most frames have few or no events

        // Create double-buffered event buffers (initialized with zeros)
        let event_buffer_size = (max_events * std::mem::size_of::<Event>()) as u64;
        let zeros = vec![0u8; event_buffer_size as usize];
        let event_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Buffer 0"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Buffer 1"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        // Create double-buffered counter buffers
        let counter_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Counter 0"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Counter 1"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            }),
        ];

        // CPU staging buffers
        let staging_events = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Event Staging Buffer"),
            size: event_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Event Counter Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let state = EventSystemState {
            gpu_write_index: 0,
            cpu_read_index: 1, // Start reading from buffer 1
            events_read: 0,
            max_events,
        };

        Self {
            event_buffers,
            counter_buffers,
            staging_events,
            staging_counter,
            state: Mutex::new(state),
            readback_in_flight: AtomicBool::new(false),
        }
    }

    /// Schedule copying the current read buffer to staging for CPU access.
    /// Returns false if a readback is already in flight.
    pub fn try_schedule_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        if self.readback_in_flight.load(Ordering::Acquire) {
            return false;
        }

        let (cpu_read_index, gpu_write_index, max_events) = {
            let mut state = self.state.lock();
            let temp = state.gpu_write_index;
            state.gpu_write_index = state.cpu_read_index;
            state.cpu_read_index = temp;
            state.events_read = 0;
            (state.cpu_read_index, state.gpu_write_index, state.max_events)
        };

        queue.write_buffer(&self.counter_buffers[gpu_write_index], 0, bytemuck::cast_slice(&[0u32]));
        let zeros = vec![Event::zeroed(); max_events];
        queue.write_buffer(&self.event_buffers[gpu_write_index], 0, bytemuck::cast_slice(&zeros));

        encoder.copy_buffer_to_buffer(
            &self.counter_buffers[cpu_read_index], 0,
            &self.staging_counter, 0,
            std::mem::size_of::<u32>() as u64
        );

        let copy_size = (max_events * std::mem::size_of::<Event>()) as u64;
        encoder.copy_buffer_to_buffer(
            &self.event_buffers[cpu_read_index], 0,
            &self.staging_events, 0,
            copy_size
        );

        true
    }

    /// Start async mapping of the staging buffers after command submission
    pub fn begin_async_mapping(&self) -> BufferMapping {
        let (sender, receiver) = mpsc::channel();

        // Start mapping both buffers
        let counter_slice = self.staging_counter.slice(..);
        let sender_clone = sender.clone();
        counter_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender_clone.send(res);
        });

        let event_slice = self.staging_events.slice(..);
        event_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        BufferMapping {
            receiver,
            completed: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
        }
    }

    /// Try to read events from the staging buffers if mapping is complete
    pub fn try_read_events(&self, mapping: &BufferMapping) -> Option<Vec<Event>> {
        loop {
            match mapping.receiver.try_recv() {
                Ok(result) => match result {
                    Ok(()) => {
                        mapping.completed.fetch_add(1, Ordering::AcqRel);
                    }
                    Err(_) => {
                        mapping.failed.store(true, Ordering::Release);
                    }
                },
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    mapping.failed.store(true, Ordering::Release);
                    break;
                }
            }
        }

        if mapping.failed.load(Ordering::Acquire) {
            self.staging_counter.unmap();
            self.staging_events.unmap();
            return Some(Vec::new());
        }

        if mapping.completed.load(Ordering::Acquire) < 2 {
            return None;
        }

        // Both mappings completed, read the data
        let event_count = {
            let counter_slice = self.staging_counter.slice(..);
            let counter_data = counter_slice.get_mapped_range();
            *bytemuck::from_bytes::<u32>(&counter_data)
        };

        let processed_events = {
            let event_slice = self.staging_events.slice(..);
            let event_data = event_slice.get_mapped_range();
            let events = bytemuck::cast_slice::<u8, Event>(&event_data);

            let mut state = self.state.lock();
            let mut processed_events = Vec::new();
            let start_idx = state.events_read;
            let mut end_idx = start_idx;

            if event_count > 0 {
                end_idx = (start_idx + event_count as usize).min(state.max_events);
                for i in start_idx..end_idx {
                    let event = events[i];
                    if event.event_type != 0 {
                        processed_events.push(event);
                    }
                }
            } else {
                // Fallback: scan for any non-zero events if counter appears stale.
                for (idx, event) in events.iter().enumerate().take(state.max_events) {
                    if event.event_type != 0 {
                        processed_events.push(*event);
                        end_idx = idx + 1;
                    }
                }
            }

            state.events_read = end_idx;
            processed_events
        };

        // Clean up mappings - all slices have been dropped by now
        self.staging_counter.unmap();
        self.staging_events.unmap();

        Some(processed_events)
    }

    pub fn read_events_blocking(&self, device: &wgpu::Device) -> Vec<Event> {
        let wait_for_map = |device: &wgpu::Device, receiver: &mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>| {
            loop {
                device.poll(wgpu::MaintainBase::Wait);
                match receiver.try_recv() {
                    Ok(Ok(())) => return true,
                    Ok(Err(_)) => return false,
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => return false,
                }
            }
        };

        let event_count = {
            let (sender, receiver) = mpsc::channel();
            let counter_slice = self.staging_counter.slice(..);
            counter_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = sender.send(res);
            });

            if !wait_for_map(device, &receiver) {
                return Vec::new();
            }

            let counter_data = counter_slice.get_mapped_range();
            let count = *bytemuck::from_bytes::<u32>(&counter_data);
            drop(counter_data);
            self.staging_counter.unmap();
            count
        };

        let (sender, receiver) = mpsc::channel();
        let event_slice = self.staging_events.slice(..);
        event_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        if !wait_for_map(device, &receiver) {
            return Vec::new();
        }

        let processed_events = {
            let event_data = event_slice.get_mapped_range();
            let events = bytemuck::cast_slice::<u8, Event>(&event_data);

            let mut state = self.state.lock();
            let mut processed_events = Vec::new();
            let start_idx = state.events_read;
            let mut end_idx = start_idx;

            if event_count > 0 {
                end_idx = (start_idx + event_count as usize).min(state.max_events);
                for i in start_idx..end_idx {
                    let event = events[i];
                    if event.event_type != 0 {
                        processed_events.push(event);
                    }
                }
            } else {
                for (idx, event) in events.iter().enumerate().take(state.max_events) {
                    if event.event_type != 0 {
                        processed_events.push(*event);
                        end_idx = idx + 1;
                    }
                }
            }

            state.events_read = end_idx;
            drop(event_data);
            processed_events
        };

        self.staging_events.unmap();

        processed_events
    }

    /// Get the GPU buffers for binding to shaders (event_buffer, counter_buffer)
    pub fn gpu_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        let state = self.state.lock();
        (&self.event_buffers[state.gpu_write_index], &self.counter_buffers[state.gpu_write_index])
    }

    /// Get the GPU buffers for a specific index (event_buffer, counter_buffer)
    pub fn gpu_buffers_for_index(&self, index: usize) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.event_buffers[index], &self.counter_buffers[index])
    }

    pub fn gpu_write_index(&self) -> usize {
        let state = self.state.lock();
        state.gpu_write_index
    }

    pub fn is_readback_in_flight(&self) -> bool {
        self.readback_in_flight.load(Ordering::Acquire)
    }

    pub fn start_readback(&self) {
        self.readback_in_flight.store(true, Ordering::Release);
    }

    pub fn finish_readback(&self) {
        self.readback_in_flight.store(false, Ordering::Release);
    }

    pub fn reset_both_counters(&self, queue: &wgpu::Queue) {
        // Reset both counter buffers to ensure clean state during reset
        for i in 0..2 {
            queue.write_buffer(&self.counter_buffers[i], 0, bytemuck::cast_slice(&[0u32]));
        }
        let mut state = self.state.lock();
        state.events_read = 0;
        self.readback_in_flight.store(false, Ordering::Release);
    }
}
