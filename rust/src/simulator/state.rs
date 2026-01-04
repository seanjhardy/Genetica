use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use futures::channel::oneshot;
use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::{Mutex};
use std::sync::mpsc;

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
}

impl EventSystem {
    /// Create a new double-buffered event system
    pub fn new(device: &wgpu::Device) -> Self {
        let max_events = 1; // Much smaller buffer - most frames have few or no events

        // Create double-buffered event buffers (initialized with zeros)
        let event_buffer_size = (max_events * std::mem::size_of::<Event>()) as u64;
        let zeros = vec![0u8; event_buffer_size as usize];
        let event_buffers = [
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Buffer 0"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Event Buffer 1"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        }
    }


    /// Schedule copying the current read buffer to staging for CPU access
    pub fn schedule_read_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        let state = self.state.lock();
        // Copy counter from current read buffer
        encoder.copy_buffer_to_buffer(
            &self.counter_buffers[state.cpu_read_index], 0,
            &self.staging_counter, 0,
            std::mem::size_of::<u32>() as u64
        );

        // Copy all events from current read buffer
        let copy_size = (state.max_events * std::mem::size_of::<Event>()) as u64;
        encoder.copy_buffer_to_buffer(
            &self.event_buffers[state.cpu_read_index], 0,
            &self.staging_events, 0,
            copy_size
        );
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

        BufferMapping { receiver }
    }

    /// Try to read events from the staging buffers if mapping is complete
    pub fn try_read_events(&self, mapping: &BufferMapping) -> Vec<Event> {
        // Check if both mappings completed successfully (2 messages expected)
        let mut success_count = 0;
        for _ in 0..2 {
            match mapping.receiver.try_recv() {
                Ok(Ok(())) => success_count += 1,
                Ok(Err(_)) => return Vec::new(), // Mapping failed
                Err(_) => return Vec::new(), // Not ready yet
            }
        }

        if success_count != 2 {
            return Vec::new(); // Both mappings didn't complete successfully
        }

        // Both mappings completed, read the data
        let event_count = {
            let counter_slice = self.staging_counter.slice(..);
            let counter_data = counter_slice.get_mapped_range();
            *bytemuck::from_bytes::<u32>(&counter_data)
        };

        // Early return if no events - avoid expensive event buffer processing
        if event_count == 0 {
            // Clean up mappings - no slices are held at this point
            self.staging_counter.unmap();
            self.staging_events.unmap();
            return Vec::new();
        }

        // Read events only if there are events to process
        let processed_events = {
            let event_slice = self.staging_events.slice(..);
            let event_data = event_slice.get_mapped_range();
            let events = bytemuck::cast_slice::<u8, Event>(&event_data);

            let mut state = self.state.lock();
            // Read all events that haven't been read yet
            let mut processed_events = Vec::new();
            let start_idx = state.events_read;
            let end_idx = (start_idx + event_count as usize).min(state.max_events);

            for i in start_idx..end_idx {
                let event = events[i];
                if event.event_type != 0 {
                    processed_events.push(event);
                }
            }

            // Update how many we've read
            state.events_read = end_idx;

            processed_events
        };

        // Clean up mappings - all slices have been dropped by now
        self.staging_counter.unmap();
        self.staging_events.unmap();

        processed_events
    }

    /// Get the GPU buffers for binding to shaders (event_buffer, counter_buffer)
    pub fn gpu_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        let state = self.state.lock();
        (&self.event_buffers[state.gpu_write_index], &self.counter_buffers[state.gpu_write_index])
    }

    /// Reset the current write buffer's counter
    pub fn swap_buffers(&self) {
        let mut state = self.state.lock();
        let temp = state.gpu_write_index;
        state.gpu_write_index = state.cpu_read_index;
        state.cpu_read_index = temp;
        state.events_read = 0;
    }

    pub fn reset_write_counter(&self, queue: &wgpu::Queue) {
        let state = self.state.lock();
        queue.write_buffer(&self.counter_buffers[state.gpu_write_index], 0, bytemuck::cast_slice(&[0u32]));
    }

    pub fn reset_both_counters(&self, queue: &wgpu::Queue) {
        // Reset both counter buffers to ensure clean state during reset
        for i in 0..2 {
            queue.write_buffer(&self.counter_buffers[i], 0, bytemuck::cast_slice(&[0u32]));
        }
    }
}