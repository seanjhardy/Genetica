use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Ring-buffered event system for reliable GPU->CPU communication
const EVENT_RING_SIZE: usize = 3;

/// Tracks mapping state for a ring entry
enum EventReadState {
    Idle,
    CopyQueued,
    CopySubmitted,
    Mapping {
        counter_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        events_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        counter_done: bool,
        events_done: bool,
    },
}

pub struct EventSystem {
    /// GPU event buffers (ring buffered)
    event_buffers: Vec<wgpu::Buffer>,
    /// GPU atomic counters for events
    counter_buffers: Vec<wgpu::Buffer>,
    /// CPU staging buffer for reading events (per ring entry)
    staging_events: Vec<wgpu::Buffer>,
    /// CPU staging buffer for reading counter (per ring entry)
    staging_counters: Vec<wgpu::Buffer>,
    /// Mutable state protected by mutex
    state: Mutex<EventSystemState>,
    /// Per-entry readback state
    read_states: Vec<Mutex<EventReadState>>,
    /// Tracking whether a staging buffer is currently mapped (per entry)
    events_mapped: Vec<AtomicBool>,
    counters_mapped: Vec<AtomicBool>,
}

/// Mutable state of the EventSystem
struct EventSystemState {
    /// Current buffer index for GPU writing
    gpu_write_index: usize,
    /// Last buffer index that was scheduled for CPU reading
    last_read_index: usize,
    /// Maximum events per buffer
    max_events: usize,
    /// Number of buffers in the ring
    ring_size: usize,
    /// Last event count read from GPU (for debugging)
    last_event_count: u32,
    /// Last processed event count from a readback (for debugging)
    last_processed_count: usize,
    /// Last readback index scheduled (awaiting submit)
    last_scheduled_index: Option<usize>,
}

impl EventSystem {
    /// Create a new double-buffered event system
    pub fn new(device: &wgpu::Device) -> Self {
        let max_events = 2000; // Much smaller buffer - most frames have few or no events

        // Create ring-buffered event buffers (initialized with zeros)
        let event_buffer_size = (max_events * std::mem::size_of::<Event>()) as u64;
        let zeros = vec![0u8; event_buffer_size as usize];
        let mut event_buffers = Vec::with_capacity(EVENT_RING_SIZE);
        let mut counter_buffers = Vec::with_capacity(EVENT_RING_SIZE);
        let mut staging_events = Vec::with_capacity(EVENT_RING_SIZE);
        let mut staging_counters = Vec::with_capacity(EVENT_RING_SIZE);
        let mut read_states = Vec::with_capacity(EVENT_RING_SIZE);
        let mut events_mapped = Vec::with_capacity(EVENT_RING_SIZE);
        let mut counters_mapped = Vec::with_capacity(EVENT_RING_SIZE);
        for i in 0..EVENT_RING_SIZE {
            event_buffers.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Event Buffer {}", i)),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }));
            counter_buffers.push(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Event Counter {}", i)),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            }));
            staging_events.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Event Staging Buffer {}", i)),
                size: event_buffer_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            staging_counters.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Event Counter Staging {}", i)),
                size: std::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            read_states.push(Mutex::new(EventReadState::Idle));
            events_mapped.push(AtomicBool::new(false));
            counters_mapped.push(AtomicBool::new(false));
        }

        let state = EventSystemState {
            gpu_write_index: 0,
            last_read_index: EVENT_RING_SIZE.saturating_sub(1),
            max_events,
            ring_size: EVENT_RING_SIZE,
            last_event_count: 0,
            last_processed_count: 0,
            last_scheduled_index: None,
        };

        Self {
            event_buffers,
            counter_buffers,
            staging_events,
            staging_counters,
            state: Mutex::new(state),
            read_states,
            events_mapped,
            counters_mapped,
        }
    }

    /// Schedule copying the current read buffer to staging for CPU access.
    /// Returns false if a readback is already in flight.
    pub fn try_schedule_readback(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
    ) -> bool {
        let (read_index, write_index, max_events, _ring_size) = {
            let state = self.state.lock();
            let read_index = state.gpu_write_index;
            let next_write = (state.gpu_write_index + 1) % state.ring_size;
            (read_index, next_write, state.max_events, state.ring_size)
        };

        {
            let st = self.read_states[read_index].lock();
            if !matches!(*st, EventReadState::Idle) {
                // event readback: ring entry still busy, skipping schedule
                return false;
            }
        }

        if self.events_mapped[read_index].load(Ordering::Acquire)
            || self.counters_mapped[read_index].load(Ordering::Acquire)
        {
            // event readback: ring entry still mapped, skipping schedule
            return false;
        }

        {
            let mut state = self.state.lock();
            state.last_read_index = read_index;
            state.gpu_write_index = write_index;
            state.last_scheduled_index = Some(read_index);
        }

        *self.read_states[read_index].lock() = EventReadState::CopyQueued;

        // Prepare the next write buffer.
        queue.write_buffer(&self.counter_buffers[write_index], 0, bytemuck::cast_slice(&[0u32]));
        let zeros = vec![Event::zeroed(); max_events];
        queue.write_buffer(&self.event_buffers[write_index], 0, bytemuck::cast_slice(&zeros));

        // Copy last frame's buffer into its dedicated staging slot.
        encoder.copy_buffer_to_buffer(
            &self.counter_buffers[read_index], 0,
            &self.staging_counters[read_index], 0,
            std::mem::size_of::<u32>() as u64
        );
        let copy_size = (max_events * std::mem::size_of::<Event>()) as u64;
        encoder.copy_buffer_to_buffer(
            &self.event_buffers[read_index], 0,
            &self.staging_events[read_index], 0,
            copy_size
        );

        true
    }

    /// Kick off async mappings for any ring entries that have finished their GPU copy.
    pub fn begin_pending_mappings(&self) {
        for i in 0..self.read_states.len() {
            let mut st = self.read_states[i].lock();
            if !matches!(*st, EventReadState::CopySubmitted) {
                continue;
            }

            let counter_slice = self.staging_counters[i].slice(..);
            let (counter_tx, counter_rx) = mpsc::channel();
            counter_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = counter_tx.send(res);
            });

            let event_slice = self.staging_events[i].slice(..);
            let (events_tx, events_rx) = mpsc::channel();
            event_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = events_tx.send(res);
            });

            self.counters_mapped[i].store(true, Ordering::Release);
            self.events_mapped[i].store(true, Ordering::Release);
            *st = EventReadState::Mapping {
                counter_rx,
                events_rx,
                counter_done: false,
                events_done: false,
            };
        }
    }

    /// Mark the most recently scheduled readback as submitted so it can be mapped.
    pub fn mark_readback_submitted(&self) {
        let idx = {
            let mut state = self.state.lock();
            state.last_scheduled_index.take()
        };
        let Some(index) = idx else { return; };
        let mut st = self.read_states[index].lock();
        if matches!(*st, EventReadState::CopyQueued) {
            *st = EventReadState::CopySubmitted;
        }
    }

    /// Try to harvest any completed mappings. Returns all events read this call.
    pub fn drain_ready_events(&self) -> Vec<Event> {
        let mut all_events = Vec::new();
        for i in 0..self.read_states.len() {
            let mut st = self.read_states[i].lock();
            let (counter_rx, events_rx, counter_done, events_done) = match &mut *st {
                EventReadState::Mapping {
                    counter_rx,
                    events_rx,
                    counter_done,
                    events_done,
                } => (counter_rx, events_rx, counter_done, events_done),
                _ => continue,
            };

            if !*counter_done {
                match counter_rx.try_recv() {
                    Ok(Ok(())) => *counter_done = true,
                    Ok(Err(e)) => {
                        println!("event readback: counter mapping failed for slot {}: {:?}", i, e);
                        self.staging_counters[i].unmap();
                        self.staging_events[i].unmap();
                        self.counters_mapped[i].store(false, Ordering::Release);
                        self.events_mapped[i].store(false, Ordering::Release);
                        *st = EventReadState::Idle;
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        println!("event readback: counter mapping disconnected for slot {}", i);
                        self.staging_counters[i].unmap();
                        self.staging_events[i].unmap();
                        self.counters_mapped[i].store(false, Ordering::Release);
                        self.events_mapped[i].store(false, Ordering::Release);
                        *st = EventReadState::Idle;
                        continue;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }

            if !*events_done {
                match events_rx.try_recv() {
                    Ok(Ok(())) => *events_done = true,
                    Ok(Err(e)) => {
                        println!("event readback: events mapping failed for slot {}: {:?}", i, e);
                        self.staging_counters[i].unmap();
                        self.staging_events[i].unmap();
                        self.counters_mapped[i].store(false, Ordering::Release);
                        self.events_mapped[i].store(false, Ordering::Release);
                        *st = EventReadState::Idle;
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        println!("event readback: events mapping disconnected for slot {}", i);
                        self.staging_counters[i].unmap();
                        self.staging_events[i].unmap();
                        self.counters_mapped[i].store(false, Ordering::Release);
                        self.events_mapped[i].store(false, Ordering::Release);
                        *st = EventReadState::Idle;
                        continue;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }

            if *counter_done && *events_done {
                let event_count = {
                    let counter_slice = self.staging_counters[i].slice(..);
                    let counter_data = counter_slice.get_mapped_range();
                    *bytemuck::from_bytes::<u32>(&counter_data)
                };

                let processed_events = {
                    let event_slice = self.staging_events[i].slice(..);
                    let event_data = event_slice.get_mapped_range();
                    let events = bytemuck::cast_slice::<u8, Event>(&event_data);
                    let mut processed = Vec::new();
                    let max_events = {
                        let state = self.state.lock();
                        state.max_events
                    };
                    let end_idx = (event_count as usize).min(max_events);
                    if event_count > 0 {
                        for event in events.iter().take(end_idx) {
                            if event.event_type != 0 {
                                processed.push(*event);
                            }
                        }
                    } else {
                        // Fallback: scan for any non-zero events.
                        for event in events.iter().take(max_events) {
                            if event.event_type != 0 {
                                processed.push(*event);
                            }
                        }
                    }

                    processed
                };

                self.staging_counters[i].unmap();
                self.staging_events[i].unmap();
                self.counters_mapped[i].store(false, Ordering::Release);
                self.events_mapped[i].store(false, Ordering::Release);
                *st = EventReadState::Idle;

                {
                    let mut state = self.state.lock();
                    state.last_read_index = i;
                    state.last_event_count = event_count;
                    state.last_processed_count = processed_events.len();
                }

                all_events.extend(processed_events);
            }
        }

        all_events
    }

    // Legacy blocking readback helpers removed in favor of ring-buffered drain_ready_events.

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

    pub fn debug_snapshot(&self) -> (usize, usize, u32, usize) {
        let state = self.state.lock();
        (
            state.last_read_index,
            state.gpu_write_index,
            state.last_event_count,
            state.last_processed_count,
        )
    }

    pub fn ring_size(&self) -> usize {
        let state = self.state.lock();
        state.ring_size
    }

    pub fn has_pending_readback(&self) -> bool {
        for st in &self.read_states {
            match &*st.lock() {
                EventReadState::CopyQueued
                | EventReadState::CopySubmitted
                | EventReadState::Mapping { .. } => {
                    return true;
                }
                EventReadState::Idle => {}
            }
        }
        false
    }

    pub fn reset_both_counters(&self, queue: &wgpu::Queue) {
        for i in 0..self.counter_buffers.len() {
            queue.write_buffer(&self.counter_buffers[i], 0, bytemuck::cast_slice(&[0u32]));
        }
        let max_events = {
            let state = self.state.lock();
            state.max_events
        };
        let zeros = vec![Event::zeroed(); max_events];
        for buffer in &self.event_buffers {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&zeros));
        }
        {
            let mut state = self.state.lock();
            state.last_event_count = 0;
            state.last_processed_count = 0;
            state.last_scheduled_index = None;
        }
        for i in 0..self.read_states.len() {
            let mut st = self.read_states[i].lock();
            *st = EventReadState::Idle;
            if self.counters_mapped[i].swap(false, Ordering::AcqRel) {
                self.staging_counters[i].unmap();
            }
            if self.events_mapped[i].swap(false, Ordering::AcqRel) {
                self.staging_events[i].unmap();
            }
        }
        for flag in &self.events_mapped {
            flag.store(false, Ordering::Release);
        }
        for flag in &self.counters_mapped {
            flag.store(false, Ordering::Release);
        }
    }
}
