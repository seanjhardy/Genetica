use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use futures::channel::oneshot;
use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::{Mutex};
use std::sync::mpsc;


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
            _ => return last,
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
            Ok(Err(_e)) => {
                self.staging.unmap();
                *st = ReadState::Idle;
                last
            }
            Err(mpsc::TryRecvError::Empty) => last,
            Err(mpsc::TryRecvError::Disconnected) => {
                *st = ReadState::Idle;
                last
            }
        }
    }
}