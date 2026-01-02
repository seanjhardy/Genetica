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


/// A counter with its staging buffer and readback channel for GPU readback
pub struct Counter {
    pub buffer: wgpu::Buffer,
    pub staging: wgpu::Buffer,
    pub readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
}

impl Counter {
    pub fn new(device: &wgpu::Device, label: &str, initial_value: u32) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{} Counter", label)),
            contents: bytemuck::cast_slice(&[initial_value]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Counter Staging", label)),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            staging,
            readback: Mutex::new(None),
        }
    }

    pub fn schedule_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.readback.lock();
            if guard.take().is_some() {
                self.staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &self.staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn begin_map(&self) {
        let mut guard = self.readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume(&self) -> Option<u32> {
        let mut receiver_guard = self.readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.staging.slice(..).get_mapped_range();
                let value = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
                drop(mapped);
                self.staging.unmap();
                *receiver_guard = None;
                Some(value)
            }
            Ok(Err(e)) => {
                eprintln!("Counter read failed: {:?}", e);
                self.staging.unmap();
                *receiver_guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Counter readback channel disconnected");
                *receiver_guard = None;
                None
            }
        }
    }

    pub fn has_pending_readback(&self) -> bool {
        self.readback.lock().is_some()
    }
}

