// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use crate::gpu::structures::Cell;
use crate::utils::gpu::gpu_vector::GpuVector;

const CELL_CAPACITY: usize = 10_000;
const MAX_SPAWN_REQUESTS: usize = 512;

pub struct GpuBuffers {
    pub cell_vector_a: GpuVector<Cell>,
    pub cell_vector_b: GpuVector<Cell>,
    cell_read_buffer: AtomicBool,
    pub uniform_buffer: wgpu::Buffer,
    alive_counter: wgpu::Buffer,
    alive_counter_staging: wgpu::Buffer,
    alive_counter_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    spawn_request_count: wgpu::Buffer,
    spawn_requests: wgpu::Buffer,
    spawn_capacity: usize,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cells: &[u8],
        initial_uniforms: &[u8],
    ) -> Self {
        let initial_cells: &[Cell] = bytemuck::cast_slice(cells);
        let initial_count = initial_cells.len() as u32;

        let cell_vector_a = GpuVector::<Cell>::new(
            device,
            CELL_CAPACITY,
            initial_cells,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            Some("Cell Buffer A"),
        );

        let cell_vector_b = GpuVector::<Cell>::new(
            device,
            CELL_CAPACITY,
            initial_cells,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            Some("Cell Buffer B"),
        );

        cell_vector_a.initialize_free_list(queue, initial_count);
        cell_vector_b.initialize_free_list(queue, initial_count);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let alive_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Alive Counter"),
            contents: bytemuck::cast_slice(&[initial_count]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let alive_counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alive Counter Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spawn_request_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spawn Request Count"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let spawn_requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn Requests Buffer"),
            size: (MAX_SPAWN_REQUESTS * std::mem::size_of::<Cell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cell_vector_a,
            cell_vector_b,
            cell_read_buffer: AtomicBool::new(false),
            uniform_buffer,
            alive_counter,
            alive_counter_staging,
            alive_counter_readback: Mutex::new(None),
            spawn_request_count,
            spawn_requests,
            spawn_capacity: MAX_SPAWN_REQUESTS,
        }
    }
    
    /// Get the read buffer (for rendering)
    pub fn cell_buffer_read(&self) -> &wgpu::Buffer {
        if self.cell_read_buffer.load(Ordering::Acquire) {
            self.cell_vector_b.buffer()
        } else {
            self.cell_vector_a.buffer()
        }
    }
    
    /// Get the write buffer (for compute)
    pub fn cell_buffer_write(&self) -> &wgpu::Buffer {
        if self.cell_read_buffer.load(Ordering::Acquire) {
            self.cell_vector_a.buffer()
        } else {
            self.cell_vector_b.buffer()
        }
    }
    
    /// Get the read buffer's free list (for rendering)
    pub fn cell_free_list_buffer_read(&self) -> &wgpu::Buffer {
        if self.cell_read_buffer.load(Ordering::Acquire) {
            self.cell_vector_b.free_list_buffer()
        } else {
            self.cell_vector_a.free_list_buffer()
        }
    }
    
    /// Get the write buffer's free list (for compute)
    pub fn cell_free_list_buffer_write(&self) -> &wgpu::Buffer {
        if self.cell_read_buffer.load(Ordering::Acquire) {
            self.cell_vector_a.free_list_buffer()
        } else {
            self.cell_vector_b.free_list_buffer()
        }
    }

    pub fn alive_counter_buffer(&self) -> &wgpu::Buffer {
        &self.alive_counter
    }

    pub fn schedule_alive_counter_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.alive_counter_readback.lock();
            if guard.take().is_some() {
                self.alive_counter_staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.alive_counter,
            0,
            &self.alive_counter_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn begin_alive_counter_map(&self) {
        let mut guard = self.alive_counter_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.alive_counter_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_alive_counter(&self, device: &wgpu::Device) -> Option<u32> {
        let mut receiver_guard = self.alive_counter_readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        let _ = device.poll(wgpu::MaintainBase::Poll);
        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.alive_counter_staging.slice(..).get_mapped_range();
                let value = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
                drop(mapped);
                self.alive_counter_staging.unmap();
                *receiver_guard = None;
                Some(value)
            }
            Ok(Err(e)) => {
                eprintln!("Alive counter read failed: {:?}", e);
                self.alive_counter_staging.unmap();
                *receiver_guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Alive counter readback channel disconnected");
                *receiver_guard = None;
                None
            }
        }
    }

    pub fn spawn_request_count_buffer(&self) -> &wgpu::Buffer {
        &self.spawn_request_count
    }

    pub fn spawn_requests_buffer(&self) -> &wgpu::Buffer {
        &self.spawn_requests
    }

    pub fn spawn_capacity(&self) -> usize {
        self.spawn_capacity
    }

    pub fn enqueue_spawn_requests(&self, queue: &wgpu::Queue, cells: &[Cell]) -> usize {
        let count = cells.len().min(self.spawn_capacity);
        if count == 0 {
            queue.write_buffer(&self.spawn_request_count, 0, bytemuck::cast_slice(&[0u32]));
            return 0;
        }

        let cells_to_write = &cells[..count];
        queue.write_buffer(
            &self.spawn_requests,
            0,
            bytemuck::cast_slice(cells_to_write),
        );
        queue.write_buffer(
            &self.spawn_request_count,
            0,
            bytemuck::cast_slice(&[count as u32]),
        );
        count
    }

    /// Get cell buffer (for pipelines) - DEPRECATED, use cell_buffer_read instead
    pub fn cell_buffer(&self) -> &wgpu::Buffer {
        self.cell_buffer_read()
    }
    
    /// Get cell free list buffer (for pipelines) - DEPRECATED, use cell_free_list_buffer_read instead
    pub fn cell_free_list_buffer(&self) -> &wgpu::Buffer {
        self.cell_free_list_buffer_read()
    }

    /// Get cell capacity
    pub fn cell_capacity(&self) -> usize {
        self.cell_vector_a.capacity() // Both have same capacity
    }
    
    /// Get cell size (number of initialized cells)
    pub fn cell_size(&self) -> usize {
        self.cell_vector_a.size() // Both should have same size
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &[u8]) {
        queue.write_buffer(&self.uniform_buffer, 0, uniforms);
    }

}