// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::gpu::structures::{
    Cell, 
};
use crate::utils::gpu::gpu_vector::{GpuVector};


pub struct GpuBuffers {
    pub cell_vector_a: GpuVector<Cell>,
    pub cell_vector_b: GpuVector<Cell>,
    cell_read_buffer: AtomicBool,
    pub uniform_buffer: wgpu::Buffer,
    cell_event_buffer: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        cells: &[u8],
        initial_uniforms: &[u8],
    ) -> Self {
        // Create fixed-capacity buffer (pad to 1000 cells)
        const CELL_CAPACITY: usize = 10000;
        
        // Convert byte slices to typed slices
        let initial_cells: &[Cell] = bytemuck::cast_slice(cells);
        
        // Create TWO GPU vectors for cells (ping-pong buffers for double-buffering)
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
        
        // Buffer B starts with same data as A
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

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let event_capacity = cell_vector_a.capacity();
        let event_buffer_init = vec![0u32; event_capacity + 1];
        let cell_event_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Event Buffer"),
            contents: bytemuck::cast_slice(&event_buffer_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            cell_vector_a,
            cell_vector_b,
            cell_read_buffer: AtomicBool::new(false), // Start with A as read buffer
            uniform_buffer,
            cell_event_buffer,
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


    fn write_free_list(
        queue: &wgpu::Queue,
        vector: &GpuVector<Cell>,
        new_count: u32,
        remaining_indices: &[u32],
    ) {
        let count_data = [new_count];
        let count_bytes = bytemuck::cast_slice(&count_data);
        queue.write_buffer(vector.free_list_buffer(), 0, count_bytes);
        if !remaining_indices.is_empty() {
            let indices_bytes = bytemuck::cast_slice(remaining_indices);
            queue.write_buffer(vector.free_list_buffer(), 4, indices_bytes);
        }
    }

    /// Spawn a new cell by reusing a free slot (if available). Returns the populated index.
    pub fn spawn_cell(&self, device: &wgpu::Device, queue: &wgpu::Queue, cell: Cell) -> Option<u32> {
        let read_is_b = self.cell_read_buffer.load(Ordering::Acquire);
        let (write_vec, read_vec) = if read_is_b {
            (&self.cell_vector_a, &self.cell_vector_b)
        } else {
            (&self.cell_vector_b, &self.cell_vector_a)
        };

        let mut free_indices = write_vec.read_free_list(device, queue);
        let free_index = match free_indices.pop() {
            Some(idx) => idx,
            None => return None,
        };

        let new_free_count = free_indices.len() as u32;
        Self::write_free_list(queue, write_vec, new_free_count, &free_indices);
        Self::write_free_list(queue, read_vec, new_free_count, &free_indices);

        // Write the cell data into both buffers so render/compute stay in sync
        let offset = (free_index as usize * std::mem::size_of::<Cell>()) as u64;
        let cell_data = [cell];
        let cell_bytes = bytemuck::cast_slice(&cell_data);
        queue.write_buffer(write_vec.buffer(), offset, cell_bytes);
        queue.write_buffer(read_vec.buffer(), offset, cell_bytes);

        Some(free_index)
    }

    pub fn cell_event_buffer(&self) -> &wgpu::Buffer {
        &self.cell_event_buffer
    }

    pub fn reset_cell_events(&self, queue: &wgpu::Queue) {
        let zero = [0u32];
        queue.write_buffer(&self.cell_event_buffer, 0, bytemuck::cast_slice(&zero));
    }

    pub fn drain_cell_death_event_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> usize {
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Event Count Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Cell Event Count Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.cell_event_buffer, 0, &count_buffer, 0, std::mem::size_of::<u32>() as u64);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = count_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        loop {
            let _ = device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(_)) => break,
                Ok(Err(e)) => panic!("Cell event count read failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Cell event count channel disconnected");
                }
            }
        }

        let mapped_range = buffer_slice.get_mapped_range();
        let count: u32 = bytemuck::cast_slice(&mapped_range)[0];
        drop(mapped_range);
        count_buffer.unmap();

        if count == 0 {
            self.reset_cell_events(queue);
            return 0;
        }

        self.reset_cell_events(queue);
        count as usize
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

    /// Add a new cell to the GPU buffer (returns the index where it was placed)
    /// Adds to both buffers to keep them in sync (since we swap frequently)
    pub fn push_cell(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cell: Cell) -> Option<u32> {
        let write_buffer = if self.cell_read_buffer.load(Ordering::Acquire) {
            &mut self.cell_vector_a
        } else {
            &mut self.cell_vector_b
        };
        let index = write_buffer.push(device, queue, cell);
        // Also add to read buffer to keep them in sync
        if let Some(_idx) = index {
            let read_buffer = if self.cell_read_buffer.load(Ordering::Acquire) {
                &mut self.cell_vector_b
            } else {
                &mut self.cell_vector_a
            };
            read_buffer.push(device, queue, cell);
        }
        index
    }
}