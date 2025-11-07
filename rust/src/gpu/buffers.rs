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

        Self {
            cell_vector_a,
            cell_vector_b,
            cell_read_buffer: AtomicBool::new(false), // Start with A as read buffer
            uniform_buffer,
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

    /// Get the read buffer's free list length (for rendering)
    pub fn free_cells_count(&self) -> usize {
        if self.cell_read_buffer.load(Ordering::Acquire) {
            self.cell_vector_b.free_count()
        } else {
            self.cell_vector_a.free_count()
        }
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

    /// Update cell at specific index (updates write buffer)
    pub fn update_cell(&self, queue: &wgpu::Queue, index: u32, cell: Cell) {
        let write_buffer = if self.cell_read_buffer.load(Ordering::Acquire) {
            &self.cell_vector_a
        } else {
            &self.cell_vector_b
        };
        write_buffer.update_item(queue, index, cell);
    }
    
    /// Update all cells (updates write buffer)
    pub fn update_cells(&self, queue: &wgpu::Queue, cells: &[Cell]) {
        let write_buffer = if self.cell_read_buffer.load(Ordering::Acquire) {
            &self.cell_vector_a
        } else {
            &self.cell_vector_b
        };
        write_buffer.update_all(queue, cells);
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


    /// Mark a cell index as free (for reuse) - updates both buffers
    pub fn mark_cell_free(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) {
        self.cell_vector_a.mark_free(device, queue, index);
        self.cell_vector_b.mark_free(device, queue, index);
    }

    /// Read cells from GPU buffer (for cleanup)
    /// Returns all cells from read buffer - caller should check free list if needed
    pub fn read_cells(&self, device: &wgpu::Device, queue: &wgpu::Queue, _count: usize) -> Vec<u8> {
        // Read all cells from read buffer and convert to bytes
        // Note: Does not filter by free list - check free list separately if needed
        let read_buffer = if self.cell_read_buffer.load(Ordering::Acquire) {
            &self.cell_vector_b
        } else {
            &self.cell_vector_a
        };
        let cells = read_buffer.read_all(device, queue);
        bytemuck::cast_slice(&cells).to_vec()
    }
    
    /// Sync free list count from GPU (useful after shader modifications)
    /// Syncs both buffers to keep them in sync
    pub fn sync_free_cell_count(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.cell_vector_a.sync_free_count(device, queue);
        self.cell_vector_b.sync_free_count(device, queue);
    }
}