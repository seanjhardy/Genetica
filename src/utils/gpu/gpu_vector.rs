//! Generic GPU Vector with free list management - similar to C++ StaticGPUVector and GPUVector
//! Maintains a fixed-capacity buffer and tracks free slots for reuse
//! The free list buffer is the single source of truth - indices in the free list are free
//! Shaders can atomically add indices to the free list when items die
//!
//! # Example
//! ```no_run
//! use crate::gpu::gpu_vector::GpuVector;
//! use crate::gpu::structures::Cell;
//!
//! // Create a GPU vector for cells
//! let cell_vector = GpuVector::<Cell>::new(
//!     &device,
//!     1000, // capacity
//!     &initial_cells,
//!     wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
//!     Some("Cell Buffer"),
//! );
//!
//! // Allocate a new cell
//! let index = cell_vector.push(&device, &queue, new_cell);
//!
//! // Update a cell
//! cell_vector.update_item(&queue, index, updated_cell);
//!
//! // Mark cell as free when it dies (or let shader do it atomically)
//! cell_vector.mark_free(&device, &queue, index);
//! ```

use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Generic GPU-backed vector with free list management
/// Similar to C++ StaticGPUVector<T> and GPUVector<T>
/// The free list buffer is the single source of truth - items are marked free
/// by adding their index to the free list (can be done atomically in shaders)
pub struct GpuVector<T: Pod + Zeroable + Clone> {
    /// GPU buffer containing all elements (fixed capacity)
    buffer: wgpu::Buffer,
    /// GPU buffer containing free indices (can hold up to capacity free indices)
    /// Layout: [count: u32, indices: array<u32>]
    free_list_buffer: wgpu::Buffer,
    /// Current number of active elements (max index + 1)
    size: usize,
    /// Maximum capacity of the buffer
    capacity: usize,
    /// Number of free indices in the free_list_buffer (tracked on CPU for bounds checking)
    free_count: usize,
    /// Buffer usage flags
    usage: wgpu::BufferUsages,
    /// Type marker (for zero-sized type handling)
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod + Zeroable + Clone> GpuVector<T> {
    /// Create a new GPU vector with fixed capacity
    /// 
    /// # Arguments
    /// * `device` - WGPU device
    /// * `capacity` - Maximum number of elements
    /// * `initial_data` - Initial data to populate the buffer
    /// * `usage` - Buffer usage flags (STORAGE, COPY_DST, etc.)
    /// * `label` - Optional label for the buffer
    pub fn new(
        device: &wgpu::Device,
        capacity: usize,
        initial_data: &[T],
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        let size = initial_data.len();
        let actual_capacity = capacity.max(size);
        
        // Pad initial data to capacity with zero values
        let mut data = initial_data.to_vec();
        while data.len() < actual_capacity {
            data.push(bytemuck::Zeroable::zeroed());
        }
        
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(&data),
            usage,
        });
        
        // Create free list buffer (can hold up to capacity free indices)
        // Layout: [count: u32, indices: array<u32>]
        let free_list_label = label.map(|l| format!("{} Free List", l));
        let free_list_data = vec![0u32; actual_capacity + 1]; // +1 for count at offset 0
        let free_list_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: free_list_label.as_deref(),
            contents: bytemuck::cast_slice(&free_list_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        
        Self {
            buffer,
            free_list_buffer,
            size,
            capacity: actual_capacity,
            free_count: 0,
            usage,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the GPU buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    
    /// Get the free list GPU buffer (for shader binding)
    /// Layout: [count: u32, free_indices: array<u32>]
    pub fn free_list_buffer(&self) -> &wgpu::Buffer {
        &self.free_list_buffer
    }
    
    /// Get current size (number of active elements)
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Read free list count and indices from GPU
    /// Always reads from GPU (doesn't rely on CPU-side free_count for accuracy)
    pub fn read_free_list(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<u32> {
        // Read the count first (stored at offset 0)
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free List Count Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Free List Count Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.free_list_buffer, 0, &count_buffer, 0, std::mem::size_of::<u32>() as u64);
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
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected before mapping completed");
                }
            }
        }
        
        let mapped_range = buffer_slice.get_mapped_range();
        let count: u32 = bytemuck::cast_slice(&mapped_range)[0];
        drop(mapped_range);
        count_buffer.unmap();
        
        if count == 0 {
            return Vec::new();
        }
        
        // Read the free indices (stored starting at offset 4)
        let indices_size = (count as usize * std::mem::size_of::<u32>()) as u64;
        let indices_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free List Indices Buffer"),
            size: indices_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Free List Indices Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.free_list_buffer, 4, &indices_buffer, 0, indices_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = indices_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        
        loop {
            let _ = device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(_)) => break,
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected before mapping completed");
                }
            }
        }
        
        let mapped_range = buffer_slice.get_mapped_range();
        let indices: Vec<u32> = bytemuck::cast_slice(&mapped_range).to_vec();
        drop(mapped_range);
        indices_buffer.unmap();
        
        indices
    }

    /// Initialize the free list so that indices from `start_index` up to capacity are marked free.
    /// This overwrites the GPU free list buffer and should be called during setup.
    pub fn initialize_free_list(&self, queue: &wgpu::Queue, start_index: u32) {
        let capacity = self.capacity as u32;
        let start = start_index.min(capacity);
        let count = capacity.saturating_sub(start);
        let count_data = [count];
        queue.write_buffer(
            self.free_list_buffer(),
            0,
            bytemuck::cast_slice(&count_data),
        );
        if count > 0 {
            let mut indices = Vec::with_capacity(count as usize);
            // Hand out low indices first: store in descending order so pop-from-end yields small indices
            for idx in (start..capacity).rev() {
                indices.push(idx);
            }
            queue.write_buffer(
                self.free_list_buffer(),
                4,
                bytemuck::cast_slice(&indices),
            );
        }
    }

    /// Read only the number of free indices from GPU without fetching the entire list
    pub fn read_free_count(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> usize {
        let count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Free List Count Buffer"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Free List Count Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.free_list_buffer,
            0,
            &count_buffer,
            0,
            std::mem::size_of::<u32>() as u64,
        );
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
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected before mapping completed");
                }
            }
        }

        let mapped_range = buffer_slice.get_mapped_range();
        let count: u32 = bytemuck::cast_slice(&mapped_range)[0];
        drop(mapped_range);
        count_buffer.unmap();

        count as usize
    }
    
    /// Write free list count and indices to GPU
    fn write_free_list(&mut self, queue: &wgpu::Queue, indices: &[u32]) {
        let count = indices.len() as u32;
        self.free_count = indices.len();
        
        // Write count at offset 0
        queue.write_buffer(&self.free_list_buffer, 0, bytemuck::cast_slice(&[count]));
        
        // Write indices starting at offset 4 (after the count)
        if !indices.is_empty() {
            queue.write_buffer(&self.free_list_buffer, 4, bytemuck::cast_slice(indices));
        }
    }
    
    /// Mark an index as free (element was removed)
    /// This is a convenience method that requires device - use mark_free_with_device for explicit API
    /// For better performance, consider batching free operations
    pub fn mark_free(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) {
        self.mark_free_with_device(device, queue, index);
    }
    
    /// Mark an index as free (requires device for GPU read/write)
    pub fn mark_free_with_device(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) {
        if index >= self.capacity as u32 {
            return;
        }
        
        let mut free_indices = self.read_free_list(device, queue);
        
        // Check if already in list (avoid duplicates)
        if !free_indices.contains(&index) {
            free_indices.push(index);
            self.write_free_list(queue, &free_indices);
        }
    }
    
    /// Use a free index (removes it from free list, requires device)
    pub fn use_free_index_with_device(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Option<u32> {
        if self.free_count == 0 {
            return None;
        }
        
        let mut free_indices = self.read_free_list(device, queue);
        if let Some(index) = free_indices.pop() {
            self.write_free_list(queue, &free_indices);
            Some(index)
        } else {
            None
        }
    }
    
    /// Allocate a new element at the next available index
    /// Returns the index where the element was placed
    /// Requires device to read from free list on GPU
    pub fn push(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, value: T) -> Option<u32> {
        let index = if self.free_count > 0 {
            // Try to get a free index from GPU
            if let Some(free_idx) = self.use_free_index_with_device(device, queue) {
                free_idx
            } else {
                // Fallback: no capacity left if we can't allocate
                if self.size >= self.capacity {
                    return None;
                }
                let idx = self.size as u32;
                self.size += 1;
                idx
            }
        } else {
            // No free indices, append to end
            if self.size >= self.capacity {
                return None; // No capacity left
            }
            let idx = self.size as u32;
            self.size += 1;
            idx
        };
        
        // Write value to GPU buffer
        self.update_item(queue, index, value);
        Some(index)
    }
    
    /// Update an element at a specific index
    pub fn update_item(&self, queue: &wgpu::Queue, index: u32, value: T) {
        let offset = (index as usize * std::mem::size_of::<T>()) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&[value]));
    }
    
    /// Update all elements in a slice (starting from index 0)
    pub fn update_all(&self, queue: &wgpu::Queue, data: &[T]) {
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        }
    }
    
    /// Read element at specific index from GPU
    pub fn read_item(&self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) -> Option<T> {
        if index >= self.size as u32 {
            return None;
        }
        
        let item_size = std::mem::size_of::<T>();
        let offset = (index as usize * item_size) as u64;
        let buffer_size = item_size as u64;
        
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Item Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, offset, &staging_buffer, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        
        loop {
            let _ = device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(_)) => break,
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected before mapping completed");
                }
            }
        }
        
        let mapped_range = buffer_slice.get_mapped_range();
        let data: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
        drop(mapped_range);
        staging_buffer.unmap();
        
        data.first().cloned()
    }

    /// Read element at a specific index without checking current CPU-side size.
    /// Useful when the GPU mutates buffers and the CPU size is stale.
    pub fn read_item_unchecked(&self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) -> Option<T> {
        if index >= self.capacity as u32 {
            return None;
        }

        let item_size = std::mem::size_of::<T>();
        let offset = (index as usize * item_size) as u64;
        let buffer_size = item_size as u64;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Item Staging Buffer (Unchecked)"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder (Unchecked)"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, offset, &staging_buffer, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        loop {
            let _ = device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(_)) => break,
                Ok(Err(e)) => {
                    eprintln!("Buffer mapping failed: {:?}", e);
                    return None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Channel disconnected before mapping completed");
                    return None;
                }
            }
        }

        let mapped_range = buffer_slice.get_mapped_range();
        let data: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
        drop(mapped_range);
        staging_buffer.unmap();

        data.first().cloned()
    }
    
    /// Read all elements from GPU (up to size)
    /// Note: Does not filter by free list - the free list is the source of truth
    /// Callers should check if indices are in the free list before using
    pub fn read_all(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<T> {
        let item_size = std::mem::size_of::<T>();
        let buffer_size = (self.size * item_size) as u64;
        
        if buffer_size == 0 {
            return Vec::new();
        }
        
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, buffer_size);
        queue.submit(std::iter::once(encoder.finish()));
        
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        
        loop {
            let _ = device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(_)) => break,
                Ok(Err(e)) => panic!("Buffer mapping failed: {:?}", e),
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(100));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    panic!("Channel disconnected before mapping completed");
                }
            }
        }
        
        let mapped_range = buffer_slice.get_mapped_range();
        let data: Vec<T> = bytemuck::cast_slice(&mapped_range).to_vec();
        drop(mapped_range);
        staging_buffer.unmap();
        
        data
    }
    
    /// Read all active elements from GPU (excluding indices in free list)
    /// This reads both the data buffer and free list, then filters
    /// Note: This is less efficient - prefer using read_all() and checking free list separately
    pub fn read_active(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<T> {
        let all_data = self.read_all(device, queue);
        let free_indices = self.read_free_list(device, queue);
        let free_set: std::collections::HashSet<u32> = free_indices.into_iter().collect();
        
        all_data.into_iter()
            .enumerate()
            .filter(|(i, _)| !free_set.contains(&(*i as u32)))
            .map(|(_, item)| item)
            .collect()
    }
    
    /// Sync free list count from GPU (updates CPU-side free_count)
    /// Useful for keeping CPU-side count accurate after shader modifications
    pub fn sync_free_count(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let count = self.read_free_count(device, queue);
        self.free_count = count;
    }

}

/// Static GPU Vector - fixed capacity without free list management
/// Similar to C++ StaticGPUVector<T>
pub struct StaticGpuVector<T: Pod + Zeroable + Clone> {
    /// GPU buffer containing all elements (fixed capacity)
    buffer: wgpu::Buffer,
    /// Current number of elements
    size: usize,
    /// Maximum capacity of the buffer
    capacity: usize,
    /// Buffer usage flags
    usage: wgpu::BufferUsages,
    /// Type marker
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Pod + Zeroable + Clone> StaticGpuVector<T> {
    /// Create a new static GPU vector with fixed capacity
    pub fn new(
        device: &wgpu::Device,
        capacity: usize,
        initial_data: &[T],
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Self {
        let size = initial_data.len();
        let actual_capacity = capacity.max(size);
        
        // Pad initial data to capacity with zero values
        let mut data = initial_data.to_vec();
        while data.len() < actual_capacity {
            data.push(bytemuck::Zeroable::zeroed());
        }
        
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: bytemuck::cast_slice(&data),
            usage,
        });
        
        Self {
            buffer,
            size,
            capacity: actual_capacity,
            usage,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Get the GPU buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    
    /// Get current size
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Update element at specific index
    pub fn update_item(&self, queue: &wgpu::Queue, index: u32, value: T) {
        let offset = (index as usize * std::mem::size_of::<T>()) as u64;
        queue.write_buffer(&self.buffer, offset, bytemuck::cast_slice(&[value]));
    }
    
    /// Update all elements in a slice (starting from index 0)
    pub fn update_all(&self, queue: &wgpu::Queue, data: &[T]) {
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        }
    }
}
