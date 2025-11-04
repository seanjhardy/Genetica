// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::gpu::structures::{
    Cell, Lifeform, 
    GpuGene, GpuPromoter, GpuEffector, GpuRegulatoryUnit, GpuGrnMetadata,
};
use crate::gpu::gpu_vector::{GpuVector, StaticGpuVector};
use crate::gpu::grn_converter::GrnGpuData;


pub struct GpuBuffers {
    pub cell_vector_a: GpuVector<Cell>,
    pub cell_vector_b: GpuVector<Cell>,
    // Track which buffer is currently for reading (render) vs writing (compute)
    // false = A is read, B is write
    // true = B is read, A is write
    // Use AtomicBool for thread-safe interior mutability
    cell_read_buffer: AtomicBool,
    pub lifeform_vector: GpuVector<Lifeform>,
    pub uniform_buffer: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        cells: &[u8],
        lifeforms: &[u8],
        initial_uniforms: &[u8],
        grn_data: Option<GrnGpuData>,
    ) -> Self {
        // Create fixed-capacity buffer (pad to 1000 cells)
        const CELL_CAPACITY: usize = 10000;
        const LIFEFORM_CAPACITY: usize = 1000;
        const GRN_METADATA_CAPACITY: usize = 1000;
        
        // GRN component capacities (estimate based on typical GRN sizes)
        const FACTOR_CAPACITY: usize = 50000;      // ~50 factors per lifeform * 1000 lifeforms
        const PROMOTER_CAPACITY: usize = 50000;
        const EFFECTOR_CAPACITY: usize = 50000;
        const REGULATORY_UNIT_CAPACITY: usize = 20000;
        const INDEX_CAPACITY: usize = 100000;     // Total indices across all regulatory units
        const AFFINITY_CAPACITY: usize = 1000000; // Large affinity matrices
        
        // Convert byte slices to typed slices
        let initial_cells: &[Cell] = bytemuck::cast_slice(cells);
        let initial_lifeforms: &[Lifeform] = bytemuck::cast_slice(lifeforms);
        
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
        
        let lifeform_vector = GpuVector::<Lifeform>::new(
            device,
            LIFEFORM_CAPACITY,
            initial_lifeforms,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Lifeform Buffer"),
        );

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Initialize GRN buffers if data provided
        let (_factor_vector, _promoter_vector, _effector_vector, _regulatory_unit_vector,
             _promoter_indices, _factor_indices,
             _promoter_factor_affinities, _factor_effector_affinities, _factor_receptor_affinities,
             _grn_metadata_vector) = if let Some(grn) = grn_data {
            // Create GPU vectors for GRN components
            let factor_vec = GpuVector::<GpuGene>::new(
                device,
                FACTOR_CAPACITY,
                &grn.factors,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor Buffer"),
            );
            
            let promoter_vec = GpuVector::<GpuPromoter>::new(
                device,
                PROMOTER_CAPACITY,
                &grn.promoters,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter Buffer"),
            );
            
            let effector_vec = GpuVector::<GpuEffector>::new(
                device,
                EFFECTOR_CAPACITY,
                &grn.effectors,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Effector Buffer"),
            );
            
            let regulatory_unit_vec = GpuVector::<GpuRegulatoryUnit>::new(
                device,
                REGULATORY_UNIT_CAPACITY,
                &grn.regulatory_units,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Regulatory Unit Buffer"),
            );
            
            let promoter_idx_vec = StaticGpuVector::<u32>::new(
                device,
                INDEX_CAPACITY,
                &grn.promoter_indices,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter Indices Buffer"),
            );
            
            let factor_idx_vec = StaticGpuVector::<u32>::new(
                device,
                INDEX_CAPACITY,
                &grn.factor_indices,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor Indices Buffer"),
            );
            
            let affinity_pf = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &grn.promoter_factor_affinities,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter-Factor Affinities Buffer"),
            );
            
            let affinity_fe = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &grn.factor_effector_affinities,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor-Effector Affinities Buffer"),
            );
            
            let affinity_fr = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &grn.factor_receptor_affinities,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor-Receptor Affinities Buffer"),
            );
            
            let grn_meta_vec = GpuVector::<GpuGrnMetadata>::new(
                device,
                GRN_METADATA_CAPACITY,
                &grn.grn_metadata,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("GRN Metadata Buffer"),
            );
            
            (factor_vec, promoter_vec, effector_vec, regulatory_unit_vec,
             promoter_idx_vec, factor_idx_vec,
             affinity_pf, affinity_fe, affinity_fr,
             grn_meta_vec)
        } else {
            // Initialize empty GRN buffers
            let empty_factors: Vec<GpuGene> = Vec::new();
            let empty_promoters: Vec<GpuPromoter> = Vec::new();
            let empty_effectors: Vec<GpuEffector> = Vec::new();
            let empty_reg_units: Vec<GpuRegulatoryUnit> = Vec::new();
            let empty_u32: Vec<u32> = Vec::new();
            let empty_f32: Vec<f32> = Vec::new();
            let empty_metadata: Vec<GpuGrnMetadata> = Vec::new();
            
            let factor_vec = GpuVector::<GpuGene>::new(
                device,
                FACTOR_CAPACITY,
                &empty_factors,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor Buffer"),
            );
            
            let promoter_vec = GpuVector::<GpuPromoter>::new(
                device,
                PROMOTER_CAPACITY,
                &empty_promoters,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter Buffer"),
            );
            
            let effector_vec = GpuVector::<GpuEffector>::new(
                device,
                EFFECTOR_CAPACITY,
                &empty_effectors,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Effector Buffer"),
            );
            
            let regulatory_unit_vec = GpuVector::<GpuRegulatoryUnit>::new(
                device,
                REGULATORY_UNIT_CAPACITY,
                &empty_reg_units,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Regulatory Unit Buffer"),
            );
            
            let promoter_idx_vec = StaticGpuVector::<u32>::new(
                device,
                INDEX_CAPACITY,
                &empty_u32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter Indices Buffer"),
            );
            
            let factor_idx_vec = StaticGpuVector::<u32>::new(
                device,
                INDEX_CAPACITY,
                &empty_u32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor Indices Buffer"),
            );
            
            let affinity_pf = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &empty_f32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Promoter-Factor Affinities Buffer"),
            );
            
            let affinity_fe = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &empty_f32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor-Effector Affinities Buffer"),
            );
            
            let affinity_fr = StaticGpuVector::<f32>::new(
                device,
                AFFINITY_CAPACITY,
                &empty_f32,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("Factor-Receptor Affinities Buffer"),
            );
            
            let grn_meta_vec = GpuVector::<GpuGrnMetadata>::new(
                device,
                GRN_METADATA_CAPACITY,
                &empty_metadata,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Some("GRN Metadata Buffer"),
            );
            
            (factor_vec, promoter_vec, effector_vec, regulatory_unit_vec,
             promoter_idx_vec, factor_idx_vec,
             affinity_pf, affinity_fe, affinity_fr,
             grn_meta_vec)
        };

        Self {
            cell_vector_a,
            cell_vector_b,
            cell_read_buffer: AtomicBool::new(false), // Start with A as read buffer
            lifeform_vector,
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
    
    /// Swap read/write buffers (thread-safe, can be called from any thread)
    pub fn swap_cell_buffers(&self) {
        // Toggle the flag atomically
        let current = self.cell_read_buffer.load(Ordering::Acquire);
        self.cell_read_buffer.store(!current, Ordering::Release);
    }
    
    /// Get cell buffer (for pipelines) - DEPRECATED, use cell_buffer_read instead
    pub fn cell_buffer(&self) -> &wgpu::Buffer {
        self.cell_buffer_read()
    }
    
    /// Get lifeform buffer (for pipelines)
    pub fn lifeform_buffer(&self) -> &wgpu::Buffer {
        self.lifeform_vector.buffer()
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

    /// Update lifeform at specific index
    pub fn update_lifeform(&self, queue: &wgpu::Queue, index: u32, lifeform: Lifeform) {
        self.lifeform_vector.update_item(queue, index, lifeform);
    }
    
    /// Update all lifeforms
    pub fn update_lifeforms(&self, queue: &wgpu::Queue, lifeforms: &[Lifeform]) {
        self.lifeform_vector.update_all(queue, lifeforms);
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

    /// Add a new lifeform to the GPU buffer (returns the index where it was placed)
    pub fn push_lifeform(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, lifeform: Lifeform) -> Option<u32> {
        self.lifeform_vector.push(device, queue, lifeform)
    }

    /// Mark a cell index as free (for reuse) - updates both buffers
    pub fn mark_cell_free(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) {
        self.cell_vector_a.mark_free(device, queue, index);
        self.cell_vector_b.mark_free(device, queue, index);
    }

    /// Mark a lifeform index as free (for reuse)
    pub fn mark_lifeform_free(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, index: u32) {
        self.lifeform_vector.mark_free(device, queue, index);
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

    /// Read lifeforms from GPU buffer
    pub fn read_lifeforms(&self, device: &wgpu::Device, queue: &wgpu::Queue, count: usize) -> Vec<u8> {
        // For StaticGpuVector, we need to manually read
        // This is a temporary solution - StaticGpuVector should have a read method
        let lifeform_size = std::mem::size_of::<Lifeform>();
        let buffer_size = (count * lifeform_size) as u64;
        
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lifeform Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(self.lifeform_vector.buffer(), 0, &staging_buffer, 0, buffer_size);
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
        let data = mapped_range.iter().copied().collect::<Vec<u8>>();
        drop(mapped_range);
        staging_buffer.unmap();
        
        data
    }
    
    /// Sync free list count from GPU (useful after shader modifications)
    /// Syncs both buffers to keep them in sync
    pub fn sync_free_cell_count(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.cell_vector_a.sync_free_count(device, queue);
        self.cell_vector_b.sync_free_count(device, queue);
    }
}

/// Timestamp query buffers for GPU profiling
pub struct TimestampBuffers {
    pub compute_timestamp_set: Option<wgpu::QuerySet>,
    pub render_timestamp_set: Option<wgpu::QuerySet>,
    pub timestamp_buffer: Option<wgpu::Buffer>,
}

impl TimestampBuffers {
    pub fn new(device: &wgpu::Device) -> Self {
        if device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let compute_ts = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Compute Timestamp Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });

            let render_ts = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Render Timestamp Query Set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });

            // Buffer size must account for alignment (256 bytes per query set)
            let query_align = 256u64;
            let buffer_size = query_align * 2; // 2 query sets, each aligned to 256 bytes

            let ts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Timestamp Buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            Self {
                compute_timestamp_set: Some(compute_ts),
                render_timestamp_set: Some(render_ts),
                timestamp_buffer: Some(ts_buffer),
            }
        } else {
            Self {
                compute_timestamp_set: None,
                render_timestamp_set: None,
                timestamp_buffer: None,
            }
        }
    }
}

