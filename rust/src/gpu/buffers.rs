// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use bytemuck::Zeroable;

use crate::gpu::structures::{
    Cell,
    CompiledRegulatoryUnit,
    GrnDescriptor,
    GenomeEntry,
    Lifeform,
    Link,
    PositionChangeEntry,
    SpeciesEntry,
    MAX_GRN_REGULATORY_UNITS,
    MAX_SPECIES_CAPACITY,
};
use crate::utils::math::Rect;
use crate::utils::gpu::gpu_vector::GpuVector;

const CELL_CAPACITY: usize = 100_000;
const CELL_HASH_TABLE_SIZE: usize = 1 << 16; // 65_536 buckets
const MAX_SPAWN_REQUESTS: usize = 512;
const LIFEFORM_CAPACITY: usize = 50_000;
const LINK_CAPACITY: usize = 30_000;
const LINK_FREE_LIST_CAPACITY: usize = LINK_CAPACITY;
const NUTRIENT_CELL_SIZE: u32 = 20;
const NUTRIENT_UNIT_SCALE: u32 = 4_000_000_000; // annoyingly we can't do atomicSubCompareExchangeWeak with f32 :sadge:
pub struct GpuBuffers {
    pub cell_vector_a: GpuVector<Cell>,
    pub cell_vector_b: GpuVector<Cell>,
    cell_read_buffer: AtomicBool,
    pub uniform_buffer: wgpu::Buffer,
    cell_counter: wgpu::Buffer,
    cell_counter_staging: wgpu::Buffer,
    cell_counter_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    lifeform_counter: wgpu::Buffer,
    lifeform_counter_staging: wgpu::Buffer,
    lifeform_counter_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    species_counter: wgpu::Buffer,
    species_counter_staging: wgpu::Buffer,
    species_counter_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    spawn_buffer: wgpu::Buffer,
    link_buffer: wgpu::Buffer,
    link_free_list: wgpu::Buffer,
    link_capacity: usize,
    nutrient_grid: RwLock<Arc<wgpu::Buffer>>,
    nutrient_grid_width: AtomicU32,
    nutrient_grid_height: AtomicU32,
    spatial_hash_bucket_heads: wgpu::Buffer,
    spatial_hash_bucket_heads_readonly: wgpu::Buffer,
    spatial_hash_next_indices: wgpu::Buffer,
    grn_descriptors: wgpu::Buffer,
    grn_units: wgpu::Buffer,
    lifeforms: wgpu::Buffer,
    lifeform_free_list: wgpu::Buffer,
    next_lifeform_id: wgpu::Buffer,
    genome_buffer: wgpu::Buffer,
    species_entries: wgpu::Buffer,
    species_free_list: wgpu::Buffer,
    next_species_id: wgpu::Buffer,
    next_gene_id: wgpu::Buffer,
    position_changes: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _cells: &[u8], // Unused - GPU handles all cell generation
        initial_uniforms: &[u8],
        bounds: Rect,
    ) -> Self {
        // GPU handles all cell generation, so we start with empty buffers
        // No need to write initial cell data - GPU will populate as needed
        let initial_cells: Vec<Cell> = Vec::new();
        let initial_count = 0u32;

        let cell_vector_a = GpuVector::<Cell>::new(
            device,
            CELL_CAPACITY,
            &initial_cells,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            Some("Cell Buffer A"),
        );

        let cell_vector_b = GpuVector::<Cell>::new(
            device,
            CELL_CAPACITY,
            &initial_cells,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            Some("Cell Buffer B"),
        );

        // Initialize free list with all indices free (since we start empty)
        cell_vector_a.initialize_free_list(queue, initial_count);
        cell_vector_b.initialize_free_list(queue, initial_count);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let cell_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Alive Counter"),
            contents: bytemuck::cast_slice(&[initial_count]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let cell_counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Alive Counter Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let lifeform_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeform Counter"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let lifeform_counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lifeform Counter Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let species_counter = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Species Counter"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let species_counter_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Species Counter Staging"),
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spawn_header_size = (std::mem::size_of::<u32>() * 2) as u64;
        let spawn_buffer_size = spawn_header_size
            + (MAX_SPAWN_REQUESTS * std::mem::size_of::<Cell>()) as u64;
        let spawn_zero = vec![0u8; spawn_buffer_size as usize];
        let spawn_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spawn Buffer"),
            contents: &spawn_zero,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let link_init = vec![Link::zeroed(); LINK_CAPACITY];
        let link_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Buffer"),
            contents: bytemuck::cast_slice(&link_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let mut link_free_list_init: Vec<u32> = Vec::with_capacity(LINK_FREE_LIST_CAPACITY + 2);
        link_free_list_init.push(LINK_FREE_LIST_CAPACITY as u32);
        link_free_list_init.push(0u32);
        link_free_list_init.extend((0..LINK_FREE_LIST_CAPACITY as u32).rev());
        let link_free_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Free List"),
            contents: bytemuck::cast_slice(&link_free_list_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let spatial_hash_init = vec![-1i32; CELL_HASH_TABLE_SIZE];
        let spatial_hash_bucket_heads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Spatial Hash Bucket Heads"),
            contents: bytemuck::cast_slice(&spatial_hash_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Readonly copy for fragment shader access (can't use atomic buffers in fragment shaders)
        let spatial_hash_bucket_heads_readonly = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Spatial Hash Bucket Heads Readonly"),
            contents: bytemuck::cast_slice(&spatial_hash_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let spatial_hash_next_indices_init = vec![-1i32; CELL_CAPACITY];
        let spatial_hash_next_indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Spatial Hash Next Indices"),
            contents: bytemuck::cast_slice(&spatial_hash_next_indices_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let grn_descriptor_init = vec![GrnDescriptor::zeroed(); LIFEFORM_CAPACITY];
        let grn_descriptors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GRN Descriptor Buffer"),
            contents: bytemuck::cast_slice(&grn_descriptor_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let total_grn_units = LIFEFORM_CAPACITY * MAX_GRN_REGULATORY_UNITS;
        let grn_units_init = vec![CompiledRegulatoryUnit::zeroed(); total_grn_units];
        let grn_units = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GRN Units Buffer"),
            contents: bytemuck::cast_slice(&grn_units_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let lifeforms_init = vec![Lifeform::zeroed(); LIFEFORM_CAPACITY];
        let lifeforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeforms Buffer"),
            contents: bytemuck::cast_slice(&lifeforms_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let genome_entries_init = vec![GenomeEntry::inactive(); LIFEFORM_CAPACITY];
        let genome_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Genome Buffer"),
            contents: bytemuck::cast_slice(&genome_entries_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let mut lifeform_free_init: Vec<u32> =
            Vec::with_capacity(LIFEFORM_CAPACITY + 2);
        lifeform_free_init.push(LIFEFORM_CAPACITY as u32);
        lifeform_free_init.push(0u32);
        lifeform_free_init.extend((0..LIFEFORM_CAPACITY as u32).rev());
        let lifeform_free_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeform Free List"),
            contents: bytemuck::cast_slice(&lifeform_free_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let next_lifeform_id = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Next Lifeform ID"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let species_entries_init = vec![SpeciesEntry::inactive(); MAX_SPECIES_CAPACITY];
        let species_entries = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Species Entries Buffer"),
            contents: bytemuck::cast_slice(&species_entries_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let mut species_free_init: Vec<u32> =
            Vec::with_capacity(MAX_SPECIES_CAPACITY + 2);
        species_free_init.push(MAX_SPECIES_CAPACITY as u32);
        species_free_init.push(0u32);
        species_free_init.extend((0..MAX_SPECIES_CAPACITY as u32).rev());
        let species_free_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Species Free List"),
            contents: bytemuck::cast_slice(&species_free_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let next_species_id = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Next Species ID"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let next_gene_id = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Next Gene ID"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let position_changes_init = vec![PositionChangeEntry::zero(); CELL_CAPACITY];
        let position_changes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Position Changes Buffer"),
            contents: bytemuck::cast_slice(&position_changes_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let grid_width = (bounds.width / NUTRIENT_CELL_SIZE as f32).ceil().max(1.0) as u32;
        let grid_height = (bounds.height / NUTRIENT_CELL_SIZE as f32).ceil().max(1.0) as u32;
        let nutrient_grid_size = (grid_width * grid_height) as usize;
        let initial_nutrients = vec![NUTRIENT_UNIT_SCALE; nutrient_grid_size];
        let nutrient_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nutrient Grid Buffer"),
            contents: bytemuck::cast_slice(&initial_nutrients),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let nutrient_grid = Arc::new(nutrient_grid);

        Self {
            cell_vector_a,
            cell_vector_b,
            cell_read_buffer: AtomicBool::new(false),
            uniform_buffer,
            cell_counter,
            cell_counter_staging,
            cell_counter_readback: Mutex::new(None),
            lifeform_counter,
            lifeform_counter_staging,
            lifeform_counter_readback: Mutex::new(None),
            species_counter,
            species_counter_staging,
            species_counter_readback: Mutex::new(None),
            spawn_buffer,
            link_buffer,
            link_free_list,
            link_capacity: LINK_CAPACITY,
            nutrient_grid: RwLock::new(Arc::clone(&nutrient_grid)),
            nutrient_grid_width: AtomicU32::new(grid_width),
            nutrient_grid_height: AtomicU32::new(grid_height),
            spatial_hash_bucket_heads,
            spatial_hash_bucket_heads_readonly,
            spatial_hash_next_indices,
            grn_descriptors,
            grn_units,
            lifeforms,
            lifeform_free_list,
            next_lifeform_id,
            next_gene_id,
            genome_buffer,
            species_entries,
            species_free_list,
            next_species_id,
            position_changes,
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

    pub fn cell_counter_buffer(&self) -> &wgpu::Buffer {
        &self.cell_counter
    }

    pub fn lifeform_counter_buffer(&self) -> &wgpu::Buffer {
        &self.lifeform_counter
    }

    pub fn species_counter_buffer(&self) -> &wgpu::Buffer {
        &self.species_counter
    }

    pub fn schedule_cell_counter_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.cell_counter_readback.lock();
            if guard.take().is_some() {
                self.cell_counter_staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.cell_counter,
            0,
            &self.cell_counter_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn begin_cell_counter_map(&self) {
        let mut guard = self.cell_counter_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.cell_counter_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_cell_counter(&self) -> Option<u32> {
        let mut receiver_guard = self.cell_counter_readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.cell_counter_staging.slice(..).get_mapped_range();
                let value = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
                drop(mapped);
                self.cell_counter_staging.unmap();
                *receiver_guard = None;
                Some(value)
            }
            Ok(Err(e)) => {
                eprintln!("Alive counter read failed: {:?}", e);
                self.cell_counter_staging.unmap();
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

    pub fn schedule_lifeform_counter_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.lifeform_counter_readback.lock();
            if guard.take().is_some() {
                self.lifeform_counter_staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.lifeform_counter,
            0,
            &self.lifeform_counter_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn begin_lifeform_counter_map(&self) {
        let mut guard = self.lifeform_counter_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.lifeform_counter_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_lifeform_counter(&self) -> Option<u32> {
        let mut receiver_guard = self.lifeform_counter_readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.lifeform_counter_staging.slice(..).get_mapped_range();
                let value = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
                drop(mapped);
                self.lifeform_counter_staging.unmap();
                *receiver_guard = None;
                Some(value)
            }
            Ok(Err(e)) => {
                eprintln!("Lifeform counter read failed: {:?}", e);
                self.lifeform_counter_staging.unmap();
                *receiver_guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Lifeform counter readback channel disconnected");
                *receiver_guard = None;
                None
            }
        }
    }

    pub fn schedule_species_counter_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.species_counter_readback.lock();
            if guard.take().is_some() {
                self.species_counter_staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.species_counter,
            0,
            &self.species_counter_staging,
            0,
            std::mem::size_of::<u32>() as u64,
        );
    }

    pub fn begin_species_counter_map(&self) {
        let mut guard = self.species_counter_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.species_counter_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_species_counter(&self) -> Option<u32> {
        let mut receiver_guard = self.species_counter_readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.species_counter_staging.slice(..).get_mapped_range();
                let value = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
                drop(mapped);
                self.species_counter_staging.unmap();
                *receiver_guard = None;
                Some(value)
            }
            Ok(Err(e)) => {
                eprintln!("Species counter read failed: {:?}", e);
                self.species_counter_staging.unmap();
                *receiver_guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Species counter readback channel disconnected");
                *receiver_guard = None;
                None
            }
        }
    }

    pub fn spawn_buffer(&self) -> &wgpu::Buffer {
        &self.spawn_buffer
    }
    
    pub fn link_buffer(&self) -> &wgpu::Buffer {
        &self.link_buffer
    }

    pub fn link_free_list_buffer(&self) -> &wgpu::Buffer {
        &self.link_free_list
    }

    pub fn link_capacity(&self) -> usize {
        self.link_capacity
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

    pub fn nutrient_grid_buffer(&self) -> Arc<wgpu::Buffer> {
        self.nutrient_grid.read().clone()
    }

    pub fn nutrient_grid_dimensions(&self) -> (u32, u32) {
        (
            self.nutrient_grid_width.load(Ordering::Relaxed),
            self.nutrient_grid_height.load(Ordering::Relaxed),
        )
    }

    pub fn nutrient_cell_size(&self) -> u32 {
        NUTRIENT_CELL_SIZE
    }
    pub fn nutrient_scale(&self) -> u32 {
        NUTRIENT_UNIT_SCALE
    }

    pub fn cell_hash_bucket_heads_buffer(&self) -> &wgpu::Buffer {
        &self.spatial_hash_bucket_heads
    }

    pub fn cell_hash_bucket_heads_readonly_buffer(&self) -> &wgpu::Buffer {
        &self.spatial_hash_bucket_heads_readonly
    }

    pub fn cell_hash_next_indices_buffer(&self) -> &wgpu::Buffer {
        &self.spatial_hash_next_indices
    }

    pub fn cell_hash_table_size(&self) -> usize {
        CELL_HASH_TABLE_SIZE
    }

    pub fn grn_descriptor_buffer(&self) -> &wgpu::Buffer {
        &self.grn_descriptors
    }

    pub fn grn_units_buffer(&self) -> &wgpu::Buffer {
        &self.grn_units
    }

    pub fn lifeforms_buffer(&self) -> &wgpu::Buffer {
        &self.lifeforms
    }

    pub fn lifeform_free_buffer(&self) -> &wgpu::Buffer {
        &self.lifeform_free_list
    }

    pub fn next_lifeform_id_buffer(&self) -> &wgpu::Buffer {
        &self.next_lifeform_id
    }

    pub fn genome_buffer(&self) -> &wgpu::Buffer {
        &self.genome_buffer
    }

    pub fn species_entries_buffer(&self) -> &wgpu::Buffer {
        &self.species_entries
    }

    pub fn species_free_buffer(&self) -> &wgpu::Buffer {
        &self.species_free_list
    }

    pub fn next_species_id_buffer(&self) -> &wgpu::Buffer {
        &self.next_species_id
    }

    pub fn next_gene_id_buffer(&self) -> &wgpu::Buffer {
        &self.next_gene_id
    }

    pub fn position_changes_buffer(&self) -> &wgpu::Buffer {
        &self.position_changes
    }

    pub fn resize_nutrient_grid(
        &self,
        device: &wgpu::Device,
        bounds: Rect,
    ) -> Arc<wgpu::Buffer> {
        let grid_width = (bounds.width / NUTRIENT_CELL_SIZE as f32).ceil().max(1.0) as u32;
        let grid_height = (bounds.height / NUTRIENT_CELL_SIZE as f32).ceil().max(1.0) as u32;
        let grid_size = (grid_width * grid_height) as usize;

        let initial_nutrients = vec![NUTRIENT_UNIT_SCALE; grid_size.max(1)];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Nutrient Grid Buffer"),
            contents: bytemuck::cast_slice(&initial_nutrients),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let buffer = Arc::new(buffer);

        {
            let mut guard = self.nutrient_grid.write();
            *guard = buffer.clone();
        }

        self.nutrient_grid_width
            .store(grid_width.max(1), Ordering::Relaxed);
        self.nutrient_grid_height
            .store(grid_height.max(1), Ordering::Relaxed);

        buffer
    }

    /// Reset all GPU buffers to their initial empty state
    pub fn reset(&self, device: &wgpu::Device, queue: &wgpu::Queue, bounds: Rect) {
        use crate::gpu::structures::{Cell, CompiledRegulatoryUnit, GrnDescriptor, GenomeEntry, Lifeform, Link, PositionChangeEntry, SpeciesEntry};
        
        // Reset cell vectors
        let initial_count = 0u32;
        
        // Clear cell buffers
        let cell_capacity = self.cell_capacity();
        let cell_zero_data = vec![Cell::zeroed(); cell_capacity];
        queue.write_buffer(self.cell_vector_a.buffer(), 0, bytemuck::cast_slice(&cell_zero_data));
        queue.write_buffer(self.cell_vector_b.buffer(), 0, bytemuck::cast_slice(&cell_zero_data));
        
        // Reset cell free lists
        self.cell_vector_a.initialize_free_list(queue, initial_count);
        self.cell_vector_b.initialize_free_list(queue, initial_count);
        
        // Reset counters
        queue.write_buffer(&self.cell_counter, 0, bytemuck::cast_slice(&[initial_count]));
        queue.write_buffer(&self.lifeform_counter, 0, bytemuck::cast_slice(&[0u32]));
        queue.write_buffer(&self.species_counter, 0, bytemuck::cast_slice(&[0u32]));
        
        // Reset spawn buffer
        let spawn_header_size = (std::mem::size_of::<u32>() * 2) as u64;
        let spawn_buffer_size = spawn_header_size + (MAX_SPAWN_REQUESTS * std::mem::size_of::<Cell>()) as u64;
        let spawn_zero = vec![0u8; spawn_buffer_size as usize];
        queue.write_buffer(&self.spawn_buffer, 0, &spawn_zero);
        
        // Reset link buffer
        let link_zero = vec![Link::zeroed(); self.link_capacity];
        queue.write_buffer(&self.link_buffer, 0, bytemuck::cast_slice(&link_zero));
        
        // Reset link free list
        let mut link_free_list_init: Vec<u32> = Vec::with_capacity(self.link_capacity + 2);
        link_free_list_init.push(self.link_capacity as u32);
        link_free_list_init.push(0u32);
        link_free_list_init.extend((0..self.link_capacity as u32).rev());
        queue.write_buffer(&self.link_free_list, 0, bytemuck::cast_slice(&link_free_list_init));
        
        // Reset spatial hash
        let spatial_hash_init = vec![-1i32; CELL_HASH_TABLE_SIZE];
        queue.write_buffer(&self.spatial_hash_bucket_heads, 0, bytemuck::cast_slice(&spatial_hash_init));
        queue.write_buffer(&self.spatial_hash_bucket_heads_readonly, 0, bytemuck::cast_slice(&spatial_hash_init));
        let spatial_hash_next_indices_init = vec![-1i32; cell_capacity];
        queue.write_buffer(&self.spatial_hash_next_indices, 0, bytemuck::cast_slice(&spatial_hash_next_indices_init));
        
        // Reset GRN descriptors
        let grn_descriptor_init = vec![GrnDescriptor::zeroed(); LIFEFORM_CAPACITY];
        queue.write_buffer(&self.grn_descriptors, 0, bytemuck::cast_slice(&grn_descriptor_init));
        
        // Reset GRN units
        let total_grn_units = LIFEFORM_CAPACITY * MAX_GRN_REGULATORY_UNITS;
        let grn_units_init = vec![CompiledRegulatoryUnit::zeroed(); total_grn_units];
        queue.write_buffer(&self.grn_units, 0, bytemuck::cast_slice(&grn_units_init));
        
        // Reset lifeforms
        let lifeforms_init = vec![Lifeform::zeroed(); LIFEFORM_CAPACITY];
        queue.write_buffer(&self.lifeforms, 0, bytemuck::cast_slice(&lifeforms_init));
        
        // Reset lifeform free list
        let mut lifeform_free_init: Vec<u32> = Vec::with_capacity(LIFEFORM_CAPACITY + 2);
        lifeform_free_init.push(LIFEFORM_CAPACITY as u32);
        lifeform_free_init.push(0u32);
        lifeform_free_init.extend((0..LIFEFORM_CAPACITY as u32).rev());
        queue.write_buffer(&self.lifeform_free_list, 0, bytemuck::cast_slice(&lifeform_free_init));
        
        // Reset next lifeform ID
        queue.write_buffer(&self.next_lifeform_id, 0, bytemuck::cast_slice(&[0u32]));
        
        // Reset genome buffer
        let genome_entries_init = vec![GenomeEntry::inactive(); LIFEFORM_CAPACITY];
        queue.write_buffer(&self.genome_buffer, 0, bytemuck::cast_slice(&genome_entries_init));
        
        // Reset species entries
        let species_entries_init = vec![SpeciesEntry::inactive(); MAX_SPECIES_CAPACITY];
        queue.write_buffer(&self.species_entries, 0, bytemuck::cast_slice(&species_entries_init));
        
        // Reset species free list
        let mut species_free_init: Vec<u32> = Vec::with_capacity(MAX_SPECIES_CAPACITY + 2);
        species_free_init.push(MAX_SPECIES_CAPACITY as u32);
        species_free_init.push(0u32);
        species_free_init.extend((0..MAX_SPECIES_CAPACITY as u32).rev());
        queue.write_buffer(&self.species_free_list, 0, bytemuck::cast_slice(&species_free_init));
        
        // Reset next species ID
        queue.write_buffer(&self.next_species_id, 0, bytemuck::cast_slice(&[0u32]));
        
        // Reset next gene ID
        queue.write_buffer(&self.next_gene_id, 0, bytemuck::cast_slice(&[0u32]));
        
        // Reset position changes
        let position_changes_init = vec![PositionChangeEntry::zero(); cell_capacity];
        queue.write_buffer(&self.position_changes, 0, bytemuck::cast_slice(&position_changes_init));
        
        // Reset nutrient grid
        let _ = self.resize_nutrient_grid(device, bounds);
    }

}