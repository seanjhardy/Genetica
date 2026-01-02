// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use bytemuck::Zeroable;

use crate::gpu::structures::{
    Cell,
    CompiledRegulatoryUnit,
    GrnDescriptor,
    GenomeEntry,
    GenomeEvent,
    Lifeform,
    Link,
    PositionChangeEntry,
    SpeciesEntry,
    SpawnRequest,
    VerletPoint,
    MAX_GRN_REGULATORY_UNITS,
    MAX_GENOME_EVENTS,
    MAX_SPECIES_CAPACITY,
};
use crate::simulator::state::Counter;
use crate::utils::math::Rect;
use crate::utils::gpu::gpu_vector::GpuVector;

pub const CELL_CAPACITY: usize = 1_000_000;
const MAX_SPAWN_REQUESTS: usize = 512;
pub const POINT_CAPACITY: usize = 1_000_000;
const LIFEFORM_CAPACITY: usize = 50_000;
pub const LINK_CAPACITY: usize = 30_000;
const LINK_FREE_LIST_CAPACITY: usize = LINK_CAPACITY;
const NUTRIENT_CELL_SIZE: u32 = 20;
const NUTRIENT_UNIT_SCALE: u32 = 4_000_000_000; // annoyingly we can't do atomicSubCompareExchangeWeak with f32 :sadge:

/// Types of counters that can be read back from the GPU
#[derive(Debug, Clone, Copy)]
pub enum CounterType {
    Points,
    Cells,
}

pub struct GpuBuffers {
    pub points: GpuVector<VerletPoint>,
    pub cells: GpuVector<Cell>,
    pub uniform_buffer: wgpu::Buffer,
    pub spawn_buffer: wgpu::Buffer,
    pub genome_event_buffer: wgpu::Buffer,
    pub link_buffer: wgpu::Buffer,
    pub link_free_list: wgpu::Buffer,
    pub link_capacity: usize,
    pub nutrient_grid: wgpu::Buffer,
    pub nutrient_grid_width: AtomicU32,
    pub nutrient_grid_height: AtomicU32,
    pub grn_descriptors: wgpu::Buffer,
    pub grn_units: wgpu::Buffer,
    pub lifeforms: wgpu::Buffer,
    pub lifeform_free_list: wgpu::Buffer,
    pub next_lifeform_id: wgpu::Buffer,
    pub genome_buffer: wgpu::Buffer,
    pub species_entries: wgpu::Buffer,
    pub species_free_list: wgpu::Buffer,
    pub next_species_id: wgpu::Buffer,
    pub next_gene_id: wgpu::Buffer,
    pub position_changes: wgpu::Buffer,
    pub points_counter: Counter,
    pub cells_counter: Counter,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _cells: &[u8], // Unused - GPU handles all cell generation
        initial_uniforms: &[u8],
        bounds: Rect,
    ) -> Self {
        let initial_cells: Vec<Cell> = Vec::new();
        let initial_count = 0u32;

        let points = GpuVector::<VerletPoint>::new(
            device,
            POINT_CAPACITY,
            &Vec::new(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Physics Buffer"),
        );
        points.initialize_free_list(queue, initial_count);

        let cells = GpuVector::<Cell>::new(
            device,
            CELL_CAPACITY,
            &initial_cells,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            Some("Cell Buffer"),
        );
        cells.initialize_free_list(queue, initial_count);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let spawn_header_size = (std::mem::size_of::<u32>() * 2) as u64;
        let spawn_buffer_size = spawn_header_size
            + (MAX_SPAWN_REQUESTS * std::mem::size_of::<SpawnRequest>()) as u64;
        let spawn_init: Vec<u8> = vec![0u8; spawn_buffer_size as usize];
        
        // leave spawn buffer empty; GPU ensure_min_population will enqueue
        let spawn_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Spawn Buffer"),
            contents: &spawn_init,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        // Genome event buffer (division copy/mutate queue)
        let genome_event_header = (std::mem::size_of::<u32>() * 2) as u64;
        let genome_event_size = (MAX_GENOME_EVENTS * std::mem::size_of::<GenomeEvent>()) as u64;
        let genome_event_buffer_size = genome_event_header + genome_event_size;
        let genome_event_zero = vec![0u8; genome_event_buffer_size as usize];
        let genome_event_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Genome Event Buffer"),
            contents: &genome_event_zero,
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

        let points_counter = Counter::new(device, "Points", initial_count);
        let cells_counter = Counter::new(device, "Cells", initial_count);

        Self {
            points,
            cells,
            uniform_buffer,
           spawn_buffer,
            genome_event_buffer,
            link_buffer,
            link_free_list,
            link_capacity: LINK_CAPACITY,
            nutrient_grid,
            nutrient_grid_width: AtomicU32::new(grid_width),
            nutrient_grid_height: AtomicU32::new(grid_height),
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
            points_counter,
            cells_counter,
        }
    }
    

    /// Consolidated counter readback methods
    pub fn has_counter_pending_readback(&self, counter_type: CounterType) -> bool {
        match counter_type {
            CounterType::Points => self.points_counter.has_pending_readback(),
            CounterType::Cells => self.cells_counter.has_pending_readback(),
        }
    }

    pub fn schedule_counter_copy(&self, counter_type: CounterType, encoder: &mut wgpu::CommandEncoder) {
        match counter_type {
            CounterType::Points => self.points_counter.schedule_copy(encoder),
            CounterType::Cells => self.cells_counter.schedule_copy(encoder),
        }
    }

    pub fn begin_counter_map(&self, counter_type: CounterType) {
        match counter_type {
            CounterType::Points => self.points_counter.begin_map(),
            CounterType::Cells => self.cells_counter.begin_map(),
        }
    }

    pub fn try_consume_counter(&self, counter_type: CounterType) -> Option<u32> {
        match counter_type {
            CounterType::Points => self.points_counter.try_consume(),
            CounterType::Cells => self.cells_counter.try_consume(),
        }
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &[u8]) {
        queue.write_buffer(&self.uniform_buffer, 0, uniforms);
    }

    pub fn nutrient_grid_dimensions(&self) -> (u32, u32) {
        (
            self.nutrient_grid_width.load(Ordering::Relaxed),
            self.nutrient_grid_height.load(Ordering::Relaxed),
        )
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

        self.nutrient_grid_width
            .store(grid_width.max(1), Ordering::Relaxed);
        self.nutrient_grid_height
            .store(grid_height.max(1), Ordering::Relaxed);

        buffer
    }

    /// Reset all GPU buffers to their initial empty state
    pub fn reset(&self, device: &wgpu::Device, queue: &wgpu::Queue, bounds: Rect) {
        use crate::gpu::structures::{Cell, CompiledRegulatoryUnit, GrnDescriptor, GenomeEntry, Lifeform, Link, PositionChangeEntry, SpeciesEntry};
        
        let initial_count = 0u32;
        let cell_capacity = self.cells.capacity();
        let points_zero = vec![VerletPoint::zeroed(); POINT_CAPACITY];
        queue.write_buffer(self.points.buffer(), 0, bytemuck::cast_slice(&points_zero));
        self.points.initialize_free_list(queue, initial_count);
        queue.write_buffer(&self.points_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));
        let cell_zero_data = vec![Cell::zeroed(); cell_capacity];
        queue.write_buffer(self.cells.buffer(), 0, bytemuck::cast_slice(&cell_zero_data));
        self.cells.initialize_free_list(queue, initial_count);

        // Reset counterss
        queue.write_buffer(&self.cells_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));
        queue.write_buffer(&self.points_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));
        
        // Reset spawn buffer
        let spawn_header_size = (std::mem::size_of::<u32>() * 2) as u64;
        let spawn_buffer_size = spawn_header_size + (MAX_SPAWN_REQUESTS * std::mem::size_of::<SpawnRequest>()) as u64;
        let spawn_init: Vec<u8> = vec![0u8; spawn_buffer_size as usize];
        // keep spawn buffer empty on reset; GPU will repopulate as needed
        queue.write_buffer(&self.spawn_buffer, 0, &spawn_init);

        // Reset genome event buffer
        let genome_event_header = (std::mem::size_of::<u32>() * 2) as u64;
        let genome_event_size = (MAX_GENOME_EVENTS * std::mem::size_of::<GenomeEvent>()) as u64;
        let genome_event_buffer_size = genome_event_header + genome_event_size;
        let genome_event_zero = vec![0u8; genome_event_buffer_size as usize];
        queue.write_buffer(&self.genome_event_buffer, 0, &genome_event_zero);
        
        // Reset link buffer
        let link_zero = vec![Link::zeroed(); self.link_capacity];
        queue.write_buffer(&self.link_buffer, 0, bytemuck::cast_slice(&link_zero));
        
        // Reset link free list
        let mut link_free_list_init: Vec<u32> = Vec::with_capacity(self.link_capacity + 2);
        link_free_list_init.push(self.link_capacity as u32);
        link_free_list_init.push(0u32);
        link_free_list_init.extend((0..self.link_capacity as u32).rev());
        queue.write_buffer(&self.link_free_list, 0, bytemuck::cast_slice(&link_free_list_init));
        
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