// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use bytemuck::Zeroable;

use crate::gpu::structures::{
    Cell, CompiledRegulatoryUnit, GrnDescriptor, Link, MAX_GRN_REGULATORY_UNITS, VerletPoint
};
use crate::simulator::state::{Counter, EventSystem};
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

pub struct GpuBuffers {
    pub points: GpuVector<VerletPoint>,
    pub cells: GpuVector<Cell>,
    pub lifeform_id: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub event_system: EventSystem,
    pub link_buffer: wgpu::Buffer,
    pub link_free_list: wgpu::Buffer,
    pub link_capacity: usize,
    pub nutrient_grid: wgpu::Buffer,
    pub nutrient_grid_width: AtomicU32,
    pub nutrient_grid_height: AtomicU32,
    pub grn_descriptors: wgpu::Buffer,
    pub grn_units: wgpu::Buffer,
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

        let points = GpuVector::<VerletPoint>::new(
            device,
            POINT_CAPACITY,
            &Vec::new(),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            Some("Physics Buffer"),
        );
        points.initialize_free_list(queue, 0);

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
        cells.initialize_free_list(queue, 0);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // event system
        let event_system = EventSystem::new(device);

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

        // Allocate compacted GRN units buffer (total units across all lifeforms)
        // For now, allocate a reasonable maximum - in production this would be dynamic
        let max_total_grn_units = LIFEFORM_CAPACITY * MAX_GRN_REGULATORY_UNITS;
        let grn_units_init = vec![CompiledRegulatoryUnit::zeroed(); max_total_grn_units];
        let grn_units = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GRN Units Buffer (Compacted)"),
            contents: bytemuck::cast_slice(&grn_units_init),
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

        let points_counter = Counter::new(device, "Points", 0);
        let cells_counter = Counter::new(device, "Cells", 0);

        let lifeform_id = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeform ID Buffer"),
            contents: bytemuck::cast_slice(&[1u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        Self {
            points,
            cells,
            uniform_buffer,
            event_system,
            link_buffer,
            link_free_list,
            link_capacity: LINK_CAPACITY,
            lifeform_id,
            nutrient_grid,
            nutrient_grid_width: AtomicU32::new(grid_width),
            nutrient_grid_height: AtomicU32::new(grid_height),
            grn_descriptors,
            grn_units,
            points_counter,
            cells_counter,
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
        use crate::gpu::structures::{Cell, CompiledRegulatoryUnit, GrnDescriptor, Link};

        let initial_count = 0u32;
        let cell_capacity = self.cells.capacity();
        let points_zero = vec![VerletPoint::zeroed(); POINT_CAPACITY];
        queue.write_buffer(self.points.buffer(), 0, bytemuck::cast_slice(&points_zero));
        self.points.initialize_free_list(queue, 0);
        queue.write_buffer(&self.points_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));
        let cell_zero_data = vec![Cell::zeroed(); cell_capacity];
        queue.write_buffer(self.cells.buffer(), 0, bytemuck::cast_slice(&cell_zero_data));
        self.cells.initialize_free_list(queue, 0);

        // Reset counters
        queue.write_buffer(&self.cells_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));
        queue.write_buffer(&self.points_counter.buffer, 0, bytemuck::cast_slice(&[initial_count]));

        // Reset lifeform ID counter to 1 (same as initialization)
        queue.write_buffer(&self.lifeform_id, 0, bytemuck::cast_slice(&[1u32]));

        // Reset event system - reset both buffers to be safe
        self.event_system.reset_both_counters(queue);
        
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
        
        // Reset nutrient grid
        let _ = self.resize_nutrient_grid(device, bounds);
    }

}