// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use bytemuck::{pod_read_unaligned, Zeroable};
use puffin::profile_scope;
use crate::gpu::structures::{
    Cell,
    CellEvent,
    CompiledGrn,
    CompiledRegulatoryUnit,
    DivisionRequest,
    GrnDescriptor,
    Link,
    LinkEvent,
    LifeformState,
    MAX_GRN_REGULATORY_UNITS,
};
use crate::utils::math::Rect;
use crate::utils::gpu::gpu_vector::GpuVector;

const CELL_CAPACITY: usize = 200_000;
const CELL_HASH_TABLE_SIZE: usize = 1 << 16; // 65_536 buckets
const MAX_SPAWN_REQUESTS: usize = 512;
const LIFEFORM_CAPACITY: usize = 20_000;
const MAX_DIVISION_REQUESTS: usize = 512;
const LINK_CAPACITY: usize = 40_000;
const LINK_FREE_LIST_CAPACITY: usize = LINK_CAPACITY;
const MAX_CELL_EVENTS: usize = 1_024;
const MAX_LINK_EVENTS: usize = 1_024;
pub const EVENT_STAGING_RING_SIZE: usize = 3;
const NUTRIENT_CELL_SIZE: u32 = 20;
const NUTRIENT_UNIT_SCALE: u32 = 4_000_000_000; // annoyingly we can't do atomicSubCompareExchangeWeak with f32 :sadge:
const AVERAGE_GRN_UNITS_PER_LIFEFORM: usize = 16;
const MAX_TOTAL_GRN_UNITS: usize = LIFEFORM_CAPACITY * AVERAGE_GRN_UNITS_PER_LIFEFORM;

struct GrnAllocationState {
    units: Vec<CompiledRegulatoryUnit>,
    free_lists: Vec<Vec<u32>>,
    next_free_offset: u32,
    slot_ranges: Vec<Option<(u32, u32)>>,
}

impl GrnAllocationState {
    fn new(units: Vec<CompiledRegulatoryUnit>) -> Self {
        let mut free_lists = Vec::with_capacity(MAX_GRN_REGULATORY_UNITS + 1);
        free_lists.resize_with(MAX_GRN_REGULATORY_UNITS + 1, Vec::new);
        Self {
            units,
            free_lists,
            next_free_offset: 0,
            slot_ranges: vec![None; LIFEFORM_CAPACITY],
        }
    }

    fn release_slot(&mut self, slot: usize) {
        if slot >= self.slot_ranges.len() {
            return;
        }
        if let Some((offset, len)) = self.slot_ranges[slot].take() {
            if len == 0 {
                return;
            }
            let len_usize = len as usize;
            if len_usize < self.free_lists.len() {
                self.free_lists[len_usize].push(offset);
            }
        }
    }

    fn allocate_slot(&mut self, slot: usize, len: u32) -> Option<u32> {
        if slot >= self.slot_ranges.len() {
            return None;
        }
        self.release_slot(slot);
        if len == 0 {
            return Some(0);
        }
        let len_usize = len as usize;
        if len_usize >= self.free_lists.len() {
            return None;
        }

        if let Some(offset) = self.free_lists[len_usize].pop() {
            self.slot_ranges[slot] = Some((offset, len));
            return Some(offset);
        }

        for larger in (len_usize + 1)..self.free_lists.len() {
            if let Some(offset) = self.free_lists[larger].pop() {
                let remaining = (larger as u32).saturating_sub(len);
                if remaining > 0 {
                    let rem_usize = remaining as usize;
                    if rem_usize < self.free_lists.len() {
                        self.free_lists[rem_usize].push(offset + len);
                    }
                }
                self.slot_ranges[slot] = Some((offset, len));
                return Some(offset);
            }
        }

        let total_units = self.units.len() as u32;
        if self.next_free_offset + len <= total_units {
            let offset = self.next_free_offset;
            self.next_free_offset += len;
            self.slot_ranges[slot] = Some((offset, len));
            return Some(offset);
        }

        None
    }
}

pub struct GpuBuffers {
    pub cell_vector_a: GpuVector<Cell>,
    pub cell_vector_b: GpuVector<Cell>,
    cell_read_buffer: AtomicBool,
    pub uniform_buffer: wgpu::Buffer,
    alive_counter: wgpu::Buffer,
    alive_counter_staging: wgpu::Buffer,
    alive_counter_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    lifeform_active_flags: wgpu::Buffer,
    lifeform_active_flags_staging: wgpu::Buffer,
    lifeform_active_flags_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    spawn_request_count: wgpu::Buffer,
    spawn_requests: wgpu::Buffer,
    division_request_count: wgpu::Buffer,
    division_requests: wgpu::Buffer,
    division_requests_staging: wgpu::Buffer,
    division_requests_readback: Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    spawn_capacity: usize,
    division_capacity: usize,
    link_buffer: wgpu::Buffer,
    link_free_list: wgpu::Buffer,
    link_free_list_count: wgpu::Buffer,
    link_capacity: usize,
    cell_event_count: wgpu::Buffer,
    cell_events: wgpu::Buffer,
    cell_event_staging: Vec<wgpu::Buffer>,
    cell_event_readback: Vec<Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>>,
    link_event_count: wgpu::Buffer,
    link_events: wgpu::Buffer,
    link_event_staging: Vec<wgpu::Buffer>,
    link_event_readback: Vec<Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>>,
    initial_alive_count: u32,
    nutrient_grid: RwLock<Arc<wgpu::Buffer>>,
    nutrient_grid_width: AtomicU32,
    nutrient_grid_height: AtomicU32,
    spatial_hash_bucket_heads: wgpu::Buffer,
    spatial_hash_next_indices: wgpu::Buffer,
    grn_descriptors: wgpu::Buffer,
    grn_units: wgpu::Buffer,
    grn_state: Mutex<GrnAllocationState>,
    grn_descriptors_cpu: Mutex<Vec<GrnDescriptor>>,
    lifeform_states: wgpu::Buffer,
    lifeform_states_cpu: Mutex<Vec<LifeformState>>,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cells: &[u8],
        initial_uniforms: &[u8],
        bounds: Rect,
    ) -> Self {
        let mut initial_cells: Vec<Cell> = bytemuck::cast_slice(cells).to_vec();
        if initial_cells.len() > CELL_CAPACITY {
            panic!(
                "Initial cell count {} exceeds CELL_CAPACITY {}",
                initial_cells.len(),
                CELL_CAPACITY
            );
        }

        for cell in initial_cells.iter_mut() {
            cell.is_alive = 1;
        }

        let initial_count = initial_cells.len() as u32;

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

        let mut lifeform_active_init = vec![0u32; LIFEFORM_CAPACITY];
        for cell in &initial_cells {
            if cell.is_alive != 0 {
                let slot = cell.lifeform_slot as usize;
                if slot < LIFEFORM_CAPACITY {
                    lifeform_active_init[slot] = 1;
                }
            }
        }

        let lifeform_active_flags = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeform Active Flags"),
            contents: bytemuck::cast_slice(&lifeform_active_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let lifeform_active_flags_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Lifeform Active Flags Staging"),
            size: (LIFEFORM_CAPACITY * std::mem::size_of::<u32>()) as u64,
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

        let division_request_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Division Request Count"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let division_requests = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Division Requests Buffer"),
            size: (MAX_DIVISION_REQUESTS * std::mem::size_of::<DivisionRequest>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let division_requests_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Division Requests Staging"),
            size: (std::mem::size_of::<u32>()
                + MAX_DIVISION_REQUESTS * std::mem::size_of::<DivisionRequest>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let link_init = vec![Link::zeroed(); LINK_CAPACITY];
        let link_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Buffer"),
            contents: bytemuck::cast_slice(&link_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        let link_free_list_init: Vec<u32> =
            (0..LINK_FREE_LIST_CAPACITY as u32).rev().collect();
        let link_free_list = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Free List"),
            contents: bytemuck::cast_slice(&link_free_list_init),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let link_free_list_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Free Count"),
            contents: bytemuck::cast_slice(&[LINK_FREE_LIST_CAPACITY as u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let cell_event_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Event Count"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let cell_events = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Events Buffer"),
            size: (MAX_CELL_EVENTS * std::mem::size_of::<CellEvent>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cell_event_staging_size = (std::mem::size_of::<u32>()
            + MAX_CELL_EVENTS * std::mem::size_of::<CellEvent>()) as u64;
        let mut cell_event_staging = Vec::with_capacity(EVENT_STAGING_RING_SIZE);
        let mut cell_event_readback: Vec<
            Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
        > = Vec::with_capacity(EVENT_STAGING_RING_SIZE);
        for _ in 0..EVENT_STAGING_RING_SIZE {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cell Events Staging"),
                size: cell_event_staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            cell_event_staging.push(staging);
            cell_event_readback.push(Mutex::new(None));
        }

        let link_event_count = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Link Event Count"),
            contents: bytemuck::cast_slice(&[0u32]),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let link_events = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Link Events Buffer"),
            size: (MAX_LINK_EVENTS * std::mem::size_of::<LinkEvent>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let link_event_staging_size = (std::mem::size_of::<u32>()
            + MAX_LINK_EVENTS * std::mem::size_of::<LinkEvent>()) as u64;
        let mut link_event_staging = Vec::with_capacity(EVENT_STAGING_RING_SIZE);
        let mut link_event_readback: Vec<
            Mutex<Option<mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>>,
        > = Vec::with_capacity(EVENT_STAGING_RING_SIZE);
        for _ in 0..EVENT_STAGING_RING_SIZE {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Link Events Staging"),
                size: link_event_staging_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            link_event_staging.push(staging);
            link_event_readback.push(Mutex::new(None));
        }

        let spatial_hash_init = vec![-1i32; CELL_HASH_TABLE_SIZE];
        let spatial_hash_bucket_heads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cell Spatial Hash Bucket Heads"),
            contents: bytemuck::cast_slice(&spatial_hash_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        let grn_units_cpu = vec![CompiledRegulatoryUnit::zeroed(); MAX_TOTAL_GRN_UNITS];
        let grn_units = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GRN Units Buffer"),
            contents: bytemuck::cast_slice(&grn_units_cpu),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let lifeform_states_cpu = vec![LifeformState::inactive(); LIFEFORM_CAPACITY];
        let lifeform_states = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Lifeform States Buffer"),
            contents: bytemuck::cast_slice(&lifeform_states_cpu),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let grn_state = Mutex::new(GrnAllocationState::new(grn_units_cpu));

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
            alive_counter,
            alive_counter_staging,
            alive_counter_readback: Mutex::new(None),
            lifeform_active_flags,
            lifeform_active_flags_staging,
            lifeform_active_flags_readback: Mutex::new(None),
            spawn_request_count,
            spawn_requests,
            division_request_count,
            division_requests,
            division_requests_staging,
            division_requests_readback: Mutex::new(None),
            spawn_capacity: MAX_SPAWN_REQUESTS,
            division_capacity: MAX_DIVISION_REQUESTS,
            link_buffer,
            link_free_list,
            link_free_list_count,
            link_capacity: LINK_CAPACITY,
            cell_event_count,
            cell_events,
            cell_event_staging,
            cell_event_readback,
            link_event_count,
            link_events,
            link_event_staging,
            link_event_readback,
            initial_alive_count: initial_count,
            nutrient_grid: RwLock::new(Arc::clone(&nutrient_grid)),
            nutrient_grid_width: AtomicU32::new(grid_width),
            nutrient_grid_height: AtomicU32::new(grid_height),
            spatial_hash_bucket_heads,
            spatial_hash_next_indices,
            grn_descriptors,
            grn_units,
            grn_state,
            grn_descriptors_cpu: Mutex::new(vec![GrnDescriptor::zeroed(); LIFEFORM_CAPACITY]),
            lifeform_states,
            lifeform_states_cpu: Mutex::new(lifeform_states_cpu),
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

    pub fn try_consume_alive_counter(&self) -> Option<u32> {
        let mut receiver_guard = self.alive_counter_readback.lock();
        let receiver = match receiver_guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

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

    pub fn lifeform_active_flags_buffer(&self) -> &wgpu::Buffer {
        &self.lifeform_active_flags
    }

    pub fn schedule_lifeform_flags_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.lifeform_active_flags_readback.lock();
            if guard.take().is_some() {
                self.lifeform_active_flags_staging.unmap();
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.lifeform_active_flags,
            0,
            &self.lifeform_active_flags_staging,
            0,
            (LIFEFORM_CAPACITY * std::mem::size_of::<u32>()) as u64,
        );
    }

    pub fn begin_lifeform_flags_map(&self) {
        let mut guard = self.lifeform_active_flags_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.lifeform_active_flags_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_lifeform_flags(&self) -> Option<Vec<u32>> {
        let mut guard = self.lifeform_active_flags_readback.lock();
        let receiver = match guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.lifeform_active_flags_staging.slice(..).get_mapped_range();
                let values = bytemuck::cast_slice::<u8, u32>(&mapped);
                let mut result = Vec::with_capacity(values.len());
                result.extend_from_slice(values);
                drop(mapped);
                self.lifeform_active_flags_staging.unmap();
                *guard = None;
                Some(result)
            }
            Ok(Err(e)) => {
                eprintln!("Lifeform flags read failed: {:?}", e);
                self.lifeform_active_flags_staging.unmap();
                *guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Lifeform flags readback channel disconnected");
                *guard = None;
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

    pub fn division_request_count_buffer(&self) -> &wgpu::Buffer {
        &self.division_request_count
    }

    pub fn division_requests_buffer(&self) -> &wgpu::Buffer {
        &self.division_requests
    }

    pub fn spawn_capacity(&self) -> usize {
        self.spawn_capacity
    }

    pub fn lifeform_capacity(&self) -> usize {
        LIFEFORM_CAPACITY
    }
    
    pub fn link_buffer(&self) -> &wgpu::Buffer {
        &self.link_buffer
    }

    pub fn link_free_list_buffer(&self) -> &wgpu::Buffer {
        &self.link_free_list
    }

    pub fn link_free_count_buffer(&self) -> &wgpu::Buffer {
        &self.link_free_list_count
    }

    pub fn link_capacity(&self) -> usize {
        self.link_capacity
    }

    pub fn link_event_count_buffer(&self) -> &wgpu::Buffer {
        &self.link_event_count
    }

    pub fn link_events_buffer(&self) -> &wgpu::Buffer {
        &self.link_events
    }

    pub fn cell_event_count_buffer(&self) -> &wgpu::Buffer {
        &self.cell_event_count
    }

    pub fn cell_events_buffer(&self) -> &wgpu::Buffer {
        &self.cell_events
    }

    pub fn cell_event_capacity(&self) -> usize {
        MAX_CELL_EVENTS
    }

    pub fn link_event_capacity(&self) -> usize {
        MAX_LINK_EVENTS
    }

    pub fn event_staging_ring_size(&self) -> usize {
        EVENT_STAGING_RING_SIZE
    }
    
    pub fn initial_alive_count(&self) -> u32 {
        self.initial_alive_count
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

    pub fn schedule_division_requests_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut guard = self.division_requests_readback.lock();
            if guard.take().is_some() {
                self.division_requests_staging.unmap();
            }
        }

        let count_size = std::mem::size_of::<u32>() as u64;
        let requests_size =
            (self.division_capacity * std::mem::size_of::<DivisionRequest>()) as u64;

        encoder.copy_buffer_to_buffer(
            &self.division_request_count,
            0,
            &self.division_requests_staging,
            0,
            count_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.division_requests,
            0,
            &self.division_requests_staging,
            count_size,
            requests_size,
        );
        encoder.clear_buffer(&self.division_request_count, 0, None);
    }

    pub fn begin_division_requests_map(&self) {
        let mut guard = self.division_requests_readback.lock();
        if guard.is_some() {
            return;
        }
        let slice = self.division_requests_staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_division_requests(&self) -> Option<Vec<DivisionRequest>> {
        let mut guard = self.division_requests_readback.lock();
        let receiver = match guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapped = self.division_requests_staging.slice(..).get_mapped_range();
                let count_size = std::mem::size_of::<u32>();
                if mapped.len() < count_size {
                    self.division_requests_staging.unmap();
                    *guard = None;
                    return Some(Vec::new());
                }
                let mut count_bytes = [0u8; 4];
                count_bytes.copy_from_slice(&mapped[..count_size]);
                let count = u32::from_le_bytes(count_bytes);
                let capped_count = count.min(self.division_capacity as u32);
                let total_bytes =
                    capped_count as usize * std::mem::size_of::<DivisionRequest>();
                let mut result = Vec::with_capacity(capped_count as usize);
                if capped_count > 0 {
                    let requests_bytes =
                        &mapped[count_size..count_size + total_bytes];
                    let typed =
                        bytemuck::cast_slice::<u8, DivisionRequest>(requests_bytes);
                    result.extend_from_slice(typed);
                }
                drop(mapped);
                self.division_requests_staging.unmap();
                *guard = None;
                Some(result)
            }
            Ok(Err(e)) => {
                eprintln!("Division requests read failed: {:?}", e);
                self.division_requests_staging.unmap();
                *guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Division requests readback channel disconnected");
                *guard = None;
                None
            }
        }
    }

    pub fn schedule_cell_events_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging_index: usize,
    ) {
        assert!(
            staging_index < self.cell_event_staging.len(),
            "staging_index out of range"
        );
        {
            let mut guard = self.cell_event_readback[staging_index].lock();
            if guard.take().is_some() {
                self.cell_event_staging[staging_index].unmap();
            }
        }
        let count_size = std::mem::size_of::<u32>() as u64;
        let events_size =
            (MAX_CELL_EVENTS * std::mem::size_of::<CellEvent>()) as u64;

        encoder.copy_buffer_to_buffer(
            &self.cell_event_count,
            0,
            &self.cell_event_staging[staging_index],
            0,
            count_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.cell_events,
            0,
            &self.cell_event_staging[staging_index],
            count_size,
            events_size,
        );
        encoder.clear_buffer(&self.cell_event_count, 0, None);
    }

    pub fn begin_cell_events_map(&self, staging_index: usize) {
        assert!(
            staging_index < self.cell_event_staging.len(),
            "staging_index out of range"
        );
        let mut guard = self.cell_event_readback[staging_index].lock();
        if guard.is_some() {
            return;
        }
        let slice = self.cell_event_staging[staging_index].slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_cell_events(
        &self,
        staging_index: usize,
    ) -> Option<Vec<CellEvent>> {
        assert!(
            staging_index < self.cell_event_staging.len(),
            "staging_index out of range"
        );
        let mut guard = self.cell_event_readback[staging_index].lock();
        let receiver = match guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapping = self.cell_event_staging[staging_index]
                    .slice(..)
                    .get_mapped_range();
                let count_size = std::mem::size_of::<u32>();
                if mapping.len() < count_size {
                    drop(mapping);
                    self.cell_event_staging[staging_index].unmap();
                    *guard = None;
                    return Some(Vec::new());
                }
                let mut count_bytes = [0u8; 4];
                count_bytes.copy_from_slice(&mapping[..count_size]);
                let count = u32::from_le_bytes(count_bytes);
                let capped_count = count.min(MAX_CELL_EVENTS as u32);
                let events_bytes_len =
                    capped_count as usize * std::mem::size_of::<CellEvent>();
                if mapping.len() < count_size + events_bytes_len {
                    eprintln!("Cell events buffer smaller than expected");
                    self.cell_event_staging[staging_index].unmap();
                    *guard = None;
                    return None;
                }
                let mut result = Vec::with_capacity(capped_count as usize);
                let stride = std::mem::size_of::<CellEvent>();
                for idx in 0..capped_count as usize {
                    let start = count_size + idx * stride;
                    let end = start + stride;
                    let bytes = &mapping[start..end];
                    let event = pod_read_unaligned::<CellEvent>(bytes);
                    result.push(event);
                }
                drop(mapping);
                self.cell_event_staging[staging_index].unmap();
                *guard = None;
                Some(result)
            }
            Ok(Err(e)) => {
                eprintln!("Cell events read failed: {:?}", e);
                self.cell_event_staging[staging_index].unmap();
                *guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Cell events readback channel disconnected");
                *guard = None;
                None
            }
        }
    }

    pub fn schedule_link_events_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        staging_index: usize,
    ) {
        assert!(
            staging_index < self.link_event_staging.len(),
            "staging_index out of range"
        );
        {
            let mut guard = self.link_event_readback[staging_index].lock();
            if guard.take().is_some() {
                self.link_event_staging[staging_index].unmap();
            }
        }
        let count_size = std::mem::size_of::<u32>() as u64;
        let events_size =
            (MAX_LINK_EVENTS * std::mem::size_of::<LinkEvent>()) as u64;

        encoder.copy_buffer_to_buffer(
            &self.link_event_count,
            0,
            &self.link_event_staging[staging_index],
            0,
            count_size,
        );
        encoder.copy_buffer_to_buffer(
            &self.link_events,
            0,
            &self.link_event_staging[staging_index],
            count_size,
            events_size,
        );
        encoder.clear_buffer(&self.link_event_count, 0, None);
    }

    pub fn begin_link_events_map(&self, staging_index: usize) {
        assert!(
            staging_index < self.link_event_staging.len(),
            "staging_index out of range"
        );
        let mut guard = self.link_event_readback[staging_index].lock();
        if guard.is_some() {
            return;
        }
        let slice = self.link_event_staging[staging_index].slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        *guard = Some(receiver);
    }

    pub fn try_consume_link_events(
        &self,
        staging_index: usize,
    ) -> Option<Vec<LinkEvent>> {
        assert!(
            staging_index < self.link_event_staging.len(),
            "staging_index out of range"
        );
        let mut guard = self.link_event_readback[staging_index].lock();
        let receiver = match guard.as_ref() {
            Some(r) => r,
            None => return None,
        };

        match receiver.try_recv() {
            Ok(Ok(_)) => {
                let mapping = self.link_event_staging[staging_index]
                    .slice(..)
                    .get_mapped_range();
                let count_size = std::mem::size_of::<u32>();
                if mapping.len() < count_size {
                    drop(mapping);
                    self.link_event_staging[staging_index].unmap();
                    *guard = None;
                    return Some(Vec::new());
                }
                let mut count_bytes = [0u8; 4];
                count_bytes.copy_from_slice(&mapping[..count_size]);
                let count = u32::from_le_bytes(count_bytes);
                let capped_count = count.min(MAX_LINK_EVENTS as u32);
                let events_bytes_len =
                    capped_count as usize * std::mem::size_of::<LinkEvent>();
                if mapping.len() < count_size + events_bytes_len {
                    eprintln!("Link events buffer smaller than expected");
                    self.link_event_staging[staging_index].unmap();
                    *guard = None;
                    return None;
                }
                let mut result = Vec::with_capacity(capped_count as usize);
                let stride = std::mem::size_of::<LinkEvent>();
                for idx in 0..capped_count as usize {
                    let start = count_size + idx * stride;
                    let end = start + stride;
                    let bytes = &mapping[start..end];
                    let event = pod_read_unaligned::<LinkEvent>(bytes);
                    result.push(event);
                }
                drop(mapping);
                self.link_event_staging[staging_index].unmap();
                *guard = None;
                Some(result)
            }
            Ok(Err(e)) => {
                eprintln!("Link events read failed: {:?}", e);
                self.link_event_staging[staging_index].unmap();
                *guard = None;
                None
            }
            Err(mpsc::TryRecvError::Empty) => None,
            Err(mpsc::TryRecvError::Disconnected) => {
                eprintln!("Link events readback channel disconnected");
                *guard = None;
                None
            }
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

    pub fn cell_hash_next_indices_buffer(&self) -> &wgpu::Buffer {
        &self.spatial_hash_next_indices
    }

    pub fn cell_hash_table_size(&self) -> usize {
        CELL_HASH_TABLE_SIZE
    }

    pub fn write_links(&self, queue: &wgpu::Queue, offset: usize, links: &[Link]) {
        let end = offset
            .checked_add(links.len())
            .expect("link write range overflow");
        assert!(
            end <= self.link_capacity,
            "link write exceeds capacity ({} > {})",
            end,
            self.link_capacity
        );
        let byte_offset = (offset * std::mem::size_of::<Link>()) as u64;
        queue.write_buffer(
            &self.link_buffer,
            byte_offset,
            bytemuck::cast_slice(links),
        );
    }

    pub fn grn_descriptor_buffer(&self) -> &wgpu::Buffer {
        &self.grn_descriptors
    }

    pub fn grn_units_buffer(&self) -> &wgpu::Buffer {
        &self.grn_units
    }

    pub fn lifeform_states_buffer(&self) -> &wgpu::Buffer {
        &self.lifeform_states
    }

    pub fn grn_descriptor_cpu(&self, slot: u32) -> GrnDescriptor {
        let descriptors = self.grn_descriptors_cpu.lock();
        descriptors
            .get(slot as usize)
            .copied()
            .unwrap_or_else(GrnDescriptor::zeroed)
    }

    pub fn write_lifeform_state(&self, queue: &wgpu::Queue, slot: u32, state: LifeformState) {
        let slot_idx = slot as usize;
        if slot_idx >= LIFEFORM_CAPACITY {
            return;
        }

        {
            let mut cpu = self.lifeform_states_cpu.lock();
            if let Some(entry) = cpu.get_mut(slot_idx) {
                *entry = state;
            }
        }

        let offset = (slot_idx * std::mem::size_of::<LifeformState>()) as u64;
        queue.write_buffer(
            &self.lifeform_states,
            offset,
            bytemuck::cast_slice(std::slice::from_ref(&state)),
        );
    }

    pub fn clear_lifeform_state(&self, queue: &wgpu::Queue, slot: u32) {
        self.write_lifeform_state(queue, slot, LifeformState::inactive());
    }

    pub fn write_grn_slot(&self, queue: &wgpu::Queue, slot: u32, compiled: &CompiledGrn) -> GrnDescriptor {
        profile_scope!("Write GRN Slot");
        let slot_idx = slot as usize;
        if slot_idx >= LIFEFORM_CAPACITY {
            return GrnDescriptor::zeroed();
        }

        let mut descriptor = compiled.descriptor;
        let mut state = {
            profile_scope!("Acquire GRN Allocator");
            self.grn_state.lock()
        };
        descriptor.unit_count = compiled.units.len() as u32;
        if descriptor.unit_count == 0 {
            descriptor.unit_offset = 0;
            {
                profile_scope!("Release Empty Slot");
                state.release_slot(slot_idx);
            }
            drop(state);
            self.write_grn_descriptor(queue, slot_idx, &descriptor);
            return descriptor;
        }

        if descriptor.unit_count as usize > MAX_TOTAL_GRN_UNITS {
            drop(state);
            eprintln!(
                "GRN allocation exceeds global capacity (slot {}, units {}).",
                slot_idx, descriptor.unit_count
            );
            self.write_grn_descriptor(queue, slot_idx, &GrnDescriptor::zeroed());
            return GrnDescriptor::zeroed();
        }

        match {
            profile_scope!("Allocate GRN Range");
            state.allocate_slot(slot_idx, descriptor.unit_count)
        } {
            Some(offset) => {
                let start = offset as usize;
                let end = start + descriptor.unit_count as usize;
                if end > state.units.len() {
                    drop(state);
                    eprintln!(
                        "GRN allocation out of bounds for slot {} (offset {}, count {}).",
                        slot_idx, offset, descriptor.unit_count
                    );
                    self.write_grn_descriptor(queue, slot_idx, &GrnDescriptor::zeroed());
                    return GrnDescriptor::zeroed();
                }
                state.units[start..end].copy_from_slice(&compiled.units);
                descriptor.unit_offset = offset;

                let unit_byte_offset =
                    (start * std::mem::size_of::<CompiledRegulatoryUnit>()) as u64;
                let bytes = bytemuck::cast_slice(&compiled.units);
                drop(state);

                self.write_grn_descriptor(queue, slot_idx, &descriptor);
                {
                    profile_scope!("Upload GRN Units");
                    queue.write_buffer(&self.grn_units, unit_byte_offset, bytes);
                }
                descriptor
            }
            None => {
                drop(state);
                eprintln!(
                    "Unable to allocate GRN storage for slot {} (requested {}).",
                    slot_idx, descriptor.unit_count
                );
                self.write_grn_descriptor(queue, slot_idx, &GrnDescriptor::zeroed());
                GrnDescriptor::zeroed()
            }
        }
    }

    pub fn clear_grn_slot(&self, queue: &wgpu::Queue, slot: u32) {
        profile_scope!("Clear GRN Slot");
        let slot_idx = slot as usize;
        if slot_idx >= LIFEFORM_CAPACITY {
            return;
        }
        {
            let mut state = self.grn_state.lock();
            state.release_slot(slot_idx);
        }
        self.write_grn_descriptor(queue, slot_idx, &GrnDescriptor::zeroed());
    }

    fn write_grn_descriptor(&self, queue: &wgpu::Queue, slot_idx: usize, descriptor: &GrnDescriptor) {
        profile_scope!("Upload GRN Descriptor");
        let descriptor_offset = (slot_idx * std::mem::size_of::<GrnDescriptor>()) as u64;
        queue.write_buffer(
            &self.grn_descriptors,
            descriptor_offset,
            bytemuck::cast_slice(std::slice::from_ref(descriptor)),
        );
        {
            let mut cpu = self.grn_descriptors_cpu.lock();
            if let Some(entry) = cpu.get_mut(slot_idx) {
                *entry = *descriptor;
            }
        }
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

}