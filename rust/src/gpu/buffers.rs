// GPU buffers module - manages GPU buffer creation and updates

use wgpu;
use wgpu::util::DeviceExt;

/// GPU buffers for points, uniforms, and timestamps
pub struct GpuBuffers {
    pub point_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        points: &[u8],
        initial_uniforms: &[u8],
    ) -> Self {
        let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Buffer"),
            contents: points,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: initial_uniforms,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            point_buffer,
            uniform_buffer,
        }
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &[u8]) {
        queue.write_buffer(&self.uniform_buffer, 0, uniforms);
    }
}

/// Timestamp query buffers for GPU profiling
pub struct TimestampBuffers {
    pub compute_timestamp_set: Option<wgpu::QuerySet>,
    pub collision_timestamp_set: Option<wgpu::QuerySet>,
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

            let collision_ts = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Collision Timestamp Query Set"),
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
            let buffer_size = query_align * 3; // 3 query sets, each aligned to 256 bytes

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
                collision_timestamp_set: Some(collision_ts),
                render_timestamp_set: Some(render_ts),
                timestamp_buffer: Some(ts_buffer),
            }
        } else {
            Self {
                compute_timestamp_set: None,
                collision_timestamp_set: None,
                render_timestamp_set: None,
                timestamp_buffer: None,
            }
        }
    }
}

