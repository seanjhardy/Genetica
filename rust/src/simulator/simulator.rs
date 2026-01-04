use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::atomic::AtomicUsize;
use crate::utils::math::{Rect};
use puffin::profile_scope;
use crate::gpu::buffers::{CELL_CAPACITY, GpuBuffers, POINT_CAPACITY};
use crate::gpu::pipelines::{ComputePipelines};
use crate::gpu::uniforms::Uniforms;
use crate::simulator::environment::Environment;
use crate::simulator::state::{PauseState, SimSlot, SlotState};
use crate::genetic_algorithm::GeneticAlgorithm;

const WORKGROUP_SIZE: u32 = 512;
const SIM_STATE_RING_SIZE: usize = 1;
const CELL_UPDATE_INTERVAL: usize = 10;

pub struct Simulation {
    pub render_slot: usize,
    pub slots: Vec<SimSlot>,
    step: Arc<AtomicUsize>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    environment: Arc<parking_lot::Mutex<Environment>>,
    pub genetic_algorithm: GeneticAlgorithm,
    current_bounds: Rect,
    initial_bounds: Rect,
    paused_state: PauseState,
}

impl Simulation {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        environment: Arc<parking_lot::Mutex<Environment>>,
        initial_uniforms: Uniforms,
    ) -> Self {
        let initial_bounds = environment.lock().get_bounds();
        let initial_uniform_bytes = bytemuck::bytes_of(&initial_uniforms).to_vec();

        let mut slots = Vec::with_capacity(SIM_STATE_RING_SIZE);
        for i in 0..SIM_STATE_RING_SIZE {
            let buffers = Arc::new(GpuBuffers::new(
                &device,
                &queue,
                &[],
                &initial_uniform_bytes,
                initial_bounds,
            ));
            buffers.update_uniforms(&queue, &initial_uniform_bytes);

            let compute_pipelines = ComputePipelines::new(
                &device,
                &buffers,
            );

            let state = if i == 0 {
                SlotState::Completed
            } else {
                SlotState::Free
            };

            slots.push(SimSlot {
                buffers,
                compute_pipelines,
                completion: None,
                step_id: 0,
                state,
            });
        }
        
        Self {
            device,
            queue,
            environment,
            slots,
            step: Arc::new(AtomicUsize::new(0)),
            genetic_algorithm: GeneticAlgorithm::new(),
            current_bounds: initial_bounds,
            initial_bounds,
            render_slot: 0,
            paused_state: PauseState::new(false),
        }
    }

    pub fn get_render_buffers(&self) -> Arc<GpuBuffers> {
        self.slots[self.render_slot].buffers.clone()
    }

    pub fn is_paused(&self) -> bool {
        self.paused_state.is_paused()
    }

    pub fn set_paused(&mut self, paused: bool) {
        self.paused_state.set(paused);
    }

    pub fn step_simulation(&mut self, encoder: &mut wgpu::CommandEncoder) {
        profile_scope!("Simulation Step");
        self.step.fetch_add(1, Ordering::Relaxed);

        let slot = &mut self.slots[self.render_slot];
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Simulation Pass"),
            timestamp_writes: None,
        });

        // Update points
        pass.set_pipeline(&slot.compute_pipelines.update_points);
        pass.set_bind_group(0, &slot.compute_pipelines.update_points_bind_group, &[]);
        let dispatch = ((POINT_CAPACITY as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        pass.dispatch_workgroups(dispatch, 1, 1);

        if self.step.load(Ordering::Relaxed) % CELL_UPDATE_INTERVAL == 0 {
            pass.set_pipeline(&slot.compute_pipelines.update_cells);
            pass.set_bind_group(0, &slot.compute_pipelines.update_cells_bind_group, &[]);
            let dispatch = ((CELL_CAPACITY as u32) + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch, 1, 1);
        }

        // Spawn new points each simulation step
        pass.set_pipeline(&slot.compute_pipelines.spawn_cells);
        pass.set_bind_group(0, &slot.compute_pipelines.spawn_cells_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1); // Single thread execution

        // Rotate buffers: scratch becomes current, current becomes previous, previous becomes scratch.
        //self.slots.rotate_left(1);

        // Process pending events
        //self.genetic_algorithm.process_events(self.step.load(Ordering::Relaxed), slot.buffers.event_buffer);
    }

    pub fn reset(&mut self) {
        let _ = self.device.poll(wgpu::MaintainBase::Wait);

        for slot in self.slots.iter_mut() {
            slot.buffers
                .reset(&self.device, &self.queue, self.initial_bounds);
            slot.step_id = 0;
            slot.state = SlotState::Free;
            slot.completion = None;
        }

        if let Some(render) = self.slots.first_mut() {
            render.state = SlotState::Completed;
        }

        {
            let mut env = self.environment.lock();
            env.set_bounds(self.initial_bounds);
        }

        self.step.store(0, Ordering::Relaxed);
        self.current_bounds = self.initial_bounds;
    }

    pub fn get_step(&self) -> usize {
        self.step.load(Ordering::Relaxed)
    }

}
