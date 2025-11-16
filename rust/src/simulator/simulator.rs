// Simulator module - main simulation loop and window management

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorIcon},
};

use crate::utils::math::{Rect, Vec2};
use crate::utils::camera::{Camera, KeyStates};
use crate::ui::{UiParser, UiRenderer, UIManager};
use crate::utils::strings::format_number;
use puffin::profile_scope;
use crate::utils::gpu::device::GpuDevice;
use crate::gpu::buffers::GpuBuffers;
use crate::gpu::pipelines::{ComputePipelines, RenderPipelines};
use crate::gpu::uniforms::Uniforms;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::gpu::structures::Cell;
use crate::simulator::environment::Environment;
use crate::simulator::state::{GpuTransferState, PauseState, PopulationState, SubmissionState};
use crate::simulator::renderer::Renderer;

const SIMULATION_DELTA_TIME: f32 = 0.1; // 100ms per simulation step
/// Simulation structure
pub struct Simulation {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    compute_pipelines: ComputePipelines,
    buffers: Arc<GpuBuffers>,
    environment: Arc<parking_lot::Mutex<Environment>>,
    pause: PauseState,
    speed: Arc<parking_lot::Mutex<f32>>, // Shared speed for thread-safe access
    population: PopulationState,
    transfers: GpuTransferState,
    current_bounds: Rect,
    
    // Simulation parameters
    workgroup_size: u32,
    
    // Step counter (atomic for thread-safe access)
    step_count: Arc<AtomicU64>,
    submission: SubmissionState,
}

impl Simulation {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        compute_pipelines: ComputePipelines,
        buffers: Arc<GpuBuffers>,
        environment: Arc<parking_lot::Mutex<Environment>>,
        speed: Arc<parking_lot::Mutex<f32>>,
    ) -> Self {
        let initial_bounds = environment.lock().get_bounds();
        let simulation = Self {
            device,
            queue,
            compute_pipelines,
            buffers,
            environment,
            pause: PauseState::new(false),
            speed,
            population: PopulationState::new(),
            transfers: GpuTransferState::default(),
            current_bounds: initial_bounds,
            workgroup_size: 128,
            step_count: Arc::new(AtomicU64::new(0)),
            submission: SubmissionState::new(32, Duration::from_millis(10)),
        };

        simulation
    }
    
    pub fn set_paused(&mut self, paused: bool) {
        self.pause.set(paused);
    }
    
    // Run a single simulation step (called as fast as possible)
    pub fn step(&mut self) {
        profile_scope!("Simulation Step");
        
        // Skip simulation if paused
        if self.pause.is_paused() {
            thread::sleep(Duration::from_millis(10));
            return;
        }
        
        // Control simulation rate based on speed
        // Base rate: 100 microseconds sleep = ~10,000 steps/sec
        // At speed 1.0, we sleep 100us per step
        // At speed 0.5, we sleep 200us per step (half speed)
        // At speed 2.0, we sleep 50us per step (double speed)
        let speed = {
            profile_scope!("Simulation Rate Control");
            let speed = (*self.speed.lock()).max(0.0);
            if speed > 0.0 && speed < 1.0 {
                let base_sleep_micros = 100.0;
                let adjusted_sleep = (base_sleep_micros / speed.max(0.01)) as u64;
                if adjusted_sleep > 0 {
                    thread::sleep(Duration::from_micros(adjusted_sleep));
                }
            } else if speed == 0.0 {
                thread::sleep(Duration::from_millis(5));
            }
            speed
        };

        let iterations = (10.0 * speed).max(1.0).floor() as u32;
        self.run_compute_batch(iterations);
        // Poll GPU readbacks less frequently - only every 10 steps to reduce overhead
        if self.step_count.load(Ordering::Relaxed) % 10 == 0 {
            self.poll_gpu_readbacks();
        }
    }
    
    pub fn get_step_count(&self) -> u64 {
        self.step_count.load(Ordering::Relaxed)
    }
    
    pub fn get_buffers(&self) -> Arc<GpuBuffers> {
        self.buffers.clone()
    }

    pub fn paused_handle(&self) -> Arc<AtomicBool> {
        self.pause.handle()
    }

    pub fn is_paused(&self) -> bool {
        self.pause.is_paused()
    }

    fn flush_pending_submissions(&mut self) {
        profile_scope!("Flush Pending Submissions");
        if self.submission.pending_command_buffers.is_empty() {
            return;
        }

        {
            profile_scope!("wgpu::Queue::submit");
            self.queue
                .submit(self.submission.pending_command_buffers.drain(..));
        }
        self.submission.record_submission_time();
    }

    fn run_compute_batch(&mut self, iterations: u32) {
        if iterations == 0 {
            return;
        }

        // Submit simulation compute pass (non-blocking - GPU operations are async)
        let command_buffer = {
            profile_scope!("Encode Cell Simulation");
            let mut encoder = {
                profile_scope!("Create Command Encoder");
                self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Simulation Encoder"),
                })
            };

            {
                profile_scope!("Dispatch Nutrient Regeneration");
                let mut nutrient_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Nutrient Regeneration Pass"),
                    timestamp_writes: None,
                });
                let (grid_w, grid_h) = self.buffers.nutrient_grid_dimensions();
                let total_cells = grid_w.saturating_mul(grid_h);
                if total_cells > 0 {
                    let workgroup_size: u32 = 256;
                    let workgroups = (total_cells + workgroup_size - 1) / workgroup_size;
                    nutrient_pass.set_pipeline(&self.compute_pipelines.update_nutrients);
                    nutrient_pass.set_bind_group(0, &self.compute_pipelines.update_nutrients_bind_group, &[]);
                    for _ in 0..iterations {
                        nutrient_pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
            }

            {
                profile_scope!("Dispatch Cell Simulation Compute Batch");
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Compute Pass"),
                    timestamp_writes: None,
                });

                let num_cells = self.buffers.cell_capacity() as u32;
                let cell_workgroups = (num_cells + self.workgroup_size - 1) / self.workgroup_size;
                let hash_table_size = self.buffers.cell_hash_table_size() as u32;
                let hash_workgroups =
                    (hash_table_size + self.workgroup_size - 1) / self.workgroup_size;
                let link_capacity = self.buffers.link_capacity() as u32;
                let link_workgroups =
                    (link_capacity + self.workgroup_size - 1) / self.workgroup_size;

                compute_pass.set_bind_group(
                    0,
                    &self.compute_pipelines.update_cells_bind_group,
                    &[],
                );
                for _ in 0..iterations {
                    compute_pass.set_pipeline(&self.compute_pipelines.reset_cell_hash);
                    compute_pass.dispatch_workgroups(hash_workgroups, 1, 1);

                    if link_capacity > 0 {
                        compute_pass.set_pipeline(&self.compute_pipelines.update_links);
                        compute_pass.dispatch_workgroups(link_workgroups, 1, 1);
                    }

                    if cell_workgroups > 0 {
                        compute_pass.set_pipeline(&self.compute_pipelines.build_cell_hash);
                        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);

                        compute_pass.set_pipeline(&self.compute_pipelines.update_cells);
                        compute_pass.dispatch_workgroups(cell_workgroups, 1, 1);
                    }
                }
            }

            // Try to consume any ready readbacks before scheduling new copies
            // This frees up staging buffer slots and prevents "buffer still mapped" errors
            {
                profile_scope!("Consume Ready Readbacks");
                // Poll device to check for ready readbacks (non-blocking)
                let _ = self.device.poll(wgpu::MaintainBase::Poll);
                
                // Try to consume counters if ready
                if self.transfers.cell_counter_pending {
                    if let Some(value) = self.buffers.try_consume_cell_counter() {
                        self.population.cells = value;
                        self.transfers.cell_counter_pending = false;
                    }
                }
                if self.transfers.lifeform_counter_pending {
                    if let Some(value) = self.buffers.try_consume_lifeform_counter() {
                        self.population.lifeforms = value;
                        self.transfers.lifeform_counter_pending = false;
                    }
                }
                if self.transfers.species_counter_pending {
                    if let Some(value) = self.buffers.try_consume_species_counter() {
                        self.population.species = value;
                        self.transfers.species_counter_pending = false;
                    }
                }
            }

            // Schedule counter copies
            if !self.transfers.cell_counter_pending {
                self.buffers
                    .schedule_cell_counter_copy(&mut encoder);
                self.transfers.cell_counter_pending = true;
            }
            if !self.transfers.lifeform_counter_pending {
                self.buffers
                    .schedule_lifeform_counter_copy(&mut encoder);
                self.transfers.lifeform_counter_pending = true;
            }
            if !self.transfers.species_counter_pending {
                self.buffers
                    .schedule_species_counter_copy(&mut encoder);
                self.transfers.species_counter_pending = true;
            }

            profile_scope!("Finish Command Encoder");
            encoder.finish()
        };

        {
            profile_scope!("Submit Simulation Commands");
            {
                profile_scope!("Queue Command Buffer");
                self.submission.pending_command_buffers.push(command_buffer);
            }
            if self.submission.pending_command_buffers.len() >= self.submission.submission_batch_size
                || self.submission.last_submission.elapsed() >= self.submission.max_submission_delay
            {
                self.flush_pending_submissions();
            }
        }

        // Increment step counter
        self.step_count
            .fetch_add(iterations as u64, Ordering::Relaxed);

    }

    /// Poll GPU for readbacks (counters) - renamed from maintain_minimum_cell_population
    /// Only flushes submissions when we actually need to read back data
    fn poll_gpu_readbacks(&mut self) {
        // Only flush if we have pending readbacks that need GPU completion
        let needs_flush = self.transfers.cell_counter_pending
            || self.transfers.lifeform_counter_pending
            || self.transfers.species_counter_pending;
        
        if needs_flush && !self.submission.pending_command_buffers.is_empty() {
            profile_scope!("Flush Before Readback");
            let flush_start = std::time::Instant::now();
            self.flush_pending_submissions();
            let flush_duration = flush_start.elapsed();
            
            // Log if flush takes significant time (indicates GPU is busy)
            if flush_duration.as_millis() > 1 {
                eprintln!("[GPU Profiling] Flush took {}ms (GPU may be busy)", flush_duration.as_millis());
            }
        }

        let mut counters_need_poll = false;

        if self.transfers.cell_counter_pending {
            {
                profile_scope!("Begin Alive Counter Map");
                self.buffers.begin_cell_counter_map();
            }
            counters_need_poll = true;
        }
        if self.transfers.lifeform_counter_pending {
            {
                profile_scope!("Begin Lifeform Counter Map");
                self.buffers.begin_lifeform_counter_map();
            }
            counters_need_poll = true;
        }
        if self.transfers.species_counter_pending {
            {
                profile_scope!("Begin Species Counter Map");
                self.buffers.begin_species_counter_map();
            }
            counters_need_poll = true;
        }

        if counters_need_poll {
            profile_scope!("Poll GPU Counters");
            let poll_start = std::time::Instant::now();
            let _ = self.device.poll(wgpu::MaintainBase::Poll);
            let poll_duration = poll_start.elapsed();
            
            // Log if polling takes significant time
            if poll_duration.as_millis() > 1 {
                eprintln!("[GPU Profiling] Poll took {}ms", poll_duration.as_millis());
            }
        }

        if self.transfers.cell_counter_pending {
            profile_scope!("Consume Alive Counter");
            let readback_start = std::time::Instant::now();
            if let Some(value) = self.buffers.try_consume_cell_counter() {
                let readback_duration = readback_start.elapsed();
                if readback_duration.as_millis() > 1 {
                    eprintln!("[GPU Profiling] Alive counter readback took {}ms", readback_duration.as_millis());
                }
                self.population.cells = value;
                self.transfers.cell_counter_pending = false;
            }
        }

        if self.transfers.lifeform_counter_pending {
            profile_scope!("Consume Lifeform Counter");
            if let Some(value) = self.buffers.try_consume_lifeform_counter() {
                self.population.lifeforms = value;
                self.transfers.lifeform_counter_pending = false;
            }
        }

        if self.transfers.species_counter_pending {
            profile_scope!("Consume Species Counter");
            if let Some(value) = self.buffers.try_consume_species_counter() {
                self.population.species = value;
                self.transfers.species_counter_pending = false;
            }
        }

        let bounds = {
            profile_scope!("Fetch Environment Bounds");
            let env = self.environment.lock();
            env.get_bounds()
        };

        if bounds != self.current_bounds {
            self.handle_bounds_resize(bounds);
        }
    }


    fn handle_bounds_resize(&mut self, bounds: Rect) {
        profile_scope!("Handle Bounds Resize");
        let new_nutrient_buffer = {
            profile_scope!("Resize Nutrient Grid Buffer");
            self.buffers
                .resize_nutrient_grid(&self.device, bounds)
        };

        profile_scope!("Rebuild Compute Pipelines");
        self.compute_pipelines = ComputePipelines::new(
            &self.device,
            self.buffers.cell_buffer_write(),
            &self.buffers.uniform_buffer,
            self.buffers.cell_free_list_buffer_write(),
            self.buffers.cell_counter_buffer(),
            self.buffers.spawn_buffer(),
            new_nutrient_buffer.as_ref(),
            self.buffers.link_buffer(),
            self.buffers.link_free_list_buffer(),
            self.buffers.cell_hash_bucket_heads_buffer(),
            self.buffers.cell_hash_next_indices_buffer(),
            self.buffers.grn_descriptor_buffer(),
            self.buffers.grn_units_buffer(),
            self.buffers.lifeforms_buffer(),
            self.buffers.lifeform_free_buffer(),
            self.buffers.next_lifeform_id_buffer(),
            self.buffers.genome_buffer(),
            self.buffers.species_entries_buffer(),
            self.buffers.species_free_buffer(),
            self.buffers.next_species_id_buffer(),
            self.buffers.next_gene_id_buffer(),
            self.buffers.lifeform_counter_buffer(),
            self.buffers.species_counter_buffer(),
            self.buffers.position_changes_buffer(),
        );

        self.current_bounds = bounds;
    }

}

/// Application structure - manages rendering at 60 FPS and window events
pub struct Application {
    simulation: Arc<parking_lot::Mutex<Simulation>>,
    render_pipelines: RenderPipelines,
    
    // Window and surface
    gpu: GpuDevice,
    
    // Rendering resources
    bounds_renderer: BoundsRenderer,
    ui_renderer: UiRenderer,
    ui_manager: UIManager,
    
    // Camera and UI
    camera: Camera,
    environment: Arc<parking_lot::Mutex<Environment>>,
    bounds: Rect,
    key_states: KeyStates,
    last_cursor_pos: Vec2,
    // Performance tracking
    last_render_step_count: u64,
    last_frame_time: Instant,
    last_render_time: Instant,
    target_frame_duration: std::time::Duration,
    frame_count: u32,
    last_fps_update: Instant,
    fps_frames: u32,
    fps: f32,
    last_cleanup: Instant,
    cleanup_interval: std::time::Duration,
    last_nutrient_dims: (u32, u32),
    
    // Simulation time tracking
    real_time: f32,
    speed: Arc<parking_lot::Mutex<f32>>, // Shared with simulation thread
    simulation_paused: Arc<AtomicBool>,
    
    // Deferred resize to avoid blocking the event loop
    pending_resize: Option<winit::dpi::PhysicalSize<u32>>,

    // Rendering control
    rendering_enabled: bool,
    show_grid: bool,
}

impl Application {
    
    fn new_initialized(
        _window: &Window,
        gpu: GpuDevice,
        render_pipelines: RenderPipelines,
        buffers: Arc<GpuBuffers>,
        bounds_renderer: BoundsRenderer,
        ui_renderer: UiRenderer,
        ui_manager: UIManager,
        camera: Camera,
        environment: Arc<parking_lot::Mutex<Environment>>,
        bounds: Rect,
    ) -> Self {
        let device = Arc::new(gpu.device.clone());
        let queue = Arc::new(gpu.queue.clone());
        
        // Clone compute_pipelines for simulation (they'll be moved into Simulation)
        // We need to create a new ComputePipelines for the simulation since it will own them
        // But we also need to keep a reference for render - so we'll create another one
        // Actually, let's clone the pipeline references by creating new pipelines
        // For now, we'll pass the same compute_pipelines to Simulation and store a reference here
        // Since Simulation needs to own them, we'll need to clone/create them separately
        let nutrient_buffer = buffers.nutrient_grid_buffer();

        let compute_pipelines_for_sim = ComputePipelines::new(
            &gpu.device,
            buffers.cell_buffer_write(), // Compute writes to write buffer
            &buffers.uniform_buffer,
            buffers.cell_free_list_buffer_write(), // Compute uses write buffer's free list
            buffers.cell_counter_buffer(),
            buffers.spawn_buffer(),
            nutrient_buffer.as_ref(),
            buffers.link_buffer(),
            buffers.link_free_list_buffer(),
            buffers.cell_hash_bucket_heads_buffer(),
            buffers.cell_hash_next_indices_buffer(),
            buffers.grn_descriptor_buffer(),
            buffers.grn_units_buffer(),
            buffers.lifeforms_buffer(),
            buffers.lifeform_free_buffer(),
            buffers.next_lifeform_id_buffer(),
            buffers.genome_buffer(),
            buffers.species_entries_buffer(),
            buffers.species_free_buffer(),
            buffers.next_species_id_buffer(),
            buffers.next_gene_id_buffer(),
            buffers.lifeform_counter_buffer(),
            buffers.species_counter_buffer(),
            buffers.position_changes_buffer(),
        );
        let speed = Arc::new(parking_lot::Mutex::new(0.05));

        let initial_nutrient_dims = buffers.nutrient_grid_dimensions();

        let simulation_inner = Simulation::new(
            device,
            queue,
            compute_pipelines_for_sim,
            buffers.clone(),
            environment.clone(),
            speed.clone(),
        );
        let simulation_paused = simulation_inner.paused_handle();
        let simulation = Arc::new(parking_lot::Mutex::new(simulation_inner));

        Self {
            simulation,
            render_pipelines,
            gpu,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            environment,
            bounds,
            key_states: KeyStates::default(),
            last_cursor_pos: Vec2::new(0.0, 0.0),
            last_render_step_count: 0,
            last_frame_time: Instant::now(),
            last_render_time: Instant::now(),
            target_frame_duration: std::time::Duration::from_secs_f64(1.0 / 60.0),
            frame_count: 0,
            last_fps_update: Instant::now(),
            fps_frames: 0,
            fps: 0.0,
            last_cleanup: Instant::now(),
            cleanup_interval: std::time::Duration::from_secs(5),
            last_nutrient_dims: initial_nutrient_dims,
            pending_resize: None,
            real_time: 0.0,
            speed,
            simulation_paused,
            rendering_enabled: true,
            show_grid: false,
        }
    }
    
    // Spawn simulation thread (runs as fast as possible)
    pub fn spawn_simulation_thread(&self) {
        let simulation = self.simulation.clone();
        
        thread::spawn(move || {
            // With double-buffering, we can run simulation as fast as possible!
            // Compute writes to write buffer, render reads from read buffer - no race conditions
            // Sleep time is controlled inside step() based on speed
            loop {
                simulation.lock().step();
            }
        });
    }
    
    // Render at 60fps (called from main event loop)
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        profile_scope!("Render Frame");
        let now = Instant::now();
        
        // Handle pending resize at the start of the render frame (avoids blocking event loop)
        {
            profile_scope!("Handle Pending Resize");
            if let Some(new_size) = self.pending_resize.take() {
                self.gpu.resize(new_size);
                
                // Update camera view size
                self.camera
                    .set_view_size(Vec2::new(new_size.width as f32, new_size.height as f32));
                
                self.ui_renderer.resize(new_size, &self.gpu.device, &self.gpu.config, &self.gpu.queue);
                // Update UI manager size
                self.ui_manager.resize(new_size.width as f32, new_size.height as f32);
            }
        }
        
        // Periodically collect free indices from dead cells (do this in render, not simulation update)
        // Note: sync_free_cell_count requires mutable access, so we skip it here
        // The simulation thread can handle this if needed
        if now.duration_since(self.last_cleanup) >= self.cleanup_interval {
            self.last_cleanup = now;
            // Sync is skipped - buffers are read-only from render thread
        }
        
        // Update camera and UI (happens every render frame)
        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        if !self.rendering_enabled {
            self.last_render_time = now;
            return Ok(());
        }
        
        // Get current simulation state and clone buffers
        let (current_step, buffers, bounds, alive_count, lifeform_count, species_count) = {
            profile_scope!("Sync Simulation State");
            let simulation = self.simulation.lock();
            // No need to flush - GPU works asynchronously and render doesn't need immediate completion
            {
                profile_scope!("Get Simulation State");
                let current_step = simulation.get_step_count();
                let buffers = simulation.get_buffers();
                let alive_count = simulation.population.cells;
                let lifeform_count = simulation.population.lifeforms;
                let species_count = simulation.population.species;
                drop(simulation);
                let environment = self.environment.lock();
                let bounds = environment.get_bounds();
                (current_step, buffers, bounds, alive_count, lifeform_count, species_count)
            }
        };

        let nutrient_dims = buffers.nutrient_grid_dimensions();
        if nutrient_dims != self.last_nutrient_dims {
            profile_scope!("Rebuild Render Pipelines");
            let nutrient_buffer = buffers.nutrient_grid_buffer();
            self.render_pipelines = RenderPipelines::new(
                &self.gpu.device,
                &self.gpu.config,
                buffers.cell_buffer(),
                &buffers.uniform_buffer,
                buffers.cell_free_list_buffer(),
                buffers.link_buffer(),
                nutrient_buffer.as_ref(),
                buffers.cell_hash_bucket_heads_buffer(),
                buffers.cell_hash_bucket_heads_readonly_buffer(),
                buffers.cell_hash_next_indices_buffer(),
            );
            self.last_nutrient_dims = nutrient_dims;
        }
        
        // Calculate simulation rate
        let _steps_per_frame = current_step - self.last_render_step_count;
        self.last_render_step_count = current_step;
        
        // Update framerate tracking
        self.fps_frames += 1;
        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.05 {
            self.fps = (self.fps_frames as f32 / now.duration_since(self.last_fps_update).as_secs_f32()).round();
            self.fps_frames = 0;
            self.last_fps_update = now;
        }

        // Update camera (uses actual frame time for smooth camera movement)
        self.camera.update(delta_time, &self.key_states);

        // Update camera scene bounds when environment bounds change
        self.camera.set_scene_bounds(Some(bounds));

        // Update bounds border
        self.bounds = bounds;
        
        // Update simulation time (advance only if playing)
        {
            profile_scope!("Advance Simulation Clock");
            if !self.simulation_paused.load(Ordering::Relaxed) {
                let speed = *self.speed.lock();
                self.real_time += delta_time * speed;
            }
        }
        
        // Update uniforms for rendering (use actual render delta time for UI effects)
        // CRITICAL: Only the render thread updates uniforms to avoid race conditions
        // The compute shader uses the same uniform buffer but only reads bounds/capacity
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let view_size = Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32);
        let active_cells = alive_count as f32;
        
        let _render_delta = now.duration_since(self.last_render_time).as_secs_f32();
        
        // Use SIMULATION_DELTA_TIME for compute shader physics, but render_delta for UI
        // The compute shader uses delta_time for physics calculations
        // Note: Speed is controlled by running more/fewer simulation steps, not by changing dt
        let (nutrient_grid_width, nutrient_grid_height) = buffers.nutrient_grid_dimensions();
        let uniforms = Uniforms::new(
            SIMULATION_DELTA_TIME, // Use fixed timestep for physics consistency
            [camera_pos.x, camera_pos.y],
            zoom,
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
            active_cells,
            buffers.nutrient_cell_size(),
            buffers.nutrient_scale(),
            nutrient_grid_width,
            nutrient_grid_height,
        );

        // Update uniforms on the buffers (this updates the uniform buffer for both compute and render)
        // This happens on the render thread before rendering to ensure consistency
        {
            profile_scope!("Update Uniform Buffer");
            buffers.update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));
        }

        // Update UI with current state
        {
            profile_scope!("Update UI State");
            let time_string = self.get_time_string(current_step);
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                if let Some(fps_component) = screen.find_element_by_id("fps") {
                    let fps_text = format!("{}fps", self.fps);
                    fps_component.update_text(&fps_text);
                }
                if let Some(step_component) = screen.find_element_by_id("step") {
                    let step_text = format_number(format!("{}", current_step));
                    step_component.update_text(&step_text);
                }
                
                // Update time string if there's a component with id "time"
                if let Some(time_component) = screen.find_element_by_id("time") {
                    time_component.update_text(&time_string);
                }

                // Get current simulation state and clone buffers
                let cell_count = alive_count as usize;
                let lifeform_display_count = lifeform_count as usize;
                if let Some(species_component) = screen.find_element_by_id("species") {
                    let species_text = format_number(format!("{}", species_count));
                    species_component.update_text(&species_text);
                }
                if let Some(lifeforms_component) = screen.find_element_by_id("lifeforms") {
                    let lifeforms_text = format_number(format!("{}", lifeform_display_count));
                    lifeforms_component.update_text(&lifeforms_text);
                }
                if let Some(cells_component) = screen.find_element_by_id("cells") {
                    let cells_text = format_number(format!("{}", cell_count));
                    cells_component.update_text(&cells_text);
                }
            }
        };
        
        // Render frame
        let bounds_corners: [Vec2; 4] = [
            Vec2::new(self.bounds.left, self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.bottom()),
            Vec2::new(self.bounds.left, self.bounds.bottom()),
        ];
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        
        // Use actual cell size (number of initialized cells) instead of capacity
        // This ensures we only render the cells that actually exist
        let num_cells_to_render = buffers.cell_capacity();
        let num_links_to_render = buffers.link_capacity();
        
        // Get surface texture and create encoder
        let output = {
            profile_scope!("get_current_texture");
            self.gpu.surface.get_current_texture()?
        };
        
        let view = {
            profile_scope!("create_view");
            output.texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Command Encoder"),
        });

        // STEP 1: Render simulation to viewport texture (if it exists)
        // This renders cells and bounds to the viewport texture
        {
            profile_scope!("Render Simulation Viewport");
            let mut environment = self.environment.lock();
            Renderer::render_simulation(
                &mut self.gpu,
                &*buffers,
                &self.render_pipelines,
                &mut self.bounds_renderer,
                &mut *environment,
                &mut self.ui_renderer,
                &mut self.ui_manager,
                bounds_corners,
                camera_pos,
                zoom,
                num_cells_to_render,
                num_links_to_render,
                self.show_grid,
                &mut encoder,
            );
        }

        // STEP 2: Render UI (backgrounds, viewport textures as sprites, overlays, text)
        // This renders everything in HTML order
        {
            profile_scope!("Render UI");
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                // Render all UI elements (queues text but doesn't draw it yet)
                for element in screen.get_elements_mut() {
                    self.ui_renderer.render(element, &self.gpu.device, &self.gpu.queue, &mut encoder, &view);
                }
                
                // Now render all queued text in a single batch
                self.ui_renderer.render_text(&self.gpu.device, &self.gpu.queue, &mut encoder, &view);
            }
        }

        // Submit and present
        {
            profile_scope!("Submit Frame");
            let command_buffer = encoder.finish();
            self.gpu.queue.submit(std::iter::once(command_buffer));
            output.present();
        }

        self.frame_count += 1;
        
        self.last_render_time = now;
        
        Ok(())
    }
    
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        // Defer resize to render thread to avoid blocking the event loop
        // This prevents freezing when resizing the window
        self.pending_resize = Some(new_size);
    }
    
    pub fn handle_keyboard_input(&mut self, pressed: bool, key_code: KeyCode) {
        match key_code {
            KeyCode::KeyW => self.key_states.w = pressed,
            KeyCode::KeyA => self.key_states.a = pressed,
            KeyCode::KeyS => self.key_states.s = pressed,
            KeyCode::KeyD => self.key_states.d = pressed,
            _ => {}
        }
    }
    
    pub fn handle_mouse_wheel(&mut self, delta: f32, mouse_pos: Vec2) {
        self.camera.zoom(delta, mouse_pos);
    }
    
    pub fn handle_mouse_move(&mut self, mouse_pos: Vec2, ui_hovered: bool) {
        // Convert screen coordinates to world coordinates
        let world_pos = self.camera.screen_to_world(mouse_pos);

        // Update environment drag handler
        let mut environment = self.environment.lock();
        environment
            .update(world_pos, self.camera.get_zoom(), ui_hovered);
    }
    
    pub fn handle_mouse_press(&mut self, pressed: bool) {
        if pressed && !self.rendering_enabled {
            self.set_ui_visibility(true);
            return;
        }

        let mut environment = self.environment.lock();
        environment.handle_mouse_press(pressed);
    }
    
    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        let environment = self.environment.lock();
        environment.get_cursor_hint()
    }
    
    pub fn get_time_string(&self, step: u64) -> String {
        let time = step as f32 * SIMULATION_DELTA_TIME;
        let hours = (time / 3600.0).floor() as i32;
        let minutes = ((time % 3600.0) / 60.0).floor() as i32;
        let seconds = (time % 60.0).floor() as i32;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
    
    pub fn set_last_cursor_pos(&mut self, pos: Vec2) {
        self.last_cursor_pos = pos;
    }
    
    pub fn speed_up(&mut self) {
        let mut speed = self.speed.lock();
        *speed *= 1.5;
        drop(speed);
        self.update_speed_display();
    }
    
    pub fn slow_down(&mut self) {
        let mut speed = self.speed.lock();
        *speed /= 1.5;
        drop(speed);
        self.update_speed_display();
    }
    
    pub fn toggle_paused(&mut self) {
        let new_state = {
            let was_paused = self.simulation_paused.load(Ordering::Relaxed);
            let mut simulation = self.simulation.lock();
            simulation.set_paused(!was_paused);
            simulation.is_paused()
        };

        
        // Update play button icon (toggle between play and pause)
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(play_icon) = screen.find_element_by_id("playBtnIcon") {
                use crate::ui::components::ComponentType;
                if let ComponentType::Image(ref mut image) = play_icon.component_type {
                    if new_state {
                        // Currently paused - show play icon (clicking will resume)
                        image.set_source("play");
                        image.base_source = Some("play".to_string());
                        image.set_group_hover_source("playHighlighted");
                    } else {
                        // Currently playing - show pause icon (clicking will pause)
                        image.set_source("pause");
                        image.base_source = Some("pause".to_string());
                        image.set_group_hover_source("pauseHighlighted");
                    }
                }
            }
        }
    }

    pub fn toggle_ui(&mut self) {
        let mut target_visibility = None;
        {
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                if let Some(ui_element) = screen.find_element_by_id("UI") {
                    target_visibility = Some(!ui_element.is_visible());
                }
            }
        }

        if let Some(visible) = target_visibility {
            self.set_ui_visibility(visible);
        }
    }
    
    fn update_speed_display(&mut self) {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(speed_component) = screen.find_element_by_id("speed") {
                let speed = *self.speed.lock();
                let speed_text = format!("x{:.3}", speed);
                speed_component.update_text(&speed_text);
            }
        }
    }

    fn toggle_grid(&mut self) {
        self.show_grid = !self.show_grid;
    }
    
    fn handle_ui_function(&mut self, function_name: &str) {
        match function_name {
            "speedUp" => self.speed_up(),
            "slowDown" => self.slow_down(),
            "togglePaused" => self.toggle_paused(),
            "showUI" => self.toggle_ui(),
            "toggleGrid" => self.toggle_grid(),
            _ => {
                // Unknown function - log it for debugging
                eprintln!("Unknown UI function: {}", function_name);
            }
        }
    }
    
}

impl Application {
    fn set_ui_visibility(&mut self, visible: bool) {
        self.rendering_enabled = visible;

        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(ui_element) = screen.find_element_by_id("UI") {
                ui_element.set_visible(visible);
            }

            if let Some(show_ui_icon) = screen.find_element_by_id("showUIIcon") {
                use crate::ui::components::ComponentType;
                if let ComponentType::Image(ref mut image) = show_ui_icon.component_type {
                    if visible {
                        image.set_source("eye");
                        image.base_source = Some("eye".to_string());
                        image.set_group_hover_source("eyeHighlighted");
                    } else {
                        image.set_source("noEye");
                        image.base_source = Some("noEye".to_string());
                        image.set_group_hover_source("noEyeHighlighted");
                    }
                }
            }
        }
    }
}

/// Minimal Simulator wrapper - preserves original behavior exactly
/// This is the simplest possible wrapper that maintains the original pattern
pub struct Simulator {
    window: Option<Window>,
    application: Option<Application>,
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            window: None,
            application: None,
        }
    }

    pub fn run(mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(&mut self).unwrap();
    }
}

impl ApplicationHandler for Simulator {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Genetica Rust - Verlet Integration")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).unwrap();
        window.set_title("Genetica");

        let application = pollster::block_on(Application::new_from_window(&window));

        // Spawn simulation thread
        application.spawn_simulation_thread();

        self.window = Some(window);
        self.application = Some(application);
        
        // Request initial redraw to start the render loop
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(ref mut application) = self.application {
                    application.resize(physical_size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(ref mut application) = self.application {
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        application.handle_keyboard_input(
                            event.state == winit::event::ElementState::Pressed,
                            key_code,
                        );
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(ref mut application) = self.application {
                    let delta_scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 10.0,
                    };
                    application.handle_mouse_wheel(delta_scroll, application.last_cursor_pos);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(ref mut application) = self.application {
                    let pos = Vec2::new(position.x as f32, position.y as f32);
                    application.set_last_cursor_pos(pos);
                    
                    // Update UI hover states
                    let ui_hovered = if let Some(screen) = application.ui_manager.get_screen("simulation") {
                        screen.update(0.0, (pos.x, pos.y))
                    } else {
                        false
                    };
                    
                    application.handle_mouse_move(pos, ui_hovered);
                    if let Some(ref window) = self.window {
                        let cursor = match application.get_cursor_hint() {
                            Some("ew-resize") => CursorIcon::ColResize,
                            Some("ns-resize") => CursorIcon::RowResize,
                            Some("nwse-resize") => CursorIcon::NwseResize,
                            Some("nesw-resize") => CursorIcon::NeswResize,
                            _ => CursorIcon::Default,
                        };
                        window.set_cursor(cursor);
                    }
                }
            }
            WindowEvent::MouseInput { button, state: button_state, .. } => {
                if let Some(ref mut application) = self.application {
                    if button == winit::event::MouseButton::Left {
                        // Handle UI clicks first
                        if button_state == winit::event::ElementState::Pressed {
                            let mouse_pos = application.last_cursor_pos;
                            if let Some(handler_name) = application.ui_manager.handle_click((mouse_pos.x, mouse_pos.y)) {
                                if !handler_name.is_empty() {
                                    // Call the UI function
                                    application.handle_ui_function(&handler_name);
                                }
                                // Don't pass click to world if UI consumed it
                                return;
                            }
                        }
                        
                        // If UI didn't consume the click, pass it to the world
                        application.handle_mouse_press(
                            button_state == winit::event::ElementState::Pressed,
                        );
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                puffin::GlobalProfiler::lock().new_frame();
                if let Some(ref mut application) = self.application {
                    let now = Instant::now();
                    
                    // Only render if enough time has passed (60 FPS cap)
                    let time_since_last_render = now.duration_since(application.last_render_time);
                    if time_since_last_render >= application.target_frame_duration {
                        // Render frame only - no simulation
                        if let Err(e) = application.render() {
                            eprintln!("Error rendering frame: {}", e);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ref application) = self.application {
            let now = Instant::now();
            
            // Schedule next wake time for rendering
            let next_frame_time = application.last_render_time + application.target_frame_duration;
            
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(next_frame_time));
            
            // Request redraw when it's time for rendering
            if let Some(ref window) = self.window {
                if now >= next_frame_time {
                    window.request_redraw();
                }
            }
        } else {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
        }
    }
}

impl Application {
    async fn new_from_window(window: &Window) -> Self {
        let size = window.inner_size();

        // Initialize GPU device
        let gpu = GpuDevice::new(window).await;

        // Initialize environment
        let initial_bounds = Rect::new(0.0, 0.0, size.width as f32, size.height as f32);
        let mut environment = Environment::new(initial_bounds, &gpu);

        // Initialize camera (must be before buffers since we need camera data for uniforms)
        let camera = Camera::new(
            Vec2::new(size.width as f32, size.height as f32),
            Some(initial_bounds),
        );

        // Start with no pre-existing cells; GPU will populate as needed
        let cells: Vec<Cell> = Vec::new();

        // Initialize GPU buffers with initial data
        let bounds = environment.get_bounds();
        let camera_pos = camera.get_position();
        let zoom = camera.get_zoom();
        let view_size = Vec2::new(size.width as f32, size.height as f32);
        
        let buffers = Arc::new(GpuBuffers::new(
            &gpu.device,
            &gpu.queue,
            bytemuck::cast_slice(&cells),
            bytemuck::cast_slice(&[Uniforms::zeroed()]),
            bounds,
        ));
        
        let initial_cell_count = buffers.cell_size() as f32;

        let (nutrient_grid_width, nutrient_grid_height) = buffers.nutrient_grid_dimensions();
        let initial_uniforms = Uniforms::new(
            0.0,
            [camera_pos.x, camera_pos.y],
            zoom,
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
            initial_cell_count,
            buffers.nutrient_cell_size(),
            buffers.nutrient_scale(),
            nutrient_grid_width,
            nutrient_grid_height,
        );
        
        buffers.update_uniforms(&gpu.queue, bytemuck::cast_slice(&[initial_uniforms]));


        let render_nutrient_buffer = buffers.nutrient_grid_buffer();
        let render_pipelines = RenderPipelines::new(
            &gpu.device,
            &gpu.config,
            buffers.cell_buffer(),
            &buffers.uniform_buffer,
            buffers.cell_free_list_buffer(),
            buffers.link_buffer(),
            render_nutrient_buffer.as_ref(),
            buffers.cell_hash_bucket_heads_buffer(),
            buffers.cell_hash_bucket_heads_readonly_buffer(),
            buffers.cell_hash_next_indices_buffer(),
        );

        // Initialize bounds renderer (includes planet background)
        let bounds_renderer = BoundsRenderer::new(&gpu.device, &gpu.queue, &gpu.config);
        
        // Initialize planet GPU resources
        environment.planet_mut().initialize(&gpu.device, gpu.config.format);

        // Initialize UI renderer
        let ui_renderer = UiRenderer::new(
            &gpu.device,
            &gpu.queue,
            &gpu.config,
        );

        // Parse UI from HTML/CSS files and create UIManager with Screen
        let ui_manager = {
            let window_size = gpu.config.width as f32;
            let window_height = gpu.config.height as f32;
            let mut manager = UIManager::new(window_size, window_height);
            
            let screen = match UiParser::parse_to_screen(
                "assets/ui/simulation.html",
                &["assets/ui/tailwind.css"],
            ) {
                Ok(screen) => {
                    screen
                }
                Err(_e) => {
                    // Create a fallback screen with a visible background component
                    let mut fallback = crate::ui::Component::new(crate::ui::ComponentType::View(crate::ui::View::new()));
                    fallback.style.width = crate::ui::Size::Percent(100.0);
                    fallback.style.height = crate::ui::Size::Percent(100.0);
                    fallback.style.background_color = crate::ui::Color::new(0.5, 0.5, 0.5, 1.0); // Gray background
                    let mut screen = crate::ui::Screen::new();
                    screen.add_element(fallback);
                    screen
                }
            };
            
            manager.add_screen("simulation".to_string(), screen);
            // Set current screen to ensure it's active
            manager.set_current_screen("simulation".to_string());
            manager
        };
        
        // Wrap environment in Arc for sharing
        let environment = Arc::new(parking_lot::Mutex::new(environment));

        Self::new_initialized(
            window,
            gpu,
            render_pipelines,
            buffers,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            environment,
            initial_bounds,
        )
    }
}


