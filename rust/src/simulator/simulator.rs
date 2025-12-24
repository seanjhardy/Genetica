use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

const SIMULATION_DELTA_TIME: f32 = 0.1;

pub struct Simulation {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    compute_pipelines: ComputePipelines,
    buffers: Arc<GpuBuffers>,
    environment: Arc<parking_lot::Mutex<Environment>>,
    pause: PauseState,
    speed: Arc<parking_lot::Mutex<f32>>,
    population: PopulationState,
    transfers: GpuTransferState,
    current_bounds: Rect,
    initial_bounds: Rect,
    workgroup_size: u32,
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
        Self {
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
            initial_bounds,
            workgroup_size: 128,
            step_count: Arc::new(AtomicU64::new(0)),
            submission: SubmissionState::new(8, Duration::from_millis(2)),
        }
    }
    
    pub fn set_paused(&mut self, paused: bool) {
        self.pause.set(paused);
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

    fn flush_submissions(&mut self) {
        if self.submission.pending_command_buffers.is_empty() {
            return;
        }
        self.queue.submit(self.submission.pending_command_buffers.drain(..));
        self.submission.record_submission_time();
    }

    pub fn run_frame(&mut self) -> u32 {
        profile_scope!("Simulation Frame");
        
        self.poll_pending_counters();
        
        if self.pause.is_paused() {
            return 0;
        }
        
        let speed = (*self.speed.lock()).max(0.0);
        if speed <= 0.0 {
            return 0;
        }
        
        let bounds = self.environment.lock().get_bounds();
        if bounds != self.current_bounds {
            self.handle_bounds_resize(bounds);
        }
        
        let iterations = (speed * 2.0).max(1.0).min(50.0) as u32;
        self.run_compute_iterations(iterations);
        self.flush_submissions();
        
        iterations
    }

    fn run_compute_iterations(&mut self, max_iterations: u32) -> u32 {
        if max_iterations == 0 {
            return 0;
        }

        let cell_capacity = self.buffers.cell_capacity() as u32;
        let cell_workgroups = (cell_capacity + self.workgroup_size - 1) / self.workgroup_size;
        let hash_table_size = self.buffers.cell_hash_table_size() as u32;
        let hash_workgroups = (hash_table_size + self.workgroup_size - 1) / self.workgroup_size;
        let link_capacity = self.buffers.link_capacity() as u32;
        let link_workgroups = (link_capacity + self.workgroup_size - 1) / self.workgroup_size;

        let command_buffer = {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Simulation Encoder"),
            });

            {
                let mut nutrient_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Nutrient Pass"),
                    timestamp_writes: None,
                });
                let (grid_w, grid_h) = self.buffers.nutrient_grid_dimensions();
                let total_cells = grid_w.saturating_mul(grid_h);
                if total_cells > 0 {
                    let workgroups = (total_cells + 255) / 256;
                    nutrient_pass.set_pipeline(&self.compute_pipelines.update_nutrients);
                    nutrient_pass.set_bind_group(0, &self.compute_pipelines.update_nutrients_bind_group, &[]);
                    for _ in 0..max_iterations {
                        nutrient_pass.dispatch_workgroups(workgroups, 1, 1);
                    }
                }
            }

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Cell Compute Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_bind_group(0, &self.compute_pipelines.update_cells_bind_group, &[]);

                for _ in 0..max_iterations {
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

            self.schedule_counter_copies(&mut encoder);
            encoder.finish()
        };

        self.submission.pending_command_buffers.push(command_buffer);
        self.step_count.fetch_add(max_iterations as u64, Ordering::Relaxed);
        max_iterations
    }

    fn schedule_counter_copies(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if !self.transfers.cell_counter_pending {
            self.buffers.schedule_cell_counter_copy(encoder);
            self.transfers.cell_counter_pending = true;
        }
        if !self.transfers.lifeform_counter_pending {
            self.buffers.schedule_lifeform_counter_copy(encoder);
            self.transfers.lifeform_counter_pending = true;
        }
        if !self.transfers.species_counter_pending {
            self.buffers.schedule_species_counter_copy(encoder);
            self.transfers.species_counter_pending = true;
        }
    }

    fn poll_pending_counters(&mut self) {
        if self.transfers.cell_counter_pending {
            self.buffers.begin_cell_counter_map();
        }
        if self.transfers.lifeform_counter_pending {
            self.buffers.begin_lifeform_counter_map();
        }
        if self.transfers.species_counter_pending {
            self.buffers.begin_species_counter_map();
        }

        let _ = self.device.poll(wgpu::MaintainBase::Poll);

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

    fn handle_bounds_resize(&mut self, bounds: Rect) {
        let new_nutrient_buffer = self.buffers.resize_nutrient_grid(&self.device, bounds);

        self.compute_pipelines = ComputePipelines::new(
            &self.device,
            self.buffers.cell_buffer(),
            &self.buffers.uniform_buffer,
            self.buffers.cell_free_list_buffer(),
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
            self.buffers.genome_event_buffer(),
        );

        self.current_bounds = bounds;
    }

    pub fn reset(&mut self) {
        self.flush_submissions();
        let _ = self.device.poll(wgpu::MaintainBase::Wait);
        
        self.buffers.reset(&self.device, &self.queue, self.initial_bounds);
        
        {
            let mut env = self.environment.lock();
            env.set_bounds(self.initial_bounds);
        }
        
        if self.current_bounds != self.initial_bounds {
            self.handle_bounds_resize(self.initial_bounds);
        }
        
        self.step_count.store(0, Ordering::Relaxed);
        self.population = PopulationState::new();
        self.transfers = GpuTransferState::default();
    }
}

pub struct Application {
    simulation: Arc<parking_lot::Mutex<Simulation>>,
    simulation_thread: Option<std::thread::JoinHandle<()>>,
    simulation_running: Arc<AtomicBool>,
    render_pipelines: RenderPipelines,
    gpu: GpuDevice,
    bounds_renderer: BoundsRenderer,
    ui_renderer: UiRenderer,
    ui_manager: UIManager,
    camera: Camera,
    initial_camera_position: Vec2,
    initial_camera_zoom: f32,
    environment: Arc<parking_lot::Mutex<Environment>>,
    bounds: Rect,
    key_states: KeyStates,
    last_cursor_pos: Vec2,
    last_render_step_count: u64,
    last_frame_time: Instant,
    last_render_time: Instant,
    frame_count: u64,
    last_fps_update: Instant,
    fps_frames: u32,
    fps: f32,
    last_cleanup: Instant,
    cleanup_interval: Duration,
    last_nutrient_dims: (u32, u32),
    pending_resize: Option<winit::dpi::PhysicalSize<u32>>,
    real_time: f32,
    speed: Arc<parking_lot::Mutex<f32>>,
    simulation_paused: Arc<AtomicBool>,
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
        let nutrient_buffer = buffers.nutrient_grid_buffer();

        let compute_pipelines = ComputePipelines::new(
            &gpu.device,
            buffers.cell_buffer(),
            &buffers.uniform_buffer,
            buffers.cell_free_list_buffer(),
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
            buffers.genome_event_buffer(),
        );
        
        let speed = Arc::new(parking_lot::Mutex::new(1.0));
        let initial_nutrient_dims = buffers.nutrient_grid_dimensions();

        let simulation_inner = Simulation::new(
            device,
            queue,
            compute_pipelines,
            buffers.clone(),
            environment.clone(),
            speed.clone(),
        );
        let simulation_paused = simulation_inner.paused_handle();
        let simulation = Arc::new(parking_lot::Mutex::new(simulation_inner));
        let simulation_running = Arc::new(AtomicBool::new(true));

        let initial_camera_position = camera.get_position();
        let initial_camera_zoom = camera.get_zoom();
        
        // Spawn background simulation thread
        let sim_thread_handle = {
            let simulation = simulation.clone();
            let running = simulation_running.clone();
            std::thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    {
                        let mut sim = simulation.lock();
                        sim.run_frame();
                    }
                    // Small yield to prevent busy-waiting when paused
                    std::thread::sleep(Duration::from_micros(100));
                }
            })
        };
        
        let mut app = Self {
            simulation,
            simulation_thread: Some(sim_thread_handle),
            simulation_running,
            render_pipelines,
            gpu,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            initial_camera_position,
            initial_camera_zoom,
            environment,
            bounds,
            key_states: KeyStates::default(),
            last_cursor_pos: Vec2::new(0.0, 0.0),
            last_render_step_count: 0,
            last_frame_time: Instant::now(),
            last_render_time: Instant::now(),
            frame_count: 0,
            last_fps_update: Instant::now(),
            fps_frames: 0,
            fps: 0.0,
            last_cleanup: Instant::now(),
            cleanup_interval: Duration::from_secs(5),
            last_nutrient_dims: initial_nutrient_dims,
            pending_resize: None,
            real_time: 0.0,
            speed,
            simulation_paused,
            rendering_enabled: true,
            show_grid: false,
        };
        app.update_speed_display();
        app
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        profile_scope!("Render Frame");
        let now = Instant::now();
        
        if let Some(new_size) = self.pending_resize.take() {
            self.gpu.resize(new_size);
            self.camera.set_view_size(Vec2::new(new_size.width as f32, new_size.height as f32));
            self.ui_renderer.resize(new_size, &self.gpu.device, &self.gpu.config, &self.gpu.queue);
            self.ui_manager.resize(new_size.width as f32, new_size.height as f32);
        }
        
        if now.duration_since(self.last_cleanup) >= self.cleanup_interval {
            self.last_cleanup = now;
        }
        
        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        let (current_step, buffers, bounds, alive_count, lifeform_count, species_count) = {
            profile_scope!("Read Simulation State");
            let simulation = self.simulation.lock();
            let current_step = simulation.get_step_count();
            let buffers = simulation.get_buffers();
            let alive_count = simulation.population.cells;
            let lifeform_count = simulation.population.lifeforms;
            let species_count = simulation.population.species;
            drop(simulation);
            let environment = self.environment.lock();
            let bounds = environment.get_bounds();
            (current_step, buffers, bounds, alive_count, lifeform_count, species_count)
        };

        if !self.rendering_enabled {
            self.last_render_time = now;
            return Ok(());
        }

        let nutrient_dims = buffers.nutrient_grid_dimensions();
        if nutrient_dims != self.last_nutrient_dims {
            let nutrient_buffer = buffers.nutrient_grid_buffer();
            self.render_pipelines = RenderPipelines::new(
                &self.gpu.device,
                &self.gpu.queue,
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
        
        self.last_render_step_count = current_step;
        
        self.fps_frames += 1;
        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.05 {
            self.fps = (self.fps_frames as f32 / now.duration_since(self.last_fps_update).as_secs_f32()).round();
            self.fps_frames = 0;
            self.last_fps_update = now;
        }

        self.camera.update(delta_time, &self.key_states);
        self.camera.set_scene_bounds(Some(bounds));
        self.bounds = bounds;
        
        if !self.simulation_paused.load(Ordering::Relaxed) {
            let speed = *self.speed.lock();
            self.real_time += delta_time * speed;
        }
        
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let view_size = Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32);
        let (nutrient_grid_width, nutrient_grid_height) = buffers.nutrient_grid_dimensions();
        
        let uniforms = Uniforms::new(
            SIMULATION_DELTA_TIME,
            [camera_pos.x, camera_pos.y],
            zoom,
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
            alive_count as f32,
            buffers.nutrient_cell_size(),
            buffers.nutrient_scale(),
            nutrient_grid_width,
            nutrient_grid_height,
        );

        buffers.update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));

        {
            let time_string = self.get_time_string(current_step);
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                if let Some(fps_component) = screen.find_element_by_id("fps") {
                    fps_component.update_text(&format!("{}fps", self.fps));
                }
                if let Some(step_component) = screen.find_element_by_id("step") {
                    step_component.update_text(&format_number(format!("{}", current_step)));
                }
                if let Some(time_component) = screen.find_element_by_id("time") {
                    time_component.update_text(&time_string);
                }
                if let Some(species_component) = screen.find_element_by_id("species") {
                    species_component.update_text(&format_number(format!("{}", species_count)));
                }
                if let Some(lifeforms_component) = screen.find_element_by_id("lifeforms") {
                    lifeforms_component.update_text(&format_number(format!("{}", lifeform_count)));
                }
                if let Some(cells_component) = screen.find_element_by_id("cells") {
                    cells_component.update_text(&format_number(format!("{}", alive_count)));
                }
            }
        }
        
        let bounds_corners: [Vec2; 4] = [
            Vec2::new(self.bounds.left, self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.bottom()),
            Vec2::new(self.bounds.left, self.bounds.bottom()),
        ];
        
        let num_cells_to_render = buffers.cell_capacity();
        let num_links_to_render = buffers.link_capacity();
        
        let output = {
            profile_scope!("get_current_texture");
            self.gpu.surface.get_current_texture()?
        };
        
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
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

        {
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    self.ui_renderer.render(element, &self.gpu.device, &self.gpu.queue, &mut encoder, &view);
                }
                self.ui_renderer.render_text(&self.gpu.device, &self.gpu.queue, &mut encoder, &view);
            }
        }

        {
            profile_scope!("Submit Frame");
            self.gpu.queue.submit(std::iter::once(encoder.finish()));
            output.present();
        }

        self.frame_count += 1;
        self.last_render_time = now;
        
        Ok(())
    }
    
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
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
        let world_pos = self.camera.screen_to_world(mouse_pos);
        let mut environment = self.environment.lock();
        environment.update(world_pos, self.camera.get_zoom(), ui_hovered);
    }
    
    pub fn handle_mouse_press(&mut self, pressed: bool) {
        if pressed && !self.rendering_enabled {
            self.set_ui_visibility(true);
            return;
        }
        
        if pressed {
            let mouse_pos = (self.last_cursor_pos.x, self.last_cursor_pos.y);
            if let Some(handler) = self.ui_manager.handle_click(mouse_pos) {
                self.handle_ui_action(&handler);
                return;
            }
        }
        
        let mut environment = self.environment.lock();
        environment.handle_mouse_press(pressed);
    }
    
    fn handle_ui_action(&mut self, action: &str) {
        match action {
            "togglePaused" | "play_pause" => self.toggle_pause(),
            "speedUp" | "speed_up" => self.speed_up(),
            "slowDown" | "slow_down" => self.slow_down(),
            "reset" => self.reset_simulation(),
            "toggleGrid" | "toggle_grid" => self.toggle_grid(),
            "showUI" => self.set_ui_visibility(true),
            _ => {}
        }
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
        *speed = (*speed * 1.2).min(10.0);
        drop(speed);
        self.update_speed_display();
    }
    
    pub fn slow_down(&mut self) {
        let mut speed = self.speed.lock();
        *speed = (*speed / 1.2).max(0.01);
        drop(speed);
        self.update_speed_display();
    }
    
    pub fn toggle_pause(&mut self) {
        let is_paused = self.simulation_paused.load(Ordering::Relaxed);
        self.simulation_paused.store(!is_paused, Ordering::Relaxed);
        self.update_play_pause_button();
    }
    
    fn update_play_pause_button(&mut self) {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(play_pause) = screen.find_element_by_id("play_pause") {
                let is_paused = self.simulation_paused.load(Ordering::Relaxed);
                let icon_path = if is_paused {
                    "assets/icons/play.png"
                } else {
                    "assets/icons/pause.png"
                };
                if let crate::ui::ComponentType::Image(ref mut image) = play_pause.component_type {
                    image.set_source(icon_path);
                }
            }
        }
    }
    
    fn update_speed_display(&mut self) {
        let speed = *self.speed.lock();
        let speed_str = if speed >= 1.0 {
            format!("{:.1}x", speed)
        } else {
            format!("{:.2}x", speed)
        };
        
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(speed_component) = screen.find_element_by_id("speed") {
                speed_component.update_text(&speed_str);
            }
        }
    }
    
    pub fn toggle_grid(&mut self) {
        self.show_grid = !self.show_grid;
    }
    
    pub fn set_ui_visibility(&mut self, visible: bool) {
        self.rendering_enabled = visible;
    }
    
    pub fn toggle_ui_visibility(&mut self) {
        self.rendering_enabled = !self.rendering_enabled;
    }
    
    pub fn reset_simulation(&mut self) {
        let mut simulation = self.simulation.lock();
        simulation.reset();
        
        self.camera.set_position(self.initial_camera_position);
        self.camera.set_zoom(self.initial_camera_zoom);
        
        self.real_time = 0.0;
        self.last_render_step_count = 0;
    }
    
    pub fn shutdown(&mut self) {
        self.simulation_running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.simulation_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub struct ApplicationWrapper {
    app: Option<Application>,
    profiler_server: Option<puffin_http::Server>,
}

impl ApplicationWrapper {
    pub fn new() -> Self {
        let profiler_server = puffin_http::Server::new("127.0.0.1:8585").ok();
        puffin::set_scopes_on(true);
        
        Self {
            app: None,
            profiler_server,
        }
    }
}

impl ApplicationHandler for ApplicationWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }
        
        let window_attrs = Window::default_attributes()
            .with_title("Genetica")
            .with_inner_size(winit::dpi::LogicalSize::new(1600.0, 900.0));
        
        let window = match event_loop.create_window(window_attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                eprintln!("Failed to create window: {:?}", e);
                event_loop.exit();
                return;
            }
        };
        
        let app = match pollster::block_on(Application::new(window)) {
            Ok(app) => app,
            Err(e) => {
                eprintln!("Failed to initialize application: {:?}", e);
                event_loop.exit();
                return;
            }
        };
        
        self.app = Some(app);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        puffin::GlobalProfiler::lock().new_frame();
        
        let Some(app) = self.app.as_mut() else {
            return;
        };
        
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            
            WindowEvent::Resized(physical_size) => {
                if physical_size.width > 0 && physical_size.height > 0 {
                    app.resize(physical_size);
                }
            }
            
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key_code) = event.physical_key {
                    let pressed = event.state.is_pressed();
                    
                    match key_code {
                        KeyCode::Space if pressed => app.toggle_pause(),
                        KeyCode::Equal | KeyCode::NumpadAdd if pressed => app.speed_up(),
                        KeyCode::Minus | KeyCode::NumpadSubtract if pressed => app.slow_down(),
                        KeyCode::KeyG if pressed => app.toggle_grid(),
                        KeyCode::KeyH if pressed => app.toggle_ui_visibility(),
                        KeyCode::KeyR if pressed => app.reset_simulation(),
                        _ => app.handle_keyboard_input(pressed, key_code),
                    }
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                let mouse_pos = Vec2::new(position.x as f32, position.y as f32);
                let ui_hovered = if let Some(screen) = app.ui_manager.get_screen("simulation") {
                    screen.update(0.0, (mouse_pos.x, mouse_pos.y))
                } else {
                    false
                };
                app.handle_mouse_move(mouse_pos, ui_hovered);
                app.set_last_cursor_pos(mouse_pos);
            }
            
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                };
                app.handle_mouse_wheel(scroll, app.last_cursor_pos);
            }
            
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    let pressed = state.is_pressed();
                    app.handle_mouse_press(pressed);
                }
            }
            
            WindowEvent::RedrawRequested => {
                profile_scope!("RedrawRequested");
                
                if let Some(hint) = app.get_cursor_hint() {
                    let cursor = match hint {
                        "grab" | "grabbing" => CursorIcon::Grab,
                        "nwse-resize" => CursorIcon::NwseResize,
                        "nesw-resize" => CursorIcon::NeswResize,
                        "ew-resize" => CursorIcon::EwResize,
                        "ns-resize" => CursorIcon::NsResize,
                        _ => CursorIcon::Default,
                    };
                    app.gpu.window().set_cursor(cursor);
                } else {
                    app.gpu.window().set_cursor(CursorIcon::Default);
                }
                
                match app.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        app.gpu.resize(app.gpu.size);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        eprintln!("Out of GPU memory!");
                        event_loop.exit();
                    }
                    Err(e) => eprintln!("Render error: {:?}", e),
                }
                
                app.gpu.window().request_redraw();
            }
            
            _ => {}
        }
    }
}

impl Application {
    pub async fn new(window: Arc<Window>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let gpu = GpuDevice::new(window.clone()).await;
        
        let initial_bounds = Rect::new(-500.0, -500.0, 1000.0, 1000.0);
        let mut environment = Environment::new(initial_bounds, &gpu);
        
        let view_size = Vec2::new(gpu.config.width as f32, gpu.config.height as f32);
        let camera = Camera::new(view_size, Some(initial_bounds));

        let (nutrient_grid_width, nutrient_grid_height) = {
            let width = ((initial_bounds.width / 20.0).ceil() as u32).max(1);
            let height = ((initial_bounds.height / 20.0).ceil() as u32).max(1);
            (width, height)
        };

        let initial_uniforms = Uniforms::new(
            SIMULATION_DELTA_TIME,
            [camera.get_position().x, camera.get_position().y],
            camera.get_zoom(),
            initial_bounds.left,
            initial_bounds.top,
            initial_bounds.right(),
            initial_bounds.bottom(),
            gpu.config.width as f32,
            gpu.config.height as f32,
            0.0,
            20,
            4_000_000_000,
            nutrient_grid_width,
            nutrient_grid_height,
        );
        
        let cells_data: Vec<Cell> = Vec::new();

        let buffers = Arc::new(GpuBuffers::new(
            &gpu.device,
            &gpu.queue,
            bytemuck::cast_slice(&cells_data),
            bytemuck::cast_slice(&[initial_uniforms]),
            initial_bounds,
        ));
        
        buffers.update_uniforms(&gpu.queue, bytemuck::cast_slice(&[initial_uniforms]));

        let render_nutrient_buffer = buffers.nutrient_grid_buffer();
        let render_pipelines = RenderPipelines::new(
            &gpu.device,
            &gpu.queue,
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

        let bounds_renderer = BoundsRenderer::new(&gpu.device, &gpu.queue, &gpu.config);
        environment.planet_mut().initialize(&gpu.device, gpu.config.format);

        let ui_renderer = UiRenderer::new(&gpu.device, &gpu.queue, &gpu.config);

        let ui_manager = {
            let mut manager = UIManager::new(gpu.config.width as f32, gpu.config.height as f32);
            
            let screen = match UiParser::parse_to_screen(
                "assets/ui/simulation.html",
                &["assets/ui/tailwind.css"],
            ) {
                Ok(screen) => screen,
                Err(_e) => {
                    let mut fallback = crate::ui::Component::new(crate::ui::ComponentType::View(crate::ui::View::new()));
                    fallback.style.width = crate::ui::Size::Percent(100.0);
                    fallback.style.height = crate::ui::Size::Percent(100.0);
                    fallback.style.background_color = crate::ui::Color::new(0.5, 0.5, 0.5, 1.0);
                    let mut screen = crate::ui::Screen::new();
                    screen.add_element(fallback);
                    screen
                }
            };
            
            manager.add_screen("simulation".to_string(), screen);
            manager.set_current_screen("simulation".to_string());
            manager
        };
        
        let environment = Arc::new(parking_lot::Mutex::new(environment));

        Ok(Self::new_initialized(
            &window,
            gpu,
            render_pipelines,
            buffers,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            environment,
            initial_bounds,
        ))
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ApplicationWrapper::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

