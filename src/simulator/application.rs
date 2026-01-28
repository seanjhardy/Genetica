use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::sync::mpsc;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{PhysicalKey, KeyCode},
    window::{Window, WindowId, CursorIcon},
    dpi::PhysicalSize,
};
use crate::utils::math::{Rect, Vec2};
use crate::utils::camera::{Camera, KeyStates};
use crate::ui::{UiParser, UiRenderer, UIManager};
use crate::utils::strings::format_number;
use puffin::profile_scope;
use crate::utils::gpu::device::GpuDevice;
use crate::gpu::pipelines::RenderPipelines;
use crate::gpu::uniforms::Uniforms;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::gpu::buffers::{GpuBuffers, CELL_CAPACITY};
use crate::gpu::structures::{PickParams, PickResult, Point};
use crate::simulator::environment::Environment;
use crate::simulator::renderer::Renderer;
use crate::simulator::simulator::Simulation;
use crate::simulator::poll_thread::PollThread;
use rand::Rng;

const SIMULATION_DELTA_TIME: f32 = 1.0;
const PICK_WORKGROUP_SIZE: u32 = 256;


pub struct Application {
    simulation: Arc<parking_lot::Mutex<Simulation>>,
    render_pipelines: RenderPipelines,
    renderer: crate::simulator::renderer::Renderer,
    pub gpu: GpuDevice,
    bounds_renderer: BoundsRenderer,
    ui_renderer: UiRenderer,
    pub ui_manager: UIManager,
    camera: Camera,
    initial_camera_position: Vec2,
    initial_camera_zoom: f32,
    environment: Arc<parking_lot::Mutex<Environment>>,
    bounds: Rect,
    key_states: KeyStates,
    pub last_cursor_pos: Vec2,
    ui_hovered: bool,
    ui_cursor_hint: Option<&'static str>,
    mouse_pressed: bool,
    selected_cell: Option<u32>,
    selected_point: Option<u32>,
    drag_distance: f32,
    last_drag_dir: Vec2,
    last_drag_center: Vec2,
    dragging_cell: bool,
    last_frame_time: Instant,
    last_render_time: Instant,
    frame_count: u64,
    last_fps_update: Instant,
    fps: f32,
    last_cleanup: Instant,
    cleanup_interval: Duration,
    last_nutrient_dims: (u32, u32),
    pending_resize: Option<PhysicalSize<u32>>,
    real_time: f32,
    application_time: f32,
    is_real_time: bool,
    speed: Arc<parking_lot::Mutex<f32>>,
    realtime_frame_counter: u32,
    rendering_enabled: bool,
    show_grid: bool,
    uniforms_need_update: bool,
    last_render_slot: usize,
    poll_thread: PollThread,
    paused_state: Arc<parking_lot::Mutex<bool>>,
    genetic_algorithm: Arc<parking_lot::Mutex<crate::genetic_algorithm::GeneticAlgorithm>>,
    step_counter: Arc<AtomicUsize>,
    frame_start_nanos: Arc<AtomicU64>,
    render_buffers: Arc<parking_lot::Mutex<Arc<GpuBuffers>>>,
    app_start: Arc<Instant>,

    fps_frames: u32,
    compute_time_accum: Duration,
    compute_iterations: u32,
}

impl Application {

    pub fn new_initialized(
        _window: &Window,
        gpu: GpuDevice,
        render_pipelines: RenderPipelines,
        simulation: Simulation,
        bounds_renderer: BoundsRenderer,
        ui_renderer: UiRenderer,
        ui_manager: UIManager,
        camera: Camera,
        environment: Arc<parking_lot::Mutex<Environment>>,
        bounds: Rect,
        speed: Arc<parking_lot::Mutex<f32>>,
    ) -> Self {
        let simulation_arc = Arc::new(parking_lot::Mutex::new(simulation));
        let paused_state = Arc::new(parking_lot::Mutex::new(false));
        let genetic_algorithm = {
            let sim = simulation_arc.lock();
            sim.genetic_algorithm.clone()
        };
        let step_counter = {
            let sim = simulation_arc.lock();
            sim.step_counter()
        };
        let frame_start_nanos = Arc::new(AtomicU64::new(0));

        let render_buffers = {
            let sim = simulation_arc.lock();
            Arc::new(parking_lot::Mutex::new(sim.get_render_buffers()))
        };
        let initial_nutrient_dims = render_buffers.lock().nutrient_grid_dimensions();

        let initial_camera_position = camera.get_position();
        let initial_camera_zoom = camera.get_zoom();

        // Create the poll thread
        let app_start = Arc::new(Instant::now());
        let poll_thread = PollThread::new(
            Arc::new(gpu.device.clone()),
            render_buffers.clone(),
            paused_state.clone(),
            genetic_algorithm.clone(),
            step_counter.clone(),
            frame_start_nanos.clone(),
            app_start.clone(),
        );

        let mut app = Self {
            simulation: simulation_arc,
            render_pipelines,
            renderer: crate::simulator::renderer::Renderer::new(),
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
            ui_hovered: false,
            ui_cursor_hint: None,
            mouse_pressed: false,
            selected_cell: None,
            selected_point: None,
            drag_distance: 0.0,
            last_drag_dir: Vec2::new(1.0, 0.0),
            last_drag_center: Vec2::zero(),
            dragging_cell: false,
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
            application_time: 0.0,
            is_real_time: false,
            speed,
            realtime_frame_counter: 0,
            rendering_enabled: true,
            show_grid: false,
            uniforms_need_update: true, // Need initial update
            last_render_slot: 0,
            poll_thread,
            paused_state,
            genetic_algorithm,
            step_counter,
            frame_start_nanos,
            render_buffers,
            compute_time_accum: Duration::ZERO,
            compute_iterations: 0,
            app_start,
        };
        app.update_speed_display();
        app
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        profile_scope!("Render Frame");

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Main Encoder"),
        });

        let render_start = Instant::now();
        let elapsed = self.app_start.elapsed();
        self.frame_start_nanos
            .store(elapsed.as_nanos() as u64, Ordering::Release);
        let now = render_start;

        if let Some(new_size) = self.pending_resize.take() {
            self.gpu.resize(new_size);
            self.camera.set_view_size(Vec2::new(new_size.width as f32, new_size.height as f32));
            self.ui_renderer.resize(new_size, &self.gpu.device, &self.gpu.config, &self.gpu.queue);
            self.ui_manager.resize(new_size.width as f32, new_size.height as f32);
            self.uniforms_need_update = true;
        }

        if now.duration_since(self.last_cleanup) >= self.cleanup_interval {
            self.last_cleanup = now;
        }

        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Target 60 FPS = 16.67ms per frame. Reserve ~2ms for rendering overhead.
        let target_frame_time = 1.0 / 120.0;
        let compute_budget = target_frame_time * 0.8; // Use 80% of frame time for compute

        let speed = *self.speed.lock();

        // Calculate iterations based on previous compute performance and speed
        let iterations = if self.compute_iterations > 0 {
            let avg_compute_time = self.compute_time_accum.as_secs_f32() / self.compute_iterations as f32;
            let max_iterations = (compute_budget / avg_compute_time).floor() as usize;
            // Apply speed multiplier to iterations
            let adjusted_iterations = ((max_iterations as f32) * speed).floor() as usize;
            // Clamp between 1 and reasonable maximum (e.g., 100 iterations for faster than realtime)
            adjusted_iterations.max(1).min(100)
        } else {
            // Start with 8 iterations, adjusted by speed
            ((8.0 * speed).max(1.0) as usize).min(100)
        };

        self.fps_frames += 1;
        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.05 {
            self.fps = (self.fps_frames as f32 / now.duration_since(self.last_fps_update).as_secs_f32()).round();
            self.fps_frames = 0;
            self.last_fps_update = now;
        }

        // Update bounds from environment
        let bounds = self.environment.lock().get_bounds();
        if bounds != self.bounds {
            self.bounds = bounds;
            self.camera.set_scene_bounds(Some(bounds));
            self.uniforms_need_update = true;
        }

        // Update camera and check if it moved
        if self.camera.update(delta_time, &self.key_states) {
            self.uniforms_need_update = true;
        }

        // Move selected cell to cursor position every frame if dragging
        if self.dragging_cell {
            let adjusted_screen_pos = self.adjust_mouse_for_fisheye(self.last_cursor_pos);
            let world_pos = self.camera.screen_to_world(adjusted_screen_pos);
            self.drag_selected_cell(world_pos);
        }

        self.application_time += 1.0;

        // Schedule event readback BEFORE compute so we read the previous frame's buffer.
        let event_read_scheduled = {
            let current_buffers = self.simulation.lock().get_render_buffers();
            {
                let mut render_buffers = self.render_buffers.lock();
                if !Arc::ptr_eq(&current_buffers, &render_buffers) {
                    println!("event readback: render buffers out of sync, updating");
                }
                *render_buffers = current_buffers.clone();
            }
            current_buffers
                .event_system
                .try_schedule_readback(&mut encoder, &self.gpu.queue)
        };

        if !self.simulation.lock().is_paused() {
            puffin::profile_scope!("Compute Pass");
            let _speed = *self.speed.lock();
            self.real_time += delta_time * speed;

            // Time the compute operations
            let compute_start = Instant::now();

            if self.is_real_time {
                // In realtime mode, step once every 2 frames
                if self.realtime_frame_counter % 2 == 0 {
                    self.simulation.lock().step_simulation(&mut encoder, &self.gpu.queue);
                }
            } else {
                for _ in 0..iterations {
                    self.simulation.lock().step_simulation(&mut encoder, &self.gpu.queue);
                }
            }

            let compute_duration = compute_start.elapsed();
            // Update profiling data
            self.compute_time_accum += compute_duration;
            self.compute_iterations += iterations as u32;
        }

        // Skip rendering if UI is hidden, but still submit GPU work for simulation
        if !self.rendering_enabled {
            self.submit_gpu_work(encoder, event_read_scheduled);
            return Ok(());
        }

        // Regenerate animated noise texture every frame when rendering is enabled
        let time = self.application_time * 0.005; // Animation speed scales with simulation speed
        self.render_pipelines.regenerate_noise_texture(
            &self.gpu.device,
            &self.gpu.queue,
            time
        );

        // Check if render slot changed and update render pipelines if needed
        let (current_render_slot, buffers) = {
            let sim = self.simulation.lock();
            let current_render_slot = sim.render_slot;
            let buffers = sim.get_render_buffers();
            (current_render_slot, buffers)
        };
        if current_render_slot != self.last_render_slot {
            profile_scope!("Update Render Pipelines");
            self.render_pipelines = RenderPipelines::new(
                &self.gpu.device,
                &self.gpu.queue,
                &self.gpu.config,
                &buffers,
            );
            self.last_render_slot = current_render_slot;
            *self.render_buffers.lock() = buffers.clone();
            // Mark uniforms for update since we're now using a different buffer
            self.uniforms_need_update = true;
        }

        let (points_count, cells_count, lifeform_count, species_count) = {
                profile_scope!("Read Counters");
                let mut sim = self.simulation.lock();
                let render_slot_idx = sim.render_slot;
                let slot = &mut sim.slots[render_slot_idx];
                let buffers = &slot.buffers;
                buffers.points_counter.begin_map_if_ready();
                buffers.points_counter.schedule_copy_if_idle(&mut encoder);
                buffers.cells_counter.begin_map_if_ready();
                buffers.cells_counter.schedule_copy_if_idle(&mut encoder);

                let ga = self.genetic_algorithm.lock();
                (buffers.points_counter.try_read(),
                buffers.cells_counter.try_read(),
                ga.num_lifeforms(),
                ga.num_species())
        };

        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let viewport_size = self
            .ui_manager
            .get_screen("simulation")
            .and_then(|screen| screen.find_component_bounds("simulation"))
            .map(|bounds| Vec2::new(bounds.width, bounds.height))
            .filter(|size| size.x > 0.0 && size.y > 0.0);
        let view_size = viewport_size.unwrap_or_else(|| {
            Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32)
        });
        if view_size.x != self.gpu.config.width as f32 || view_size.y != self.gpu.config.height as f32 {
            self.uniforms_need_update = true;
        }
        //let (nutrient_grid_width, nutrient_grid_height) = buffers.nutrient_grid_dimensions();

        // Only update uniforms if camera moved, bounds changed, or window resized
        if self.uniforms_need_update {
            self.apply_uniforms_now();
        }

        let current_step = self.simulation.lock().get_step();

        {
            profile_scope!("Update UI");
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
                    cells_component.update_text(&format_number(format!("{}", cells_count)));
                }
                if let Some(points_component) = screen.find_element_by_id("points") {
                    points_component.update_text(&format_number(format!("{}", points_count)));
                }
            }
        }

        let bounds_corners: [Vec2; 4] = [
            Vec2::new(self.bounds.left, self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.top),
            Vec2::new(self.bounds.right(), self.bounds.bottom()),
            Vec2::new(self.bounds.left, self.bounds.bottom()),
        ];

        let output = {
            profile_scope!("get_current_texture");
            self.gpu.surface.get_current_texture()?
        };

        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            profile_scope!("Render Simulation");
            let mut environment = self.environment.lock();
            let render_buffers = self.simulation.lock().get_render_buffers();
        self.renderer.render_simulation(
                &mut self.gpu,
                &render_buffers,
                &mut self.render_pipelines,
                &mut self.bounds_renderer,
                &mut *environment,
                &mut self.ui_renderer,
                &mut self.ui_manager,
                bounds_corners,
                camera_pos,
                zoom,
                time,
                self.show_grid,
                &mut encoder,
            );
        }

        {
            profile_scope!("Render UI");
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    profile_scope!("Render UI Element");
                    self.ui_renderer.render(element, &self.gpu.device, &self.gpu.queue, &mut encoder, &view);
                }
                {
                    profile_scope!("Render UI Text");
                    self.ui_renderer.render_text(&self.gpu.device, &self.gpu.queue, &mut encoder, &view);
                }
            }
        }

        {
            self.submit_gpu_work(encoder, event_read_scheduled);
            output.present();
        }

        self.frame_count += 1;
        self.realtime_frame_counter += 1;
        self.last_render_time = now;

        Ok(())
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
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
        if self.camera.zoom(delta, mouse_pos) {
            self.uniforms_need_update = true;
        }
    }

    pub fn handle_mouse_move(&mut self, mouse_pos: Vec2, ui_hovered: bool) {
        self.ui_hovered = ui_hovered;
        let adjusted_screen_pos = self.adjust_mouse_for_fisheye(mouse_pos);
        let world_pos = self.camera.screen_to_world(adjusted_screen_pos);
        if self.dragging_cell && self.mouse_pressed {
            self.drag_selected_cell(world_pos);
            return;
        }
        let mut environment = self.environment.lock();
        environment.update(world_pos, self.camera.get_zoom(), ui_hovered);
    }

    pub fn handle_mouse_press(&mut self, pressed: bool) {
        self.mouse_pressed = pressed;
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

            if !self.is_mouse_in_simulation_viewport(self.last_cursor_pos) {
                self.clear_selected_cell();
                return;
            }

            if self.is_point_over_ui(self.last_cursor_pos) {
                return;
            }

            let adjusted_screen_pos = self.adjust_mouse_for_fisheye(self.last_cursor_pos);
            let world_pos = self.camera.screen_to_world(adjusted_screen_pos);
            let bounds = self.environment.lock().get_bounds();
            if !bounds.contains(world_pos) {
                self.clear_selected_cell();
                return;
            }

            if let Some(cell_idx) = self.pick_cell_at_world_pos(world_pos) {
                if self.select_cell_for_drag(cell_idx, world_pos) {
                    self.dragging_cell = true;
                } else {
                    self.clear_selected_cell();
                }
                return;
            }

            self.clear_selected_cell();
        }
        if !pressed {
            self.dragging_cell = false;
        }

        let mut environment = self.environment.lock();
        environment.handle_mouse_press(pressed);
    }

    fn handle_ui_action(&mut self, action: &str) {
        match action {
            "togglePaused" => self.toggle_pause(),
            "speedUp" => self.speed_up(),
            "slowDown" => self.slow_down(),
            "reset" => self.reset_simulation(),
            "toggleGrid" => self.toggle_grid(),
            "hideUI" => self.set_ui_visibility(false),
            "toggleRealtime" => self.toggle_realtime(),
            _ => {}
        }
    }

    fn is_point_over_ui(&mut self, mouse_pos: Vec2) -> bool {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            return screen.is_point_over_ui((mouse_pos.x, mouse_pos.y));
        }
        false
    }

    fn is_mouse_in_simulation_viewport(&mut self, mouse_pos: Vec2) -> bool {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(bounds) = screen.find_component_bounds("simulation") {
                return bounds.contains(mouse_pos);
            }
        }
        true
    }

    fn clear_selected_cell(&mut self) {
        if self.selected_cell.is_some() {
            self.selected_cell = None;
            self.selected_point = None;
            self.drag_distance = 0.0;
            self.last_drag_dir = Vec2::new(1.0, 0.0);
            self.last_drag_center = Vec2::zero();
            self.dragging_cell = false;
            self.uniforms_need_update = true;
        }
    }

    fn adjust_mouse_for_fisheye(&mut self, screen_pos: Vec2) -> Vec2 {
        let Some(screen) = self.ui_manager.get_screen("simulation") else {
            return screen_pos;
        };
        let Some(bounds) = screen.find_component_bounds("simulation") else {
            return screen_pos;
        };
        if bounds.width <= 0.0 || bounds.height <= 0.0 {
            return screen_pos;
        }

        let uv = Vec2::new(
            (screen_pos.x - bounds.left) / bounds.width,
            (screen_pos.y - bounds.top) / bounds.height,
        );
        if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
            return screen_pos;
        }

        let corrected_uv = self.inverse_fish_eye_uv(uv);
        Vec2::new(
            bounds.left + corrected_uv.x * bounds.width,
            bounds.top + corrected_uv.y * bounds.height,
        )
    }

    fn inverse_fish_eye_uv(&self, screen_uv: Vec2) -> Vec2 {
        let mut uv = screen_uv;
        for _ in 0..6 {
            let centered = Vec2::new(uv.x - 0.5, uv.y - 0.5);
            let dist = centered.length();
            let radius = 0.3 * dist * dist;
            let warped = Vec2::new(
                uv.x + centered.x * radius,
                uv.y + centered.y * radius,
            );
            let delta = Vec2::new(screen_uv.x - warped.x, screen_uv.y - warped.y);
            uv = Vec2::new(uv.x + delta.x, uv.y + delta.y);
        }
        uv
    }

    fn select_cell_for_drag(&mut self, cell_idx: u32, mouse_world_pos: Vec2) -> bool {
        let buffers = self.simulation.lock().get_render_buffers();
        let Some(cell) = buffers
            .cells
            .read_item_unchecked(&self.gpu.device, &self.gpu.queue, cell_idx)
        else {
            return false;
        };

        let Some(point) = buffers
            .points
            .read_item_unchecked(&self.gpu.device, &self.gpu.queue, cell.point_idx)
        else {
            return false;
        };

        let point_pos = Vec2::new(point.pos[0], point.pos[1]);
        self.selected_cell = Some(cell_idx);
        self.selected_point = Some(cell.point_idx);
        let grab_dir = point_pos - mouse_world_pos;
        self.drag_distance = mouse_world_pos.distance(&point_pos);
        self.last_drag_dir = if grab_dir.length() > 0.0001 {
            grab_dir.normalize()
        } else {
            Vec2::new(1.0, 0.0)
        };
        self.last_drag_center = point_pos;
        self.uniforms_need_update = true;
        true
    }

    fn drag_selected_cell(&mut self, mouse_world_pos: Vec2) {
        let Some(point_idx) = self.selected_point else {
            return;
        };

        let delta = self.last_drag_center - mouse_world_pos;
        let offset_dir = if delta.length() > 0.0001 {
            delta.normalize()
        } else {
            self.last_drag_dir
        };
        self.last_drag_dir = offset_dir;
        let target_pos = mouse_world_pos + offset_dir * self.drag_distance;
        self.last_drag_center = target_pos;

        let offset = (point_idx as usize * std::mem::size_of::<Point>()) as u64;
        let data = [target_pos.x, target_pos.y, target_pos.x, target_pos.y];
        let buffers = self.simulation.lock().get_render_buffers();
        self.gpu
            .queue
            .write_buffer(buffers.points.buffer(), offset, bytemuck::cast_slice(&data));
    }

    fn pick_cell_at_world_pos(&mut self, world_pos: Vec2) -> Option<u32> {
        let (buffers, pick_pipeline, pick_bind_group) = {
            let sim = self.simulation.lock();
            let slot = &sim.slots[sim.render_slot];
            (
                slot.buffers.clone(),
                slot.compute_pipelines.pick_cell.clone(),
                slot.compute_pipelines.pick_cell_bind_group.clone(),
            )
        };

        let pick_params = PickParams {
            mouse_pos: [world_pos.x, world_pos.y],
            _pad: [0.0, 0.0],
        };
        self.gpu.queue.write_buffer(
            &buffers.cell_pick_params,
            0,
            bytemuck::cast_slice(&[pick_params]),
        );

        let reset = PickResult::reset();
        self.gpu.queue.write_buffer(
            &buffers.cell_pick_result,
            0,
            bytemuck::cast_slice(&[reset]),
        );

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pick Cell Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Pick Cell Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pick_pipeline);
            pass.set_bind_group(0, &pick_bind_group, &[]);
            let dispatch = (CELL_CAPACITY as u32 + PICK_WORKGROUP_SIZE - 1) / PICK_WORKGROUP_SIZE;
            pass.dispatch_workgroups(dispatch, 1, 1);
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.gpu.device.poll(wgpu::MaintainBase::Wait);

        let result = self.read_pick_result(&buffers.cell_pick_result)?;
        if result.cell_index == u32::MAX {
            None
        } else {
            Some(result.cell_index)
        }
    }

    fn read_pick_result(&self, buffer: &wgpu::Buffer) -> Option<PickResult> {
        let size = std::mem::size_of::<PickResult>() as u64;
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pick Result Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Pick Result Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        loop {
            let _ = self.gpu.device.poll(wgpu::MaintainBase::Wait);
            match receiver.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(err)) => {
                    eprintln!("Pick result readback failed: {:?}", err);
                    return None;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Pick result readback channel disconnected");
                    return None;
                }
            }
        }

        let mapped = slice.get_mapped_range();
        let result = bytemuck::from_bytes::<PickResult>(&mapped[..]);
        let result = *result;
        drop(mapped);
        staging.unmap();
        Some(result)
    }

    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        if self.ui_hovered {
            self.ui_cursor_hint
        } else {
            let environment = self.environment.lock();
            environment.get_cursor_hint()
        }
    }

    pub fn get_time_string(&self, step: usize) -> String {
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
        *speed = (*speed * 1.5).min(1.0);
        drop(speed);
        self.update_speed_display();
    }

    pub fn slow_down(&mut self) {
        let mut speed = self.speed.lock();
        *speed = (*speed / 1.5).max(0.00001);
        drop(speed);
        self.update_speed_display();
    }

    pub fn toggle_pause(&mut self) {
        let is_paused = self.simulation.lock().is_paused();
        self.simulation.lock().set_paused(!is_paused);
        *self.paused_state.lock() = !is_paused;
        self.update_play_pause_button();
    }

    fn update_play_pause_button(&mut self) {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(play_btn_icon) = screen.find_element_by_id("playBtnIcon") {
                let icon_path = if self.simulation.lock().is_paused() {
                    "assets/icons/play.png"
                } else {
                    "assets/icons/pause.png"
                };
                if let crate::ui::ComponentType::Image(ref mut image) = play_btn_icon.component_type {
                    image.set_source(icon_path);
                }
            }
        }
    }

    fn update_speed_display(&mut self) {
        let speed_str = if self.is_real_time {
            "REALTIME".to_string()
        } else {
            let speed = *self.speed.lock();
            if speed >= 1.0 {
                format!("{:.1}x", speed)
            } else {
                format!("{:.2}x", speed)
            }
        };

        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(speed_component) = screen.find_element_by_id("speed") {
                speed_component.update_text(&speed_str);
            }
        }
    }

    pub fn toggle_realtime(&mut self) {
        self.is_real_time = !self.is_real_time;
        self.realtime_frame_counter = 0; // Reset frame counter when toggling
        self.update_speed_display();
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
        self.simulation.lock().reset();
        self.clear_selected_cell();

        self.camera.set_position(self.initial_camera_position);
        self.camera.set_zoom(self.initial_camera_zoom);
        {
            let mut environment = self.environment.lock();
            self.bounds = environment.get_bounds();
            environment
                .planet_mut()
                .update(&self.gpu.device, &self.gpu.queue, self.gpu.config.format);
            environment
                .planet_mut()
                .update_caustics(&self.gpu.device, &self.gpu.queue, self.application_time * 0.005);
        }
        self.uniforms_need_update = true;
        self.apply_uniforms_now();

        self.real_time = 0.0;
        self.realtime_frame_counter = 0;
    }

    fn apply_uniforms_now(&mut self) {
        let viewport_size = self
            .ui_manager
            .get_screen("simulation")
            .and_then(|screen| screen.find_component_bounds("simulation"))
            .map(|bounds| Vec2::new(bounds.width, bounds.height))
            .filter(|size| size.x > 0.0 && size.y > 0.0);
        let view_size = viewport_size.unwrap_or_else(|| {
            Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32)
        });
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let selected_cell = self.selected_cell.unwrap_or(u32::MAX);
        let spawn_seed = self.simulation.lock().spawn_seed();
        let uniforms = Uniforms::new(
            SIMULATION_DELTA_TIME,
            [camera_pos.x, camera_pos.y],
            zoom,
            self.bounds.left,
            self.bounds.top,
            self.bounds.right(),
            self.bounds.bottom(),
            view_size.x,
            view_size.y,
            0.0,
            20,
            4_000_000_000,
            100,
            100,
            selected_cell,
            spawn_seed,
        );

        let sim = self.simulation.lock();
        for slot in &sim.slots {
            slot.buffers
                .update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));
        }
        self.uniforms_need_update = false;
    }

        /// Submits GPU work to the queue
    fn submit_gpu_work(&mut self, encoder: wgpu::CommandEncoder, event_read_scheduled: bool) {
        profile_scope!("Submit GPU Work");
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        // Notify poll thread if we scheduled event reading this frame
        if event_read_scheduled {
            let render_buffers = self.render_buffers.lock();
            render_buffers.event_system.mark_readback_submitted();
            self.poll_thread.notify_event_scheduled();
        }
    }

}

pub struct ApplicationWrapper {
    app: Option<Application>,
    _profiler_server: Option<puffin_http::Server>,
}

impl ApplicationWrapper {
    pub fn new() -> Self {
        let profiler_server = puffin_http::Server::new("127.0.0.1:8586").ok();
        puffin::set_scopes_on(true);
        
        Self {
            app: None,
            _profiler_server: profiler_server,
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
                let (ui_hovered, ui_cursor_hint) = if let Some(screen) = app.ui_manager.get_screen("simulation") {
                    screen.update(0.0, (mouse_pos.x, mouse_pos.y))
                } else {
                    (false, None)
                };

                app.ui_cursor_hint = ui_cursor_hint;
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
                puffin::GlobalProfiler::lock().new_frame();
                profile_scope!("RedrawRequested");
                
                if let Some(hint) = app.get_cursor_hint() {
                    let cursor = match hint {
                        "grab" | "grabbing" => CursorIcon::Grab,
                        "nwse-resize" => CursorIcon::NwseResize,
                        "nesw-resize" => CursorIcon::NeswResize,
                        "ew-resize" => CursorIcon::EwResize,
                        "ns-resize" => CursorIcon::NsResize,
                        "pointer" => CursorIcon::Pointer,
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

        let mut rng = rand::thread_rng();
        let spawn_seed: u32 = rng.gen();

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
            u32::MAX,
            spawn_seed,
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
        let speed = Arc::new(parking_lot::Mutex::new(1.0));

        let simulation = Simulation::new(
            Arc::new(gpu.device.clone()),
            Arc::new(gpu.queue.clone()),
            environment.clone(),
            initial_uniforms,
            spawn_seed,
        );

        let render_buffers = simulation.get_render_buffers();

        let render_pipelines = RenderPipelines::new(
            &gpu.device,
            &gpu.queue,
            &gpu.config,
            &render_buffers,
        );

        Ok(Self::new_initialized(
            &window,
            gpu,
            render_pipelines,
            simulation,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            environment,
            initial_bounds,
            speed,
        ))
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        // Shutdown the poll thread when the application is dropped
        self.poll_thread.shutdown();
    }
}

impl Drop for ApplicationWrapper {
    fn drop(&mut self) {
        // The Application's drop will handle poll thread shutdown
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ApplicationWrapper::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
