use std::sync::Arc;
use std::time::{Duration, Instant};

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
use crate::gpu::pipelines::{RenderPipelines};
use crate::gpu::uniforms::Uniforms;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::simulator::environment::Environment;
use crate::simulator::renderer::Renderer;
use crate::simulator::simulator::Simulation;

const SIMULATION_DELTA_TIME: f32 = 1.0;

pub struct Application {
    simulation: Simulation,
    render_pipelines: RenderPipelines,
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
    last_render_step_count: u64,
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
    is_real_time: bool,
    speed: Arc<parking_lot::Mutex<f32>>,
    realtime_frame_counter: u32,
    rendering_enabled: bool,
    show_grid: bool,
    uniforms_need_update: bool,
    last_render_slot: usize,

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
        let render_buffers = simulation.get_render_buffers();
        let initial_nutrient_dims = render_buffers.nutrient_grid_dimensions();

        let initial_camera_position = camera.get_position();
        let initial_camera_zoom = camera.get_zoom();

        let mut app = Self {
            simulation,
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
            ui_hovered: false,
            ui_cursor_hint: None,
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
            is_real_time: false,
            speed,
            realtime_frame_counter: 0,
            rendering_enabled: true,
            show_grid: false,
            uniforms_need_update: true, // Need initial update
            last_render_slot: 0,
            compute_time_accum: Duration::ZERO,
            compute_iterations: 0,
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

        if !self.simulation.is_paused() {
            puffin::profile_scope!("Compute Pass");
            let speed = *self.speed.lock();
            self.real_time += delta_time * speed;

            // Time the compute operations
            let compute_start = Instant::now();

            if self.is_real_time {
                // In realtime mode, step once every 2 frames
                if self.realtime_frame_counter % 2 == 0 {
                    self.simulation.step_simulation(&mut encoder);
                }
            } else {
                for _ in 0..iterations {
                    self.simulation.step_simulation(&mut encoder);
                }
            }

            let compute_duration = compute_start.elapsed();

            // Update profiling data
            self.compute_time_accum += compute_duration;
            self.compute_iterations += iterations as u32;
        }

        // Check if render slot changed and update render pipelines if needed
        let current_render_slot = self.simulation.render_slot;
        if current_render_slot != self.last_render_slot {
            profile_scope!("Update Render Pipelines");
            let buffers = self.simulation.get_render_buffers();
            self.render_pipelines = RenderPipelines::new(
                &self.gpu.device,
                &self.gpu.queue,
                &self.gpu.config,
                &buffers,
            );
            self.last_render_slot = current_render_slot;
            // Mark uniforms for update since we're now using a different buffer
            self.uniforms_need_update = true;
        }

        // Skip rendering if UI is hidden, but still submit GPU work for simulation
        if !self.rendering_enabled {
            self.submit_gpu_work(encoder);
            return Ok(());
        }


        let (points_count, cells_count, lifeform_count, species_count) = if self.last_render_step_count % 100 == 0 {
                profile_scope!("Read Counters");
                let render_slot_idx = self.simulation.render_slot;
                let slot = &mut self.simulation.slots[render_slot_idx];
                let buffers = &slot.buffers;
                buffers.points_counter.begin_map_if_ready();
                buffers.points_counter.schedule_copy_if_idle(&mut encoder);
                buffers.cells_counter.begin_map_if_ready();
                buffers.cells_counter.schedule_copy_if_idle(&mut encoder);
                
                (buffers.points_counter.try_read(), 
                buffers.cells_counter.try_read(),
                self.simulation.genetic_algorithm.num_lifeforms(),
                self.simulation.genetic_algorithm.num_species())
        } else {
            let render_slot_idx = self.simulation.render_slot;
            let slot = &mut self.simulation.slots[render_slot_idx];
            let buffers = &slot.buffers;
            (
                buffers.points_counter.get_last(),
                buffers.cells_counter.get_last(),
                self.simulation.genetic_algorithm.num_lifeforms(),
                self.simulation.genetic_algorithm.num_species()
            )
        };

        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let view_size = Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32);
        //let (nutrient_grid_width, nutrient_grid_height) = buffers.nutrient_grid_dimensions();

        // Only update uniforms if camera moved, bounds changed, or window resized
        if self.uniforms_need_update {
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
                0.0, // cell count (not used for points)
                20,  // nutrient cell size (not used for points)
                4_000_000_000, // nutrient scale (not used for points)
                100, // nutrient grid width (not used for points)
                100, // nutrient grid height (not used for points)
            );

            // Update uniforms for all slots to ensure consistency when slots rotate
            for slot in &self.simulation.slots {
                slot.buffers.update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));
            }
            self.uniforms_need_update = false;
        }

        let current_step = self.simulation.get_step();

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
            Renderer::render_simulation(
                &mut self.gpu,
                &self.simulation.get_render_buffers(),
                &self.render_pipelines,
                &mut self.bounds_renderer,
                &mut *environment,
                &mut self.ui_renderer,
                &mut self.ui_manager,
                bounds_corners,
                camera_pos,
                zoom,
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
            self.submit_gpu_work(encoder);
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
        let is_paused = self.simulation.is_paused();
        self.simulation.set_paused(!is_paused);
        self.update_play_pause_button();
    }

    fn update_play_pause_button(&mut self) {
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(play_btn_icon) = screen.find_element_by_id("playBtnIcon") {
                let icon_path = if self.simulation.is_paused() {
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
        self.simulation.reset();

        self.camera.set_position(self.initial_camera_position);
        self.camera.set_zoom(self.initial_camera_zoom);
        self.uniforms_need_update = true;

        self.real_time = 0.0;
        self.last_render_step_count = 0;
        self.realtime_frame_counter = 0;
    }

        /// Submits GPU work to the queue and polls the device for async operations
    fn submit_gpu_work(&self, encoder: wgpu::CommandEncoder) {
        profile_scope!("Submit GPU Work");
        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        // Poll device to process async operations like buffer mapping
        let _ = self.gpu.device.poll(wgpu::MaintainBase::Poll);
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

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ApplicationWrapper::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}


