// Simulator module - main simulation loop and window management

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorIcon},
};

use crate::modules::math::{Rect, Vec2};
use crate::modules::camera::{Camera, KeyStates};
use crate::modules::ui::{BoundsBorder};
use crate::ui::{UiParser, UiRenderer, UIManager};
use puffin::profile_scope;
use crate::gpu::device::GpuDevice;
use crate::gpu::buffers::GpuBuffers;
use crate::gpu::pipelines::{ComputePipelines, RenderPipelines};
use crate::gpu::uniforms::Uniforms;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::simulator::environment::Environment;
use crate::simulator::renderer::Renderer;

const NUM_LIFEFORMS: usize = 500;

// Fixed simulation timestep - large value for faster simulation at lower accuracy
// This is NOT tied to render frame time - simulation runs independently
const SIMULATION_DELTA_TIME: f32 = 0.1; // 100ms per simulation step

/// Simulation structure - runs compute updates as fast as possible
pub struct Simulation {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    compute_pipelines: ComputePipelines,
    buffers: Arc<GpuBuffers>,
    environment: Arc<parking_lot::Mutex<Environment>>,
    
    // Simulation parameters
    workgroup_size: u32,
    
    // Step counter (atomic for thread-safe access)
    step_count: Arc<AtomicU64>,
}

impl Simulation {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        compute_pipelines: ComputePipelines,
        buffers: Arc<GpuBuffers>,
        environment: Arc<parking_lot::Mutex<Environment>>,
    ) -> Self {
        Self {
            device,
            queue,
            compute_pipelines,
            buffers,
            environment,
            workgroup_size: 128, // Match compute shader workgroup size
            step_count: Arc::new(AtomicU64::new(0)),
        }
    }
    
    // Run a single simulation step (called as fast as possible)
    pub fn step(&mut self) {
        // DON'T update uniforms here - the render thread handles uniform updates
        // This prevents race conditions where both threads write to uniforms simultaneously
        // The compute shader doesn't need camera/view info anyway, just physics bounds
        
        // Get bounds for compute (but don't update uniforms - render thread does that)
        let environment = self.environment.lock();
        let _bounds = environment.get_bounds();
        drop(environment);
        
        // Submit simulation compute pass (non-blocking - GPU operations are async)
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Simulation Encoder"),
        });
        
        // Compute always reads and writes to the write buffer
        // The bind group was created pointing to the write buffer and never changes
        // After compute finishes, the render thread will copy write->read
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Update Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.compute_pipelines.update);
        compute_pass.set_bind_group(0, &self.compute_pipelines.compute_bind_group, &[]);
        // Dispatch for actual cell count (but uniforms.cell_capacity bounds the shader loop)
        let num_cells = self.buffers.cell_size() as u32;
        compute_pass.dispatch_workgroups((num_cells + self.workgroup_size - 1) / self.workgroup_size, 1, 1);
        
        drop(compute_pass);
        let command_buffer = encoder.finish();
        
        // Submit compute work - GPU will execute asynchronously
        // The render thread's uniform updates will ensure correct camera/bounds for rendering
        self.queue.submit(std::iter::once(command_buffer));
        
        // Increment step counter
        self.step_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_step_count(&self) -> u64 {
        self.step_count.load(Ordering::Relaxed)
    }
    
    pub fn get_buffers(&self) -> Arc<GpuBuffers> {
        self.buffers.clone()
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
    bounds_border: BoundsBorder,
    key_states: KeyStates,
    last_cursor_pos: Vec2,
    
    // Performance tracking
    last_render_step_count: u64,
    last_frame_time: std::time::Instant,
    last_render_time: std::time::Instant,
    target_frame_duration: std::time::Duration,
    frame_count: u32,
    last_fps_update: std::time::Instant,
    fps_frames: u32,
    last_cleanup: std::time::Instant,
    cleanup_interval: std::time::Duration,
    
    // Simulation time tracking
    real_time: f32,
    speed: f32,
    
    // Deferred resize to avoid blocking the event loop
    pending_resize: Option<winit::dpi::PhysicalSize<u32>>,
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
        bounds_border: BoundsBorder,
    ) -> Self {
        let device = Arc::new(gpu.device.clone());
        let queue = Arc::new(gpu.queue.clone());
        
        // Clone compute_pipelines for simulation (they'll be moved into Simulation)
        // We need to create a new ComputePipelines for the simulation since it will own them
        // But we also need to keep a reference for render - so we'll create another one
        // Actually, let's clone the pipeline references by creating new pipelines
        // For now, we'll pass the same compute_pipelines to Simulation and store a reference here
        // Since Simulation needs to own them, we'll need to clone/create them separately
        let compute_pipelines_for_sim = ComputePipelines::new(
            &gpu.device,
            buffers.cell_buffer_write(), // Compute writes to write buffer
            buffers.lifeform_buffer(),
            &buffers.uniform_buffer,
            buffers.cell_free_list_buffer_write(), // Compute uses write buffer's free list
        );

        let simulation = Arc::new(parking_lot::Mutex::new(Simulation::new(
            device,
            queue,
            compute_pipelines_for_sim,
            buffers,
            environment.clone(),
        )));
        
        Self {
            simulation,
            render_pipelines,
            gpu,
            bounds_renderer,
            ui_renderer,
            ui_manager,
            camera,
            environment,
            bounds_border,
            key_states: KeyStates::default(),
            last_cursor_pos: Vec2::new(0.0, 0.0),
            last_render_step_count: 0,
            last_frame_time: std::time::Instant::now(),
            last_render_time: std::time::Instant::now(),
            target_frame_duration: std::time::Duration::from_secs_f64(1.0 / 60.0),
            frame_count: 0,
            last_fps_update: std::time::Instant::now(),
            fps_frames: 0,
            last_cleanup: std::time::Instant::now(),
            cleanup_interval: std::time::Duration::from_secs(5),
            pending_resize: None,
            real_time: 0.0,
            speed: 1.0,
        }
    }
    
    // Spawn simulation thread (runs as fast as possible)
    pub fn spawn_simulation_thread(&self) {
        let simulation = self.simulation.clone();
        
        thread::spawn(move || {
            // With double-buffering, we can run simulation as fast as possible!
            // Compute writes to write buffer, render reads from read buffer - no race conditions
            loop {
                simulation.lock().step();
                // Small sleep to prevent burning CPU, but simulation can run much faster now
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
        });
    }
    
    // Render at 60fps (called from main event loop)
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let now = std::time::Instant::now();
        
        // Handle pending resize at the start of the render frame (avoids blocking event loop)
        if let Some(new_size) = self.pending_resize.take() {
            self.gpu.resize(new_size);
            
            // Update camera view size
            self.camera
                .set_view_size(Vec2::new(new_size.width as f32, new_size.height as f32));
            
            self.ui_renderer.resize(new_size, &self.gpu.device, &self.gpu.config, &self.gpu.queue);
            // Update UI manager size
            self.ui_manager.resize(new_size.width as f32, new_size.height as f32);
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
        
        // Get current simulation state and clone buffers
        let (current_step, buffers, bounds) = {
            let simulation = self.simulation.lock();
            let current_step = simulation.get_step_count();
            let buffers = simulation.get_buffers();
            let environment = self.environment.lock();
            let bounds = environment.get_bounds();
            (current_step, buffers, bounds)
        };
        
        // Calculate simulation rate
        let _steps_per_frame = current_step - self.last_render_step_count;
        self.last_render_step_count = current_step;
        
        // Update framerate tracking
        self.fps_frames += 1;
        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.05 {
            let _fps = self.fps_frames as f32 / now.duration_since(self.last_fps_update).as_secs_f32();
            self.fps_frames = 0;
            self.last_fps_update = now;
        }

        // Update camera (uses actual frame time for smooth camera movement)
        self.camera.update(delta_time, &self.key_states);

        // Update camera scene bounds when environment bounds change
        self.camera.set_scene_bounds(Some(bounds));

        // Update bounds border
        self.bounds_border.set_bounds(bounds);
        
        // Update simulation time (advance only if playing)
        // TODO: Add paused state tracking
        self.real_time += delta_time * self.speed;
        
        // Update uniforms for rendering (use actual render delta time for UI effects)
        // CRITICAL: Only the render thread updates uniforms to avoid race conditions
        // The compute shader uses the same uniform buffer but only reads bounds/capacity
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let view_size = Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32);
        
        let _render_delta = now.duration_since(self.last_render_time).as_secs_f32();
        
        let num_lifeforms = {
            let environment = self.environment.lock();
            environment.num_lifeforms() as u32
        };
        
        // Use SIMULATION_DELTA_TIME for compute shader physics, but render_delta for UI
        // The compute shader uses delta_time for physics calculations
        let uniforms = Uniforms::new(
            SIMULATION_DELTA_TIME, // Use fixed timestep for physics consistency
            [camera_pos.x, camera_pos.y],
            zoom,
            2.0,
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
            buffers.cell_capacity() as u32,
            num_lifeforms,
        );

        // Update uniforms on the buffers (this updates the uniform buffer for both compute and render)
        // This happens on the render thread before rendering to ensure consistency
        buffers.update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));
        
        // CRITICAL: We'll copy write->read in the render encoder to ensure proper ordering
        // For now, just recreate render bind group (the copy happens in Renderer::render)
        
        // Recreate render bind group to point to the read buffer (always the same, but recreate for safety)
        // This is necessary because bind groups reference buffers directly
        let render_bind_group_layout = self.render_pipelines.points.get_bind_group_layout(0);
        self.render_pipelines.render_bind_group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render Bind Group"),
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.cell_buffer_read().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_free_list_buffer_read().as_entire_binding(),
                },
            ],
        });
        
        // Update UI with current state
        let time_string = self.get_time_string();
        if let Some(screen) = self.ui_manager.get_screen("simulation") {
            if let Some(fps_component) = screen.find_element_by_id("fps") {
                let fps_text = format!("FPS: {:.1}", self.fps_frames);
                fps_component.update_text(&fps_text);
            }
            
            // Update step counter if there's a component with id "step"
            if let Some(step_component) = screen.find_element_by_id("step") {
                let step_text = format!("{}", current_step);
                step_component.update_text(&step_text);
            }
            
            // Update time string if there's a component with id "time"
            if let Some(time_component) = screen.find_element_by_id("time") {
                time_component.update_text(&time_string);
            }
        }
        
        // Render frame
        let bounds_corners = self.bounds_border.get_corners();
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        
        // Use actual cell size (number of initialized cells) instead of capacity
        // This ensures we only render the cells that actually exist
        let num_cells_to_render = buffers.cell_size();
        
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
        Renderer::render_simulation(
            &mut self.gpu,
            &*buffers,
            &self.render_pipelines,
            &mut self.bounds_renderer,
            &mut self.ui_renderer,
            &mut self.ui_manager,
            bounds_corners,
            camera_pos,
            zoom,
            num_cells_to_render,
            &mut encoder,
        );

        // STEP 2: Render UI (backgrounds, viewport textures as sprites, overlays, text)
        // This renders everything in HTML order
        {
            profile_scope!("Render UI");
            if let Some(screen) = self.ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    self.ui_renderer.render(element, &self.gpu.device, &self.gpu.queue, &mut encoder, &view);
                }
            }
        }

        // Submit and present
        {
            profile_scope!("Submit Commands");
            let command_buffer = encoder.finish();
            self.gpu.queue.submit(std::iter::once(command_buffer));
        }

        {
            profile_scope!("Present Frame");
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
        let mut environment = self.environment.lock();
        environment.handle_mouse_press(pressed);
    }
    
    pub fn get_cursor_hint(&self) -> Option<&'static str> {
        let environment = self.environment.lock();
        environment.get_cursor_hint()
    }
    
    pub fn get_time_string(&self) -> String {
        let time = self.real_time as i32;
        let hours = time / 3600;
        let minutes = (time % 3600) / 60;
        let seconds = time % 60;
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
    
    pub fn set_last_cursor_pos(&mut self, pos: Vec2) {
        self.last_cursor_pos = pos;
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
                    application.handle_mouse_move(pos, false);
                }
            }
            WindowEvent::MouseInput { button, state: button_state, .. } => {
                if let Some(ref mut application) = self.application {
                    if button == winit::event::MouseButton::Left {
                        application.handle_mouse_press(
                            button_state == winit::event::ElementState::Pressed,
                        );
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                puffin::GlobalProfiler::lock().new_frame();
                if let Some(ref mut application) = self.application {
                    if let Some(ref window) = self.window {
                        let now = std::time::Instant::now();
                        
                        // Only render if enough time has passed (60 FPS cap)
                        let time_since_last_render = now.duration_since(application.last_render_time);
                        if time_since_last_render >= application.target_frame_duration {
                            // Render frame only - no simulation
                            if let Err(e) = application.render() {
                                eprintln!("Error rendering frame: {}", e);
                            }

                            // Update cursor icon based on drag handle
                            if let Some(cursor_hint) = application.get_cursor_hint() {
                                let cursor = match cursor_hint {
                                    "ew-resize" => CursorIcon::ColResize,
                                    "ns-resize" => CursorIcon::RowResize,
                                    "nwse-resize" => CursorIcon::NwseResize,
                                    "nesw-resize" => CursorIcon::NeswResize,
                                    _ => CursorIcon::Default,
                                };
                                window.set_cursor(cursor);
                            } else {
                                window.set_cursor(CursorIcon::Default);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(ref application) = self.application {
            let now = std::time::Instant::now();
            
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
        let mut environment = Environment::new(initial_bounds, NUM_LIFEFORMS);

        // Initialize camera (must be before buffers since we need camera data for uniforms)
        let camera = Camera::new(
            Vec2::new(size.width as f32, size.height as f32),
            Some(initial_bounds),
        );

        // Get initial cells and lifeforms from environment
        let (cells, lifeforms) = environment.initialize_genomes(NUM_LIFEFORMS);

        // Initialize GPU buffers with initial data
        let bounds = environment.get_bounds();
        let camera_pos = camera.get_position();
        let zoom = camera.get_zoom();
        let view_size = Vec2::new(size.width as f32, size.height as f32);
        
        let buffers = Arc::new(GpuBuffers::new(
            &gpu.device,
            bytemuck::cast_slice(&cells),
            bytemuck::cast_slice(&lifeforms),
            bytemuck::cast_slice(&[Uniforms::zeroed()]),
            None, // GRNs will be uploaded separately for compute kernels
        ));
        
        let initial_uniforms = Uniforms::new(
            0.0,
            [camera_pos.x, camera_pos.y],
            zoom,
            2.0, // point_radius - match physics collision radius
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
            buffers.cell_capacity() as u32,
            environment.num_lifeforms() as u32,
        );
        
        buffers.update_uniforms(&gpu.queue, bytemuck::cast_slice(&[initial_uniforms]));


        let render_pipelines = RenderPipelines::new(
            &gpu.device,
            &gpu.config,
            buffers.cell_buffer(),
            &buffers.uniform_buffer,
            buffers.cell_free_list_buffer(),
        );

        // Initialize timestamp buffers

        // Initialize bounds renderer
        let bounds_renderer = BoundsRenderer::new(&gpu.device, &gpu.config);

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
                &["assets/ui/tailwind.css", "assets/ui/simulation.css"],
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

        // Initialize bounds border
        let bounds_border = BoundsBorder::new(initial_bounds);
        
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
            bounds_border,
        )
    }
}

