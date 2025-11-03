// Simulator module - main simulation loop and window management

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId, CursorIcon},
};
use puffin::profile_function;

use crate::modules::math::{Rect, Vec2};
use crate::modules::camera::{Camera, KeyStates};
use crate::modules::ui::{UiState, BoundsBorder};
use crate::gpu::device::GpuDevice;
use crate::gpu::text_renderer::TextRenderer;
use crate::gpu::buffers::{GpuBuffers, TimestampBuffers};
use crate::gpu::pipelines::{ComputePipelines, RenderPipelines};
use crate::gpu::uniforms::Uniforms;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::simulator::environment::Environment;
use crate::simulator::renderer::Renderer;

const NUM_POINTS: usize = 5000;

/// Main simulator that manages window, camera, environment, and rendering
pub struct Simulator {
    app: App,
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            app: App {
                window: None,
                state: None,
            },
        }
    }

    pub fn run(mut self) {
        let event_loop = EventLoop::new().unwrap();
        event_loop.run_app(&mut self.app).unwrap();
    }
}

struct App {
    window: Option<Window>,
    state: Option<SimulatorState>,
}

struct SimulatorState {
    gpu: GpuDevice,
    compute_pipelines: ComputePipelines,
    render_pipelines: RenderPipelines,
    buffers: GpuBuffers,
    timestamps: TimestampBuffers,
    bounds_renderer: BoundsRenderer,
    text_renderer: TextRenderer<()>, // Type parameter - will be inferred at creation
    camera: Camera,
    environment: Environment,
    ui_state: UiState,
    bounds_border: BoundsBorder,
    key_states: KeyStates,
    last_cursor_pos: Vec2, // Track cursor position for mouse wheel events
    last_frame_time: std::time::Instant,
    frame_count: u32,
    last_profile_print: std::time::Instant,
    last_fps_update: std::time::Instant,
    fps_frames: u32,
    step_counter: u64,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Genetica Rust - Verlet Integration")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = event_loop.create_window(window_attributes).unwrap();

        let state = pollster::block_on(SimulatorState::new(&window));

        self.window = Some(window);
        self.state = Some(state);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                if let Some(ref mut state) = self.state {
                    state.resize(physical_size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let Some(ref mut state) = self.state {
                    if let PhysicalKey::Code(key_code) = event.physical_key {
                        state.handle_keyboard_input(
                            event.state == winit::event::ElementState::Pressed,
                            key_code,
                        );
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(ref mut state) = self.state {
                    let delta_scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 10.0,
                    };
                    // Use last known cursor position
                    state.handle_mouse_wheel(delta_scroll, state.last_cursor_pos);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(ref mut state) = self.state {
                    let pos = Vec2::new(position.x as f32, position.y as f32);
                    state.last_cursor_pos = pos;
                    state.handle_mouse_move(pos, false); // TODO: check UI hover
                }
            }
            WindowEvent::MouseInput { button, state: button_state, .. } => {
                if let Some(ref mut state) = self.state {
                    if button == winit::event::MouseButton::Left {
                        state.environment.handle_mouse_press(
                            button_state == winit::event::ElementState::Pressed,
                        );
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                puffin::GlobalProfiler::lock().new_frame();
                if let Some(ref mut state) = self.state {
                    if let Some(ref window) = self.window {
                        state.update(window);

                        // Display UI info in window title
                        let ui_text = format!("FPS: {:.1} | Step: {}", state.ui_state.framerate, state.ui_state.step);
                        window.set_title(&format!("Genetica Rust - {}", ui_text));

                        // Update cursor icon based on drag handle
                        if let Some(cursor_hint) = state.environment.get_cursor_hint() {
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

                        match state.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                if let Some(ref mut state) = self.state {
                                    state.resize(winit::dpi::PhysicalSize::new(
                                        state.gpu.config.width,
                                        state.gpu.config.height,
                                    ));
                                }
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                event_loop.exit();
                            }
                            Err(e) => eprintln!("{:?}", e),
                        }
                    }
                }
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

impl SimulatorState {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // Initialize GPU device
        let gpu = GpuDevice::new(window).await;

        // Initialize environment
        let initial_bounds = Rect::new(0.0, 0.0, size.width as f32, size.height as f32);
        let environment = Environment::new(initial_bounds, NUM_POINTS);

        // Initialize camera (must be before buffers since we need camera data for uniforms)
        let camera = Camera::new(
            Vec2::new(size.width as f32, size.height as f32),
            Some(initial_bounds),
        );

        // Initialize points
        let points = environment.initialize_points();
        
        // Initialize GPU buffers with initial uniforms
        let bounds = environment.get_bounds();
        let camera_pos = camera.get_position();
        let zoom = camera.get_zoom();
        let view_size = Vec2::new(size.width as f32, size.height as f32);
        
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
        );
        
        let buffers = GpuBuffers::new(
            &gpu.device,
            bytemuck::cast_slice(&points),
            bytemuck::cast_slice(&[initial_uniforms]),
        );

        // Initialize pipelines
        let compute_pipelines = ComputePipelines::new(
            &gpu.device,
            &buffers.point_buffer,
            &buffers.uniform_buffer,
        );

        let render_pipelines = RenderPipelines::new(
            &gpu.device,
            &gpu.config,
            &buffers.point_buffer,
            &buffers.uniform_buffer,
        );

        // Initialize timestamp buffers
        let timestamps = TimestampBuffers::new(&gpu.device);

        // Initialize bounds renderer
        let bounds_renderer = BoundsRenderer::new(&gpu.device, &gpu.config);
        
        // Initialize text renderer - create it inline so compiler can infer the type
        // The type will be TextRenderer<FontRef<'static>> where FontRef is private
        // but the compiler knows it and can satisfy trait bounds
        let font_data = include_bytes!("../../../assets/fonts/russoone-regular.ttf");
        let brush = wgpu_text::BrushBuilder::using_font_bytes(font_data)
            .unwrap()
            .build(
                &gpu.device,
                gpu.config.width,
                gpu.config.height,
                gpu.config.format,
            );
        // Create TextRenderer with inferred type - the brush's type is FontRef<'static>
        // but we can't name it, so we'll use unsafe to convert the type
        let text_renderer_typed = TextRenderer {
            brush,
            screen_width: gpu.config.width as f32,
            screen_height: gpu.config.height as f32,
        };
        // Convert to TextRenderer<()> for storage - unsafe but we know the types match
        let text_renderer: TextRenderer<()> = unsafe { std::mem::transmute(text_renderer_typed) };

        // Initialize UI state
        let ui_state = UiState::new();
        
        // Initialize bounds border
        let bounds_border = BoundsBorder::new(initial_bounds);

        Self {
            gpu,
            compute_pipelines,
            render_pipelines,
            buffers,
            timestamps,
            bounds_renderer,
            text_renderer,
            camera,
            environment,
            ui_state,
            bounds_border,
            key_states: KeyStates::default(),
            last_cursor_pos: Vec2::new(size.width as f32 / 2.0, size.height as f32 / 2.0),
            last_frame_time: std::time::Instant::now(),
            frame_count: 0,
            last_profile_print: std::time::Instant::now(),
            last_fps_update: std::time::Instant::now(),
            fps_frames: 0,
            step_counter: 0,
        }
    }

    fn update(&mut self, _window: &Window) {
        profile_function!();
        let now = std::time::Instant::now();
        let delta_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // Update framerate tracking
        self.fps_frames += 1;
        if now.duration_since(self.last_fps_update).as_secs_f32() >= 0.5 {
            let fps = self.fps_frames as f32 / now.duration_since(self.last_fps_update).as_secs_f32();
            self.ui_state.update(fps, self.step_counter);
            self.fps_frames = 0;
            self.last_fps_update = now;
        }

        // Update camera
        self.camera.update(delta_time, &self.key_states);

        // Update step counter
        self.step_counter += 1;

        // Update bounds border
        let bounds = self.environment.get_bounds();
        self.bounds_border.set_bounds(bounds);

        // Update uniforms with current bounds, camera position, and zoom
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        let view_size = Vec2::new(self.gpu.config.width as f32, self.gpu.config.height as f32);
        
        let uniforms = Uniforms::new(
            delta_time,
            [camera_pos.x, camera_pos.y],
            zoom,
            2.0, // point_radius - match physics collision radius
            bounds.left,
            bounds.top,
            bounds.right(),
            bounds.bottom(),
            view_size.x,
            view_size.y,
        );

        self.buffers
            .update_uniforms(&self.gpu.queue, bytemuck::cast_slice(&[uniforms]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        profile_function!();
        let bounds_corners = self.bounds_border.get_corners();
        let camera_pos = self.camera.get_position();
        let zoom = self.camera.get_zoom();
        
        Renderer::render(
            &mut self.gpu,
            &self.compute_pipelines,
            &self.render_pipelines,
            &self.buffers,
            &self.timestamps,
            &mut self.bounds_renderer,
            &mut self.text_renderer,
            bounds_corners,
            camera_pos,
            zoom,
            self.environment.num_points(),
            &self.ui_state.text_overlays,
            &mut self.frame_count,
            &mut self.last_profile_print,
        )
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.gpu.resize(new_size);

        // Update camera view size
        self.camera
            .set_view_size(Vec2::new(new_size.width as f32, new_size.height as f32));
        
        // Update text renderer
        self.text_renderer.resize(new_size, &self.gpu.device, &self.gpu.config, &self.gpu.queue);
    }

    fn handle_keyboard_input(&mut self, pressed: bool, key_code: KeyCode) {
        match key_code {
            KeyCode::KeyW => self.key_states.w = pressed,
            KeyCode::KeyA => self.key_states.a = pressed,
            KeyCode::KeyS => self.key_states.s = pressed,
            KeyCode::KeyD => self.key_states.d = pressed,
            _ => {}
        }
    }

    fn handle_mouse_wheel(&mut self, delta: f32, mouse_pos: Vec2) {
        self.camera.zoom(delta, mouse_pos);
    }

    fn handle_mouse_move(&mut self, mouse_pos: Vec2, ui_hovered: bool) {
        // Convert screen coordinates to world coordinates
        let world_pos = self.camera.screen_to_world(mouse_pos);

        // Update environment drag handler
        self.environment
            .update(world_pos, self.camera.get_zoom(), ui_hovered);
    }
}

