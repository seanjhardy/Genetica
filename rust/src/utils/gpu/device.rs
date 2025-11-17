// GPU device module - manages GPU initialization and surface

use wgpu;
use winit::window::Window;

/// GPU device, queue, and surface management
pub struct GpuDevice {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuDevice {
    pub async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // Create surface with 'static lifetime - surface owns its target internally
        let surface: wgpu::Surface<'static> = unsafe {
            std::mem::transmute(instance.create_surface(window).unwrap())
        };

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // Enable timestamp queries for GPU profiling
        let mut required_features = wgpu::Features::empty();

        let required_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features,
                    required_limits,
                    memory_hints: Default::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        // Choose a present mode that doesn't block on acquire_texture
        // Prefer Mailbox (triple buffering - smooth but no blocking), then Immediate (no VSync)
        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|&mode| mode == wgpu::PresentMode::Fifo)
            .or_else(|| {
                surface_caps
                    .present_modes
                    .iter()
                    .copied()
                    .find(|&mode| mode == wgpu::PresentMode::Mailbox)
            })
            .or_else(|| {
                surface_caps
                    .present_modes
                    .iter()
                    .copied()
                    .find(|&mode| mode == wgpu::PresentMode::Immediate)
            })
            .unwrap_or(surface_caps.present_modes[0]); // Fall back to first available if none exist

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode,  // Use the selected mode instead of surface_caps.present_modes[0]
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}

