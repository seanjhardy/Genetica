// Renderer module - handles GPU rendering and compute passes

use wgpu;
use puffin::profile_scope;

use crate::modules::math::Vec2;
use crate::gpu::device::GpuDevice;
use crate::gpu::buffers::{GpuBuffers, TimestampBuffers};
use crate::gpu::pipelines::{ComputePipelines, RenderPipelines};
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::gpu::text_renderer::TextRenderer;
use crate::ui::{UiRenderer, UIManager};

/// Renderer handles all GPU rendering operations
pub struct Renderer;

impl Renderer {
    /// Render a frame with compute passes and point rendering
    pub fn render(
        gpu: &mut GpuDevice,
        compute_pipelines: &ComputePipelines,
        render_pipelines: &RenderPipelines,
        _buffers: &GpuBuffers,
        timestamps: &TimestampBuffers,
        bounds_renderer: &mut BoundsRenderer,
        text_renderer: &mut TextRenderer,
        ui_renderer: &mut UiRenderer,
        ui_manager: &mut UIManager,
        bounds_corners: [Vec2; 4],
        camera_pos: Vec2,
        zoom: f32,
        num_points: usize,
        frame_count: &mut u32,
        _last_profile_print: &mut std::time::Instant,
        simulation_steps: u32,  // Add this parameter
    ) -> Result<(), wgpu::SurfaceError> {
        profile_scope!("Render Frame");
        
        let output = {
            profile_scope!("get_current_texture");
            gpu.surface.get_current_texture()?
        };
        
        let view = {
            profile_scope!("create_view");
            output.texture.create_view(&wgpu::TextureViewDescriptor::default())
        };

        // Create command encoder for both compute and render
        // NOTE: We create a new encoder per frame to ensure proper ordering
        // wgpu queues automatically serialize operations, so compute and render
        // work submitted to the same queue will execute in submission order
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Command Encoder"),
        });
        
        // CRITICAL: Copy write buffer to read buffer before rendering
        // This synchronizes compute results (in write buffer) to read buffer (for rendering)
        // Compute always uses write buffer, render always uses read buffer
        // This ensures render sees the latest compute results
        let cell_size = _buffers.cell_size();
        if cell_size > 0 {
            let cell_size_bytes = (cell_size * std::mem::size_of::<crate::gpu::structures::Cell>()) as u64;
            encoder.copy_buffer_to_buffer(
                _buffers.cell_buffer_write(),
                0,
                _buffers.cell_buffer_read(),
                0,
                cell_size_bytes,
            );
        }
        
        // Update bounds renderer
        {
            profile_scope!("Update Bounds Renderer");
            let view_size = Vec2::new(gpu.config.width as f32, gpu.config.height as f32);
            bounds_renderer.update_bounds(
                &gpu.queue,
                bounds_corners,
                camera_pos,
                zoom,
                view_size.x,
                view_size.y,
                [0.0, 1.0, 0.0, 1.0], // Green color
            );
        }

        // Run multiple simulation steps (compute shader dispatches)
        // Only run if simulation_steps > 0 (0 means simulation already ran in loop)
        if simulation_steps > 0 {
            for _step in 0..simulation_steps {
                profile_scope!("Compute Pass (Update)");
                let timestamp_writes = if _step == simulation_steps - 1 {  // Only timestamp last step
                    timestamps.compute_timestamp_set.as_ref().map(|ts| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: ts,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }
                    })
                } else {
                    None
                };

                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update Compute Pass"),
                    timestamp_writes,
                });

                compute_pass.set_pipeline(&compute_pipelines.update);
                compute_pass.set_bind_group(0, &compute_pipelines.compute_bind_group, &[]);
                // Dispatch for all cells up to capacity
                compute_pass.dispatch_workgroups((num_points as u32 + 63) / 64, 1, 1);
            }
        }

        // Render UI backgrounds FIRST (before simulation, so simulation renders on top)
        {
            profile_scope!("Render UI Backgrounds");
            eprintln!("Simulation Render: About to call ui_renderer.render_backgrounds");
            if let Some(screen) = ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    ui_renderer.render_backgrounds(element, &gpu.device, &gpu.queue, &mut encoder, &view);
                }
            }
            eprintln!("Simulation Render: ui_renderer.render_backgrounds completed");
        }

        // Render points (simulation cells)
        {
            profile_scope!("Render Pass");
            let timestamp_writes = timestamps.render_timestamp_set.as_ref().map(|ts| {
                wgpu::RenderPassTimestampWrites {
                    query_set: ts,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });

            eprintln!("Simulation Render: Starting render pass with {} points", num_points);
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Load existing content (UI backgrounds)
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes,
                ..Default::default()
            });

            render_pass.set_pipeline(&render_pipelines.points);
            // Note: We use the bind group from render_pipelines, which should be updated
            // to point to the current read buffer (done in Application::render before calling this)
            render_pass.set_bind_group(0, &render_pipelines.render_bind_group, &[]);
            // Render all cells up to capacity (shader will skip free cells)
            render_pass.draw(0..4, 0..(num_points as u32)); // 4 vertices per quad, num_points instances
            eprintln!("Simulation Render: Draw call completed");
        }

        // Render bounds border
        {
            profile_scope!("Render Bounds Border");
            bounds_renderer.render(&mut encoder, &view);
        }
        
        // Draw text (wgpu_glyph draws directly into the encoder)
        {
            profile_scope!("Draw Text");
            text_renderer.draw(&gpu.device, &gpu.queue, &mut encoder, &view);
        }

        // Render UI overlays (borders, shadows, etc.) and text on top
        {
            profile_scope!("Render UI Overlays");
            eprintln!("Simulation Render: About to call ui_renderer.render_overlays");
            if let Some(screen) = ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    ui_renderer.render_overlays(element, &gpu.device, &gpu.queue, &mut encoder, &view);
                }
            }
            eprintln!("Simulation Render: ui_renderer.render_overlays completed");
        }

        // Handle timestamp queries if supported
        if let (Some(compute_ts), Some(render_ts), Some(ts_buffer)) = (
            &timestamps.compute_timestamp_set,
            &timestamps.render_timestamp_set,
            &timestamps.timestamp_buffer,
        ) {
            // Resolve timestamp queries to buffer (offsets must be aligned to 256 bytes)
            let query_align = 256u64;

            let compute_offset = 0;
            let render_offset = query_align;

            encoder.resolve_query_set(compute_ts, 0..2, ts_buffer, compute_offset);
            encoder.resolve_query_set(render_ts, 0..2, ts_buffer, render_offset);
        }

        // Submit and present (non-blocking)
        {
            profile_scope!("Submit Commands");
            eprintln!("Simulation Render: Submitting command buffer");
            let command_buffer = encoder.finish();
            gpu.queue.submit(std::iter::once(command_buffer));
            eprintln!("Simulation Render: Command buffer submitted");
        }

        {
            profile_scope!("Present Frame");
            eprintln!("Simulation Render: Presenting frame");
            output.present();
            eprintln!("Simulation Render: Frame presented");
        }

        // Update frame count (timestamp reading can happen asynchronously later if needed)
        *frame_count += 1;

        Ok(())
    }
}

