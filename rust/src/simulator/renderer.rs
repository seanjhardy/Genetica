// Renderer module - handles GPU rendering and compute passes

use wgpu;
use puffin::profile_scope;

use crate::modules::math::Vec2;
use crate::modules::ui::TextOverlay;
use crate::gpu::device::GpuDevice;
use crate::gpu::buffers::{GpuBuffers, TimestampBuffers};
use crate::gpu::pipelines::{ComputePipelines, RenderPipelines};
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::gpu::text_renderer::TextRenderer;

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
        bounds_corners: [Vec2; 4],
        camera_pos: Vec2,
        zoom: f32,
        num_points: usize,
        text_overlays: &[TextOverlay],
        frame_count: &mut u32,
        _last_profile_print: &mut std::time::Instant,
    ) -> Result<(), wgpu::SurfaceError> {
        profile_scope!("Render Frame");
        let output = gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder for both compute and render
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        
        // Update bounds renderer
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

        // Run compute shader for verlet integration with GPU timestamps
        {
            profile_scope!("Compute Pass (Verlet)");
            let timestamp_writes = timestamps.compute_timestamp_set.as_ref().map(|ts| {
                wgpu::ComputePassTimestampWrites {
                    query_set: ts,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Verlet Compute Pass"),
                timestamp_writes,
            });

            compute_pass.set_pipeline(&compute_pipelines.verlet);
            compute_pass.set_bind_group(0, &compute_pipelines.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups((num_points as u32 + 63) / 64, 1, 1);
        }

        // Run collision detection compute shader with GPU timestamps
        {
            profile_scope!("Collision Pass");
            let timestamp_writes = timestamps.collision_timestamp_set.as_ref().map(|ts| {
                wgpu::ComputePassTimestampWrites {
                    query_set: ts,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });

            let mut collision_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Collision Compute Pass"),
                timestamp_writes,
            });

            collision_pass.set_pipeline(&compute_pipelines.collision);
            collision_pass.set_bind_group(0, &compute_pipelines.collision_bind_group, &[]);
            collision_pass.dispatch_workgroups((num_points as u32 + 63) / 64, 1, 1);
        }

        // Render points
        {
            profile_scope!("Render Pass");
            let timestamp_writes = timestamps.render_timestamp_set.as_ref().map(|ts| {
                wgpu::RenderPassTimestampWrites {
                    query_set: ts,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }
            });

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes,
                ..Default::default()
            });

            render_pass.set_pipeline(&render_pipelines.points);
            render_pass.set_bind_group(0, &render_pipelines.render_bind_group, &[]);
            render_pass.draw(0..4, 0..(num_points as u32)); // 4 vertices per quad, num_points instances
        }

        // Render bounds border (must be before finishing encoder)
        bounds_renderer.render(&mut encoder, &view);
        
        // Queue and render text overlays
        for overlay in text_overlays {
            text_renderer.queue_text(&gpu.queue, &overlay.text, overlay.position.x, overlay.position.y, overlay.color);
        }
        text_renderer.draw(&gpu.device, &gpu.queue, &mut encoder, &view);

        // Handle timestamp queries if supported
        if let (Some(compute_ts), Some(collision_ts), Some(render_ts), Some(ts_buffer)) = (
            &timestamps.compute_timestamp_set,
            &timestamps.collision_timestamp_set,
            &timestamps.render_timestamp_set,
            &timestamps.timestamp_buffer,
        ) {
            // Resolve timestamp queries to buffer (offsets must be aligned to 256 bytes)
            let query_align = 256u64;

            let compute_offset = 0;
            let collision_offset = query_align;
            let render_offset = query_align * 2;

            encoder.resolve_query_set(compute_ts, 0..2, ts_buffer, compute_offset);
            encoder.resolve_query_set(collision_ts, 0..2, ts_buffer, collision_offset);
            encoder.resolve_query_set(render_ts, 0..2, ts_buffer, render_offset);
        }

        // Submit and present (non-blocking)
        let command_buffer = encoder.finish();
        gpu.queue.submit(std::iter::once(command_buffer));
        output.present();

        // Update frame count (timestamp reading can happen asynchronously later if needed)
        *frame_count += 1;

        Ok(())
    }
}

