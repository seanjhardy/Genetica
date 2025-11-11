// Renderer module - handles simulation-specific rendering (cells and bounds to viewports)

use wgpu;
use puffin::profile_scope;

use crate::utils::math::Vec2;
use crate::utils::gpu::device::GpuDevice;
use crate::gpu::buffers::GpuBuffers;
use crate::gpu::pipelines::RenderPipelines;
use crate::gpu::bounds_renderer::BoundsRenderer;
use crate::ui::UiRenderer;

/// Renderer handles simulation rendering (cells and bounds) to viewport textures
pub struct Renderer;

impl Renderer {
    /// Render simulation cells and bounds to viewport textures
    /// Returns whether a viewport was found and rendered to
    pub fn render_simulation(
        gpu: &mut GpuDevice,
        buffers: &GpuBuffers,
        render_pipelines: &RenderPipelines,
        bounds_renderer: &mut BoundsRenderer,
        environment: &mut crate::simulator::environment::Environment,
        ui_renderer: &mut UiRenderer,
        ui_manager: &mut crate::ui::UIManager,
        bounds_corners: [Vec2; 4],
        camera_pos: Vec2,
        zoom: f32,
        num_points: usize,
        show_grid: bool,
        encoder: &mut wgpu::CommandEncoder,
    ) -> bool {
        profile_scope!("Render Simulation");
        
        // CRITICAL: Copy write buffer to read buffer before rendering
        // This synchronizes compute results (in write buffer) to read buffer (for rendering)
        let cell_capacity = buffers.cell_capacity();
        if cell_capacity > 0 {
            let cell_size_bytes = (cell_capacity * std::mem::size_of::<crate::gpu::structures::Cell>()) as u64;
            encoder.copy_buffer_to_buffer(
                buffers.cell_buffer_write(),
                0,
                buffers.cell_buffer_read(),
                0,
                cell_size_bytes,
            );

            let free_list_bytes = ((buffers.cell_capacity() + 1) * std::mem::size_of::<u32>()) as u64;
            encoder.copy_buffer_to_buffer(
                buffers.cell_free_list_buffer_write(),
                0,
                buffers.cell_free_list_buffer_read(),
                0,
                free_list_bytes,
            );
        }

        // Prepare viewport textures (compute layout and create textures)
        let mut viewport_texture_view = None;
        {
            profile_scope!("Prepare Viewports");
            if let Some(screen) = ui_manager.get_screen("simulation") {
                for element in screen.get_elements_mut() {
                    ui_renderer.compute_layout(element);
                    ui_renderer.ensure_viewport_textures(element, &gpu.device);
                    
                    // Get the simulation viewport texture from the component
                    viewport_texture_view = ui_renderer.get_viewport_texture_view(element, "simulation");
                }
            }
        }

        let viewport_texture_view = match viewport_texture_view {
            Some(view) => view,
            None => {
                return false; // No viewport exists, don't render simulation
            }
        };

        // Update bounds renderer with viewport dimensions
        {
            profile_scope!("Update Bounds");
            let bounds = environment.get_bounds();
            // Get viewport dimensions from the viewport texture
            // For now, use screen dimensions - we can get actual viewport size later if needed
            let view_size = Vec2::new(gpu.config.width as f32, gpu.config.height as f32);
            bounds_renderer.update_bounds(
                &gpu.queue,
                bounds_corners,
                bounds,
                camera_pos,
                zoom,
                view_size.x,
                view_size.y,
                [1.0, 1.0, 1.0, 1.0], // White border
            );
        }

        // Update planet texture if bounds changed
        {
            profile_scope!("Update Planet");
            environment.planet_mut().update(&gpu.device, &gpu.queue, gpu.config.format);
        }
        
        // Render planet background and bounds border to viewport texture
        {
            profile_scope!("Render Planet Background & Bounds");
            let planet_texture_view = environment.planet().texture_view();
            
            bounds_renderer.render(
                encoder,
                &gpu.device,
                &gpu.queue,
                viewport_texture_view,
                planet_texture_view,
            );
        }

        if show_grid {
            profile_scope!("Render Nutrient Overlay");
            let (grid_w, grid_h) = buffers.nutrient_grid_dimensions();
            let grid_cells = grid_w.saturating_mul(grid_h);
            if grid_cells > 0 {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Nutrient Overlay Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: viewport_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    ..Default::default()
                });
                render_pass.set_pipeline(&render_pipelines.nutrient_overlay);
                render_pass.set_bind_group(0, &render_pipelines.nutrient_bind_group, &[]);
                render_pass.draw(0..4, 0..grid_cells);
            }
        }
        
        // Render simulation cells to viewport texture (on top of planet background)
        {
            profile_scope!("Render Cells");
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Simulation Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: viewport_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,  // Load existing planet background
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                ..Default::default()
            });

            render_pass.set_pipeline(&render_pipelines.points);
            render_pass.set_bind_group(0, &render_pipelines.render_bind_group, &[]);
            render_pass.draw(0..4, 0..(num_points as u32)); // 4 vertices per quad, num_points instances
        }

        true
    }
}