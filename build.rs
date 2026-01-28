fn main() {
    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        println!("cargo:rustc-env=OUT_DIR={}", out_dir);
    }
    let compiler = wesl::Wesl::new("src/gpu/wgsl");
    let shaders = [
        ("package::kernels::cells", "kernels_cells"),
        ("package::kernels::links", "kernels_links"),
        ("package::kernels::nutrients", "kernels_nutrients"),
        ("package::kernels::spawn_cells", "kernels_spawn_cells"),
        ("package::kernels::points", "kernels_points"),
        ("package::kernels::pick_cell", "kernels_pick_cell"),
        ("package::kernels::grn", "kernels_grn"),
        ("package::shaders::cells", "shaders_cells"),
        ("package::shaders::image_texture", "shaders_image_texture"),
        ("package::shaders::links", "shaders_links"),
        ("package::shaders::nutrients", "shaders_nutrients"),
        ("package::shaders::perlin", "shaders_perlin"),
        ("package::shaders::env_texture", "shaders_env_texture"),
        ("package::shaders::terrain_height", "shaders_terrain_height"),
        ("package::shaders::terrain_rock_noise", "shaders_terrain_rock_noise"),
        ("package::shaders::terrain_shadow", "shaders_terrain_shadow"),
        ("package::shaders::terrain_composite", "shaders_terrain_composite"),
        ("package::shaders::caustics", "shaders_caustics"),
        ("package::shaders::caustics_blit", "shaders_caustics_blit"),
        ("package::shaders::caustics_composite", "shaders_caustics_composite"),
        ("package::shaders::terrain_caustics_composite", "shaders_terrain_caustics_composite"),
        ("package::shaders::post_processing", "shaders_post_processing"),
        ("package::shaders::text", "shaders_text"),
        ("package::shaders::ui_rect", "shaders_ui_rect"),
        ("package::shaders::perlin_noise_texture", "shaders_perlin_noise_texture"),
        ("package::shaders::points", "shaders_points"),
    ];

    for (root, artifact_name) in shaders {
        compiler.build_artifact(&root.parse().unwrap(), artifact_name);
    }
}
