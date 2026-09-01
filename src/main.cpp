// main.cpp - builds a scene and renders it.
//
// The scene is the classic "random spheres" layout: a large ground sphere,
// a field of small spheres with randomly chosen Lambertian/metal/dielectric
// materials, and three signature large spheres (glass, matte, metal) in
// front - good coverage of every material path in one image.
//
// Beyond the default single render, this file also drives a parameterized
// benchmark sweep (--bench / --scene, orchestrated by benchmarks/sweep.py):
// the same scene generator and camera setup can be pushed along four axes
// - image size, object count, camera orbit angle, and material mix - to see
// how the BVH's speedup over a flat linear scan responds to each one.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>

#include "bvh.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "vec3.h"

namespace {
constexpr double PI = 3.14159265358979323846;
}

// grid_radius controls object count: the field loop runs
// [-grid_radius, grid_radius) on both axes, so candidate count is roughly
// (2*grid_radius)^2 before the "too close to the signature spheres" ones are
// skipped. diffuse_frac/metal_frac select the material mix (glass gets
// whatever's left of 1.0); defaults reproduce the original fixed 80/15/5 mix
// exactly, so every existing call site keeps behaving as it always did.
hittable_list build_scene(std::mt19937& rng, int grid_radius = 11, double diffuse_frac = 0.8,
                           double metal_frac = 0.15) {
    hittable_list world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -grid_radius; a < grid_radius; a++) {
        for (int b = -grid_radius; b < grid_radius; b++) {
            double choose_mat = random_double(rng);
            point3 center(a + 0.9 * random_double(rng), 0.2, b + 0.9 * random_double(rng));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < diffuse_frac) {
                    // diffuse
                    color albedo = color::random(rng) * color::random(rng);
                    sphere_material = std::make_shared<lambertian>(albedo);
                } else if (choose_mat < diffuse_frac + metal_frac) {
                    // metal
                    color albedo = color::random(rng, 0.5, 1);
                    double fuzz = random_double(rng, 0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                } else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                }
                world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    // Wrap the flat list in a BVH so intersection is O(log N) instead of
    // O(N) - with hundreds of spheres here, this is the difference between a
    // render that takes seconds and one that takes many minutes.
    return hittable_list(std::make_shared<bvh_node>(world));
}

// Builds the same random-sphere scene but as a flat hittable_list, with no
// BVH wrapping - used by the benchmark modes to demonstrate the O(N) vs
// O(log N) difference the acceleration structure makes on intersection cost.
hittable_list build_scene_flat(std::mt19937& rng, int grid_radius = 11, double diffuse_frac = 0.8,
                                double metal_frac = 0.15) {
    std::mt19937 rng2 = rng;  // same seed sequence as build_scene's inner loop
    hittable_list world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -grid_radius; a < grid_radius; a++) {
        for (int b = -grid_radius; b < grid_radius; b++) {
            double choose_mat = random_double(rng2);
            point3 center(a + 0.9 * random_double(rng2), 0.2, b + 0.9 * random_double(rng2));
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;
                if (choose_mat < diffuse_frac) {
                    color albedo = color::random(rng2) * color::random(rng2);
                    sphere_material = std::make_shared<lambertian>(albedo);
                } else if (choose_mat < diffuse_frac + metal_frac) {
                    color albedo = color::random(rng2, 0.5, 1);
                    double fuzz = random_double(rng2, 0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                } else {
                    sphere_material = std::make_shared<dielectric>(1.5);
                }
                world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));
    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;  // NOT wrapped in a bvh_node
}

// Everything that describes "which scene, seen from where" - the knobs the
// sweep script varies one at a time while holding the rest at their default.
struct scene_config {
    int grid_radius = 11;        // -> roughly (2*grid_radius)^2 candidate objects
    double diffuse_frac = 0.8;   // fraction of field spheres that are Lambertian
    double metal_frac = 0.15;    //   ...that are metal (glass gets the remainder)
    double vfov = 20;            // vertical field of view, degrees
    double cam_angle_deg = 13.0; // orbit angle around the scene; ~13 deg reproduces
                                  // the project's original fixed camera position
    unsigned int seed = 12345;
};

// Same orbit radius/height as the original fixed camera (13,2,3), just
// parameterized by angle: lookfrom = (R cos theta, 2, R sin theta).
camera make_camera(int image_width, int spp, int max_depth, const scene_config& cfg) {
    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = spp;
    cam.max_depth = max_depth;
    cam.vfov = cfg.vfov;

    const double R = 13.3417;  // sqrt(13^2 + 3^2), matches the original camera's distance
    double theta = cfg.cam_angle_deg * PI / 180.0;
    cam.lookfrom = point3(R * std::cos(theta), 2, R * std::sin(theta));
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;
    return cam;
}

void run_benchmark() {
    // Small, fixed settings so the comparison is fast and repeatable; what
    // matters is the ratio between the two times, not the absolute numbers.
    const int width = 300, spp = 32, depth = 8;
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    scene_config cfg;  // all defaults - the original fixed scene/camera

    camera cam = make_camera(width, spp, depth, cfg);

    std::mt19937 rng_flat(cfg.seed);
    hittable_list flat_world = build_scene_flat(rng_flat, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    std::fprintf(stderr, "[no BVH] %d objects, linear scan per ray\n",
                 static_cast<int>(flat_world.objects.size()));
    auto t0 = std::chrono::high_resolution_clock::now();
    cam.render(flat_world, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double flat_seconds = std::chrono::duration<double>(t1 - t0).count();

    std::mt19937 rng_bvh(cfg.seed);
    hittable_list bvh_world = build_scene(rng_bvh, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    std::fprintf(stderr, "[BVH]    same scene, wrapped in a bounding volume hierarchy\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    cam.render(bvh_world, num_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bvh_seconds = std::chrono::duration<double>(t3 - t2).count();

    std::fprintf(stderr,
                  "\nBenchmark (%dx%d, %d spp, depth %d, %u threads):\n"
                  "  no BVH : %.2fs\n"
                  "  BVH    : %.2fs\n"
                  "  speedup: %.2fx\n",
                  width, static_cast<int>(width / cam.aspect_ratio), spp, depth, num_threads,
                  flat_seconds, bvh_seconds, flat_seconds / bvh_seconds);
}

// One (flat-vs-BVH) timing sample for a fully-specified config, printed as a
// single CSV row to stdout so a driver script can invoke this binary many
// times - once per (dimension, value, repeat) - and build up a results table
// without any sweep-looping logic living in C++. Column order matches the
// header benchmarks/sweep.py writes.
void run_bench(int width, int spp, int depth, const scene_config& cfg, unsigned int num_threads) {
    std::mt19937 rng_flat(cfg.seed);
    hittable_list flat_world = build_scene_flat(rng_flat, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    int object_count = static_cast<int>(flat_world.objects.size());

    camera cam_flat = make_camera(width, spp, depth, cfg);
    auto t0 = std::chrono::high_resolution_clock::now();
    cam_flat.render(flat_world, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double flat_seconds = std::chrono::duration<double>(t1 - t0).count();

    std::mt19937 rng_bvh(cfg.seed);
    hittable_list bvh_world = build_scene(rng_bvh, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    camera cam_bvh = make_camera(width, spp, depth, cfg);
    auto t2 = std::chrono::high_resolution_clock::now();
    cam_bvh.render(bvh_world, num_threads);
    auto t3 = std::chrono::high_resolution_clock::now();
    double bvh_seconds = std::chrono::duration<double>(t3 - t2).count();

    int height = static_cast<int>(width / cam_flat.aspect_ratio);
    double glass_frac = 1.0 - cfg.diffuse_frac - cfg.metal_frac;

    std::printf("%d,%d,%d,%d,%d,%d,%.3f,%.2f,%.3f,%.3f,%.3f,%u,%.6f,%.6f,%.4f\n", width, height,
                spp, depth, cfg.grid_radius, object_count, cfg.vfov, cfg.cam_angle_deg,
                cfg.diffuse_frac, cfg.metal_frac, glass_frac, cfg.seed, flat_seconds, bvh_seconds,
                flat_seconds / bvh_seconds);
}

// Renders one actual image (BVH-accelerated, like a normal render) for a
// fully-specified config - used by the sweep script to grab a handful of
// representative preview images per dimension, not just timing numbers.
void run_scene_render(int width, int spp, int depth, const scene_config& cfg,
                       const std::string& out_path, unsigned int num_threads) {
    std::mt19937 rng(cfg.seed);
    hittable_list world = build_scene(rng, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    camera cam = make_camera(width, spp, depth, cfg);

    std::vector<uint8_t> pixels = cam.render(world, num_threads);
    int height = cam.height();

    std::ofstream out(out_path, std::ios::binary);
    out << "P6\n" << width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
}

int main(int argc, char** argv) {
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());

    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        run_benchmark();
        return 0;
    }

    // --bench width spp depth grid_radius vfov cam_angle_deg diffuse_pct metal_pct seed
    // Prints one CSV row to stdout (see run_bench's column order above).
    // diffuse_pct/metal_pct are 0-100 ints; glass gets whatever's left.
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        if (argc < 11) {
            std::fprintf(stderr,
                          "usage: %s --bench width spp depth grid_radius vfov cam_angle_deg "
                          "diffuse_pct metal_pct seed\n",
                          argv[0]);
            return 1;
        }
        int width = std::atoi(argv[2]);
        int spp = std::atoi(argv[3]);
        int depth = std::atoi(argv[4]);
        scene_config cfg;
        cfg.grid_radius = std::atoi(argv[5]);
        cfg.vfov = std::atof(argv[6]);
        cfg.cam_angle_deg = std::atof(argv[7]);
        cfg.diffuse_frac = std::atof(argv[8]) / 100.0;
        cfg.metal_frac = std::atof(argv[9]) / 100.0;
        cfg.seed = static_cast<unsigned int>(std::strtoul(argv[10], nullptr, 10));
        run_bench(width, spp, depth, cfg, num_threads);
        return 0;
    }

    // --scene width spp depth grid_radius vfov cam_angle_deg diffuse_pct metal_pct seed out.ppm
    if (argc > 1 && std::string(argv[1]) == "--scene") {
        if (argc < 12) {
            std::fprintf(stderr,
                          "usage: %s --scene width spp depth grid_radius vfov cam_angle_deg "
                          "diffuse_pct metal_pct seed out.ppm\n",
                          argv[0]);
            return 1;
        }
        int width = std::atoi(argv[2]);
        int spp = std::atoi(argv[3]);
        int depth = std::atoi(argv[4]);
        scene_config cfg;
        cfg.grid_radius = std::atoi(argv[5]);
        cfg.vfov = std::atof(argv[6]);
        cfg.cam_angle_deg = std::atof(argv[7]);
        cfg.diffuse_frac = std::atof(argv[8]) / 100.0;
        cfg.metal_frac = std::atof(argv[9]) / 100.0;
        cfg.seed = static_cast<unsigned int>(std::strtoul(argv[10], nullptr, 10));
        std::string out_path = argv[11];

        auto t0 = std::chrono::high_resolution_clock::now();
        run_scene_render(width, spp, depth, cfg, out_path, num_threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "Wrote %s in %.2fs\n", out_path.c_str(),
                     std::chrono::duration<double>(t1 - t0).count());
        return 0;
    }

    // Defaults tuned for a fast-but-representative render; override via CLI
    // for a quick preview or a higher-quality final image, e.g.:
    //   ./raytracer 1280 200 12   (width, samples/pixel, max_depth)
    int image_width = argc > 1 ? std::atoi(argv[1]) : 800;
    int samples_per_pixel = argc > 2 ? std::atoi(argv[2]) : 100;
    int max_depth = argc > 3 ? std::atoi(argv[3]) : 12;
    std::string out_path = argc > 4 ? argv[4] : "render.ppm";

    scene_config cfg;  // defaults - the original fixed scene/camera
    std::mt19937 rng(cfg.seed);
    hittable_list world = build_scene(rng, cfg.grid_radius, cfg.diffuse_frac, cfg.metal_frac);
    camera cam = make_camera(image_width, samples_per_pixel, max_depth, cfg);

    std::fprintf(stderr, "Rendering %dx%d, %d spp, depth %d, %u threads...\n", image_width,
                 static_cast<int>(image_width / cam.aspect_ratio), samples_per_pixel, max_depth,
                 num_threads);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> pixels = cam.render(world, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();

    int height = cam.height();
    std::ofstream out(out_path, std::ios::binary);
    out << "P6\n" << image_width << ' ' << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    out.close();

    std::fprintf(stderr, "Wrote %s in %.2f seconds (%.1f Mray/s approx.)\n", out_path.c_str(),
                 seconds,
                 (static_cast<double>(image_width) * height * samples_per_pixel) / seconds / 1e6);
    return 0;
}
