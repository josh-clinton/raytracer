// main.cpp - builds a scene and renders it.
//
// The scene is the classic "random spheres" layout: a large ground sphere,
// a field of small spheres with randomly chosen Lambertian/metal/dielectric
// materials, and three signature large spheres (glass, matte, metal) in
// front - good coverage of every material path in one image.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <random>
#include <thread>

#include "bvh.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "vec3.h"

hittable_list build_scene(std::mt19937& rng) {
    hittable_list world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_mat = random_double(rng);
            point3 center(a + 0.9 * random_double(rng), 0.2, b + 0.9 * random_double(rng));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    color albedo = color::random(rng) * color::random(rng);
                    sphere_material = std::make_shared<lambertian>(albedo);
                } else if (choose_mat < 0.95) {
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
    // O(N) - with ~480 spheres here, this is the difference between a
    // render that takes seconds and one that takes many minutes.
    return hittable_list(std::make_shared<bvh_node>(world));
}

// Builds the same random-sphere scene but as a flat hittable_list, with no
// BVH wrapping - used by --benchmark to demonstrate the O(N) vs O(log N)
// difference the acceleration structure makes on intersection cost.
hittable_list build_scene_flat(std::mt19937& rng) {
    std::mt19937 rng2 = rng;  // same seed sequence as build_scene's inner loop
    hittable_list world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            double choose_mat = random_double(rng2);
            point3 center(a + 0.9 * random_double(rng2), 0.2, b + 0.9 * random_double(rng2));
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;
                if (choose_mat < 0.8) {
                    color albedo = color::random(rng2) * color::random(rng2);
                    sphere_material = std::make_shared<lambertian>(albedo);
                } else if (choose_mat < 0.95) {
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

void run_benchmark() {
    // Small, fixed settings so the comparison is fast and repeatable; what
    // matters is the ratio between the two times, not the absolute numbers.
    const int width = 300, spp = 32, depth = 8;
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = width;
    cam.samples_per_pixel = spp;
    cam.max_depth = depth;
    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    std::mt19937 rng_flat(12345);
    hittable_list flat_world = build_scene_flat(rng_flat);
    std::fprintf(stderr, "[no BVH] %d objects, linear scan per ray\n",
                 static_cast<int>(flat_world.objects.size()));
    auto t0 = std::chrono::high_resolution_clock::now();
    cam.render(flat_world, num_threads);
    auto t1 = std::chrono::high_resolution_clock::now();
    double flat_seconds = std::chrono::duration<double>(t1 - t0).count();

    std::mt19937 rng_bvh(12345);
    hittable_list bvh_world = build_scene(rng_bvh);
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

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--benchmark") {
        run_benchmark();
        return 0;
    }

    // Defaults tuned for a fast-but-representative render; override via CLI
    // for a quick preview or a higher-quality final image, e.g.:
    //   ./raytracer 1280 200 12   (width, samples/pixel, max_depth)
    int image_width = argc > 1 ? std::atoi(argv[1]) : 800;
    int samples_per_pixel = argc > 2 ? std::atoi(argv[2]) : 100;
    int max_depth = argc > 3 ? std::atoi(argv[3]) : 12;
    std::string out_path = argc > 4 ? argv[4] : "render.ppm";

    std::mt19937 rng(12345);  // fixed seed -> reproducible scene layout
    hittable_list world = build_scene(rng);

    camera cam;
    cam.aspect_ratio = 16.0 / 9.0;
    cam.image_width = image_width;
    cam.samples_per_pixel = samples_per_pixel;
    cam.max_depth = max_depth;

    cam.vfov = 20;
    cam.lookfrom = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);

    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
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
