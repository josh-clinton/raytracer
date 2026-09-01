// camera.h - builds camera rays and drives the render loop.
//
// This is where several concepts from the JD come together in code:
//   - Monte Carlo integration: each pixel averages `samples_per_pixel`
//     independent random rays (antialiasing + soft shadows + glossy
//     reflection all fall out of this one averaging loop for free).
//   - Recursive ray tracing: ray_color() recurses on scattered rays up to
//     `max_depth`, accumulating attenuation - a direct implementation of
//     the rendering equation truncated to a finite number of bounces.
//   - Parallelism: rows are distributed across a thread pool, which is the
//     minimum viable version of the "profile, debug, and optimize graphics
//     workloads for performance" bullet in the JD.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

#include "color.h"
#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"

class camera {
public:
    double aspect_ratio = 1.0;
    int image_width = 400;
    int samples_per_pixel = 100;
    int max_depth = 10;

    double vfov = 90;              // vertical field-of-view, in degrees
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);

    double defocus_angle = 0;      // depth-of-field cone angle; 0 = pinhole camera
    double focus_dist = 10;

    // Renders the scene into a flat RGB8 buffer (image_width * image_height * 3
    // bytes) using `num_threads` worker threads, one row-range each.
    std::vector<uint8_t> render(const hittable& world, int num_threads) {
        initialize();

        std::vector<uint8_t> pixels(static_cast<size_t>(image_width) * image_height * 3);
        std::atomic<int> next_row{0};
        std::mutex progress_mutex;
        int rows_done = 0;

        auto worker = [&]() {
            std::mt19937 rng(std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));
            while (true) {
                int j = next_row.fetch_add(1);
                if (j >= image_height) break;

                for (int i = 0; i < image_width; i++) {
                    color pixel_color(0, 0, 0);
                    for (int s = 0; s < samples_per_pixel; s++) {
                        ray r = get_ray(i, j, rng);
                        pixel_color += ray_color(r, max_depth, world, rng);
                    }
                    pixel_color = pixel_color / static_cast<double>(samples_per_pixel);

                    size_t idx = (static_cast<size_t>(j) * image_width + i) * 3;
                    write_color(&pixels[idx], pixel_color);
                }

                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    rows_done++;
                    if (rows_done % 32 == 0 || rows_done == image_height) {
                        std::cerr << "\rScanlines completed: " << rows_done << '/' << image_height
                                  << std::flush;
                    }
                }
            }
        };

        std::vector<std::thread> pool;
        for (int t = 0; t < num_threads; t++) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
        std::cerr << "\nDone.\n";

        return pixels;
    }

    int height() const { return image_height; }

private:
    int image_height;
    point3 center;
    point3 pixel00_loc;
    vec3 pixel_delta_u, pixel_delta_v;
    vec3 u, v, w;               // camera basis vectors
    vec3 defocus_disk_u, defocus_disk_v;

    void initialize() {
        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        center = lookfrom;

        double theta = vfov * 3.14159265358979323846 / 180.0;
        double h = std::tan(theta / 2);
        double viewport_height = 2 * h * focus_dist;
        double viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = viewport_height * -v;

        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        point3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        double defocus_radius = focus_dist * std::tan((defocus_angle / 2) * 3.14159265358979323846 / 180.0);
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    // Builds a randomly-jittered ray through pixel (i, j), optionally
    // originating from a random point on the defocus disk for depth of field.
    ray get_ray(int i, int j, std::mt19937& rng) const {
        vec3 offset = vec3(random_double(rng) - 0.5, random_double(rng) - 0.5, 0);
        point3 pixel_sample = pixel00_loc + ((i + offset.x()) * pixel_delta_u) +
                               ((j + offset.y()) * pixel_delta_v);

        point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rng);
        vec3 ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    point3 defocus_disk_sample(std::mt19937& rng) const {
        vec3 p = random_in_unit_disk(rng);
        return center + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
    }

    // The heart of the path tracer: intersect, shade, recurse.
    // Each bounce multiplies in the surface's attenuation; hitting nothing
    // terminates the path with a soft sky gradient acting as the light
    // source (a simple constant/environment-light approximation).
    color ray_color(const ray& r, int depth, const hittable& world, std::mt19937& rng) const {
        if (depth <= 0) return color(0, 0, 0);

        hit_record rec;
        if (world.hit(r, interval(0.001, std::numeric_limits<double>::infinity()), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered, rng)) {
                return attenuation * ray_color(scattered, depth - 1, world, rng);
            }
            return color(0, 0, 0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        double a = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
    }
};
