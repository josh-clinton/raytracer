// camera.h - builds camera rays and drives the render loop.
//
// This is where several concepts from the JD come together in code:
//   - Monte Carlo integration: each pixel averages `samples_per_pixel`
//     independent random rays (antialiasing + soft shadows + glossy
//     reflection all fall out of this one averaging loop for free).
//   - Recursive ray tracing: ray_color() recurses on scattered rays up to
//     `max_depth`, accumulating attenuation - a direct implementation of
//     the rendering equation truncated to a finite number of bounces.
//   - Next-event estimation: at every non-specular hit, sample_direct_lighting()
//     picks a light, importance-samples a direction toward it, and traces
//     one shadow ray - instead of hoping a blind BRDF bounce wanders onto a
//     light by chance. This is the "many lights" direct-lighting baseline
//     ReSTIR (next project step) is built to make cheaper and lower-noise:
//     ReSTIR reuses this exact "score candidates cheaply, verify the winner
//     with one ray" shape, just with many candidates resolved into one
//     reservoir instead of a single uniform pick.
//   - Parallelism: rows are distributed across a thread pool, which is the
//     minimum viable version of the "profile, debug, and optimize graphics
//     workloads for performance" bullet in the JD.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "color.h"
#include "hittable.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
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
    // bytes) using `num_threads` worker threads, one row-range each. `lights`
    // lists every emissive object next-event estimation should sample
    // directly; pass an empty vector for a scene with no explicit lights
    // (direct-light sampling is simply skipped, same as before this feature
    // existed).
    std::vector<uint8_t> render(const hittable& world,
                                 const std::vector<std::shared_ptr<sphere>>& lights,
                                 int num_threads) {
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
                        pixel_color += ray_color(r, max_depth, world, lights, rng);
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

    // Explicit direct-light sampling (next event estimation): pick one light
    // uniformly out of `lights`, importance-sample a direction toward it via
    // solid-angle cone sampling (sphere.h), and trace exactly one shadow ray
    // to verify it's actually visible. This "score a candidate cheaply, only
    // pay for a shadow ray on the one you're using" shape is deliberately
    // how ReSTIR's reservoir resampling works too, just with a single
    // candidate here instead of many resampled into one reservoir - this
    // function's noise (visible as blotchy shadows/highlights when lights
    // are numerous) is exactly what the reservoir version is measured
    // against.
    color sample_direct_lighting(const hit_record& rec, const hittable& world,
                                  const std::vector<std::shared_ptr<sphere>>& lights,
                                  const color& brdf_albedo, std::mt19937& rng) const {
        size_t light_idx = static_cast<size_t>(random_double(rng) * lights.size());
        if (light_idx >= lights.size()) light_idx = lights.size() - 1;
        const auto& light = lights[light_idx];

        // Solid-angle pdf of hitting this light from rec.p, *given* it was
        // the light picked. 0 means rec.p is inside/touching the light (or
        // otherwise degenerate) - nothing sensible to sample.
        double pdf_dir = light->pdf_value(rec.p);
        if (pdf_dir <= 0) return color(0, 0, 0);

        vec3 light_dir = light->random(rec.p, rng);  // unit direction, sampled within the light's cone
        double cos_theta_surface = dot(rec.normal, light_dir);
        if (cos_theta_surface <= 0) return color(0, 0, 0);  // light is behind the surface

        // The direction was constructed to hit the light, so this recovers
        // the exact distance to it (as a byproduct, reusing the same
        // ray-sphere intersection sphere::hit() already implements) without
        // needing separate geometry math.
        hit_record light_hit;
        if (!light->hit(ray(rec.p, light_dir), interval(0.001, std::numeric_limits<double>::infinity()),
                         light_hit)) {
            return color(0, 0, 0);  // shouldn't happen; guard against float edge cases
        }

        ray shadow_ray(rec.p, light_dir);
        hit_record shadow_rec;
        if (world.hit(shadow_ray, interval(0.001, light_hit.t - 0.001), shadow_rec)) {
            return color(0, 0, 0);  // something else sits between the surface and the light
        }

        color light_emit = light->get_material()->emitted();
        color lambertian_brdf = brdf_albedo / pi;  // Lambertian BRDF = albedo / pi
        double num_lights = static_cast<double>(lights.size());

        // Monte Carlo estimator for the direct-lighting integral:
        // Le * BRDF * cos(theta) / pdf(direction). pdf_dir is the pdf
        // *given* this light was picked; picking it happened with
        // probability 1/num_lights, so the combined pdf is pdf_dir /
        // num_lights - dividing by that is the same as multiplying by
        // num_lights, which is what appears below.
        return light_emit * lambertian_brdf * cos_theta_surface * num_lights / pdf_dir;
    }

    // The heart of the path tracer: intersect, shade, recurse.
    // Each bounce multiplies in the surface's attenuation; hitting nothing
    // terminates the path with a soft sky gradient acting as the light
    // source (a simple constant/environment-light approximation).
    //
    // `count_emission` guards against double-counting a light: it's true
    // for the primary camera ray and for any ray following a specular
    // (mirror/glass) bounce, since neither of those vertices could have
    // explicitly sampled the light via NEE. It's false for a ray following
    // a diffuse bounce, because sample_direct_lighting() already accounted
    // for whatever light is visible from that vertex - letting the
    // indirect ray *also* add emitted() if it happens to land on a light
    // would count that light twice.
    color ray_color(const ray& r, int depth, const hittable& world,
                     const std::vector<std::shared_ptr<sphere>>& lights, std::mt19937& rng,
                     bool count_emission = true) const {
        if (depth <= 0) return color(0, 0, 0);

        hit_record rec;
        if (!world.hit(r, interval(0.001, std::numeric_limits<double>::infinity()), rec)) {
            vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
        }

        color emitted = count_emission ? rec.mat->emitted() : color(0, 0, 0);

        ray scattered;
        color attenuation;
        if (!rec.mat->scatter(r, rec, attenuation, scattered, rng)) {
            return emitted;
        }

        bool specular = rec.mat->is_specular();
        color direct(0, 0, 0);
        if (!specular && !lights.empty()) {
            direct = sample_direct_lighting(rec, world, lights, attenuation, rng);
        }

        color indirect = attenuation * ray_color(scattered, depth - 1, world, lights, rng, specular);

        return emitted + direct + indirect;
    }
};
