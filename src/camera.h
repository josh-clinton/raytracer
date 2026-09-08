// camera.h - builds camera rays and drives the render loop.
//
// This is where several concepts from the JD come together in code:
//   - Monte Carlo integration: each pixel averages `samples_per_pixel`
//     independent random rays (antialiasing + soft shadows + glossy
//     reflection all fall out of this one averaging loop for free).
//   - Recursive ray tracing: ray_color() recurses on scattered rays up to
//     `max_depth`, accumulating attenuation - a direct implementation of
//     the rendering equation truncated to a finite number of bounces.
//   - Resampled importance sampling: at every non-specular hit,
//     sample_direct_lighting() draws `light_candidates` cheap (light,
//     direction) candidates, scores each by unshadowed contribution, and
//     streams them through a reservoir (reservoir.h) so the one sample that
//     actually pays for a shadow ray is whichever looked most promising -
//     not just a blind uniform pick. Same one-shadow-ray-per-hit cost as
//     plain next-event estimation, aimed far better. Spatial and temporal
//     reuse (next project steps) extend this same reservoir by merging in
//     neighboring pixels'/previous frames' already-built reservoirs as
//     extra free candidates.
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
#include "reservoir.h"
#include "sphere.h"
#include "vec3.h"

// A single (light, direction) candidate for direct-light reservoir
// resampling - the Sample type reservoir<Sample> streams through. Carries
// its own cheap unshadowed contribution so the eventual winner's color
// doesn't need to be re-derived after the reservoir has already discarded
// every other candidate.
struct light_sample {
    size_t light_idx = 0;
    vec3 direction;
    color unshadowed{0, 0, 0};  // Le * BRDF * cos(theta) - no shadow ray yet
};

class camera {
public:
    double aspect_ratio = 1.0;
    int image_width = 400;
    int samples_per_pixel = 100;
    int max_depth = 10;
    int light_candidates = 4;  // RIS candidates scored per shading point before
                                // the reservoir's one winner pays for a shadow ray

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

    // Resampled importance sampling (RIS) for direct lighting: draw
    // `light_candidates` cheap candidates - same distribution the old
    // single-sample version used (uniform light pick + solid-angle
    // direction sample, sphere.h) - score each by its unshadowed
    // contribution, and stream them through a reservoir (reservoir.h) so
    // the one that's kept is whichever actually looked most promising.
    // Only that winner ever pays for a shadow ray - same one-shadow-ray-
    // per-shading-point cost as before, just aimed by up to
    // `light_candidates` scored options instead of a single blind pick.
    color sample_direct_lighting(const hit_record& rec, const hittable& world,
                                  const std::vector<std::shared_ptr<sphere>>& lights,
                                  const color& brdf_albedo, std::mt19937& rng) const {
        double num_lights = static_cast<double>(lights.size());
        reservoir<light_sample> res;

        for (int i = 0; i < light_candidates; i++) {
            size_t idx = static_cast<size_t>(random_double(rng) * lights.size());
            if (idx >= lights.size()) idx = lights.size() - 1;
            const auto& light = lights[idx];

            // Solid-angle pdf of hitting this light from rec.p, *given* it
            // was the light picked. 0 means rec.p is inside/touching the
            // light (or otherwise degenerate) - skip this candidate rather
            // than feeding a meaningless weight into the reservoir.
            double pdf_dir = light->pdf_value(rec.p);
            if (pdf_dir <= 0) continue;

            vec3 light_dir = light->random(rec.p, rng);  // unit direction, sampled within the light's cone
            double cos_theta_surface = dot(rec.normal, light_dir);
            if (cos_theta_surface <= 0) continue;  // light is behind the surface from here

            color light_emit = light->get_material()->emitted();
            color lambertian_brdf = brdf_albedo / pi;  // Lambertian BRDF = albedo / pi
            color unshadowed = light_emit * lambertian_brdf * cos_theta_surface;  // p_hat's basis - no shadow ray yet

            // Resampling weight w_i = p_hat(x_i) / p(x_i). p_hat is this
            // candidate's scalar "importance" (luminance of its unshadowed
            // contribution - see vec3.h::luminance()); p(x_i) is the actual
            // probability this exact candidate was drawn: P(pick this
            // light) * P(this direction | light) = (1/num_lights) * pdf_dir.
            double p_hat = luminance(unshadowed);
            double p_src = pdf_dir / num_lights;
            double weight = (p_src > 0) ? p_hat / p_src : 0.0;

            res.update(light_sample{idx, light_dir, unshadowed}, weight, p_hat, rng);
        }

        if (res.M == 0 || res.p_hat_y <= 0) return color(0, 0, 0);  // every candidate was degenerate

        // Pay for exactly one shadow ray, on the reservoir's winner only.
        const light_sample& winner = res.y;
        const auto& light = lights[winner.light_idx];

        // The winning direction was constructed to hit its light, so this
        // recovers the exact distance to it (reusing sphere::hit(), which
        // already implements the intersection) without separate geometry math.
        hit_record light_hit;
        if (!light->hit(ray(rec.p, winner.direction), interval(0.001, std::numeric_limits<double>::infinity()),
                         light_hit)) {
            return color(0, 0, 0);  // shouldn't happen; guard against float edge cases
        }

        ray shadow_ray(rec.p, winner.direction);
        hit_record shadow_rec;
        if (world.hit(shadow_ray, interval(0.001, light_hit.t - 0.001), shadow_rec)) {
            return color(0, 0, 0);  // something else sits between the surface and the light
        }

        // RIS estimator: L_direct ~= f(y) * W_y. f(y) is the true,
        // shadow-verified contribution - here that's just `unshadowed`
        // again, since we've now confirmed nothing blocks it; W_y is the
        // reservoir's own bookkeeping resolving to the unbiased weight.
        return winner.unshadowed * res.W();
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
