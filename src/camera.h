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
//     build_initial_reservoir()+resolve_reservoir() draw `light_candidates`
//     cheap (light, direction) candidates, score each by unshadowed
//     contribution, and stream them through a reservoir (reservoir.h) so
//     the one sample that actually pays for a shadow ray is whichever
//     looked most promising - not just a blind uniform pick. Same one-
//     shadow-ray-per-hit cost as plain next-event estimation, aimed far
//     better.
//   - Spatial reuse: the *primary* vertex (only - see the render()/
//     trace_primary()/finish_pixel() split below) additionally folds in a
//     few neighboring pixels' already-built reservoirs before resolving,
//     via spatially_combined_reservoir() - free extra candidates, since
//     each neighbor already paid for its own RIS draws this sample.
//     Temporal reuse (next project step) extends this same mechanism
//     across frames instead of across pixels.
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

// A single (light, point-on-light) candidate for direct-light reservoir
// resampling - the Sample type reservoir<Sample> streams through. Stores
// the actual world-space point sampled on the light, not just a direction:
// a direction is only meaningful from the origin it was drawn at, but
// spatial (and temporal) reuse need to re-evaluate a candidate from a
// *different* shading point, and a fixed world point is what makes that
// possible.
struct light_sample {
    size_t light_idx = 0;
    point3 light_point;         // world-space point on the light this candidate targets
    color unshadowed{0, 0, 0};  // Le * BRDF * cos(theta) *at the point this was built/rescored for* -
                                 // no shadow ray yet, and stale if reused elsewhere without rescored()

    // Re-evaluates this candidate's unshadowed contribution against a
    // *different* shading point than whatever point built it - what
    // spatial reuse needs before folding a neighbor's winning candidate
    // into the current pixel's reservoir, since the cached `unshadowed`
    // above is only ever valid at the point it was computed for. Returns
    // unshadowed=0 (a harmless, zero-weight candidate) if this light point
    // isn't usable from the new vertex, e.g. it's behind the surface here
    // even though it wasn't behind the original surface.
    light_sample rescored(const hit_record& new_rec, const color& new_brdf_albedo,
                           const std::vector<std::shared_ptr<sphere>>& lights) const {
        vec3 to_light = light_point - new_rec.p;
        double dist_sq = to_light.length_squared();
        if (dist_sq < 1e-12) return light_sample{light_idx, light_point, color(0, 0, 0)};

        vec3 direction = to_light / std::sqrt(dist_sq);
        double cos_theta = dot(new_rec.normal, direction);
        if (cos_theta <= 0) return light_sample{light_idx, light_point, color(0, 0, 0)};

        color light_emit = lights[light_idx]->get_material()->emitted();
        color lambertian_brdf = new_brdf_albedo / pi;
        return light_sample{light_idx, light_point, light_emit * lambertian_brdf * cos_theta};
    }
};

class camera {
public:
    double aspect_ratio = 1.0;
    int image_width = 400;
    int samples_per_pixel = 100;
    int max_depth = 10;
    int light_candidates = 4;   // RIS candidates scored per shading point before
                                 // the reservoir's one winner pays for a shadow ray

    int spatial_neighbors = 4;  // extra already-built reservoirs the primary vertex tries to
                                 // fold in via spatial reuse each sample; 0 disables spatial reuse
    int spatial_radius = 20;    // pixel search radius (each axis) spatial reuse draws neighbor
                                 // candidates from

    double vfov = 90;              // vertical field-of-view, in degrees
    point3 lookfrom = point3(0, 0, 0);
    point3 lookat = point3(0, 0, -1);
    vec3 vup = vec3(0, 1, 0);

    double defocus_angle = 0;      // depth-of-field cone angle; 0 = pinhole camera
    double focus_dist = 10;

    // Renders the scene into a flat RGB8 buffer (image_width * image_height * 3
    // bytes) using `num_threads` worker threads. `lights` lists every
    // emissive object next-event estimation should sample directly; pass an
    // empty vector for a scene with no explicit lights (direct-light
    // sampling is simply skipped, same as before this feature existed).
    //
    // Runs `samples_per_pixel` full image sweeps, each split into two
    // row-parallel passes with a full barrier between them (all threads
    // finish phase A before phase B starts) - not one combined pass like
    // before spatial reuse existed. Phase A traces every pixel's primary
    // ray for this sample and builds its *initial* reservoir; phase B then
    // needs every pixel's phase-A result already sitting in `buffer` before
    // it can pull neighbors' reservoirs into its own, which the old single-
    // pass-per-pixel loop couldn't provide - a pixel processed early in a
    // single pass has no way to see a not-yet-processed neighbor's data.
    std::vector<uint8_t> render(const hittable& world,
                                 const std::vector<std::shared_ptr<sphere>>& lights,
                                 int num_threads) {
        initialize();

        size_t num_pixels = static_cast<size_t>(image_width) * image_height;
        std::vector<color> accum(num_pixels, color(0, 0, 0));
        std::vector<primary_vertex> buffer(num_pixels);

        for (int s = 0; s < samples_per_pixel; s++) {
            parallel_for_rows(num_threads, [&](int j, std::mt19937& rng) {
                for (int i = 0; i < image_width; i++) {
                    buffer[static_cast<size_t>(j) * image_width + i] = trace_primary(i, j, world, lights, rng);
                }
            });

            parallel_for_rows(num_threads, [&](int j, std::mt19937& rng) {
                for (int i = 0; i < image_width; i++) {
                    size_t idx = static_cast<size_t>(j) * image_width + i;
                    accum[idx] += finish_pixel(i, j, buffer, world, lights, rng);
                }
            });

            std::cerr << "\rSample " << (s + 1) << '/' << samples_per_pixel << std::flush;
        }
        std::cerr << "\nDone.\n";

        std::vector<uint8_t> pixels(num_pixels * 3);
        for (size_t idx = 0; idx < num_pixels; idx++) {
            write_color(&pixels[idx * 3], accum[idx] / static_cast<double>(samples_per_pixel));
        }
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

    // What phase A (trace_primary()) found at one pixel's primary hit, for
    // phase B (finish_pixel()) to spatially resolve and finish shading.
    // `kind` distinguishes the cases that need no further work beyond
    // `result` (the ray missed everything, or the path ended right here)
    // from the one phase B actually has work to do for.
    struct primary_vertex {
        enum class kind { background, terminated, no_direct, has_reservoir } k = kind::background;
        color result;                 // fully resolved color; valid when k is background or terminated
        hit_record rec;               // valid when k is no_direct or has_reservoir
        color brdf_albedo;
        color emitted_light;
        ray scattered;
        bool scatter_specular = false;
        reservoir<light_sample> res;  // initial, unresolved reservoir; valid when k == has_reservoir
    };

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

    // Runs `row_fn(j, rng)` for every row 0..image_height-1, distributed
    // across `num_threads` workers pulling rows off a shared atomic counter.
    // Factored out because render() now needs two full-image row-parallel
    // passes per sample (phase A, phase B) instead of one; each call here
    // is a full barrier (every worker finishes before the next call's
    // workers start), which is exactly what phase B needs from phase A.
    // Trades a little thread-spawn overhead (2x samples_per_pixel spawns
    // instead of 1 persistent pool) for not needing a manual barrier
    // primitive - fine at this project's scale, a documented place to
    // optimize later if it ever isn't.
    void parallel_for_rows(int num_threads, const std::function<void(int, std::mt19937&)>& row_fn) {
        std::atomic<int> next_row{0};
        auto worker = [&]() {
            std::mt19937 rng(std::random_device{}() ^ std::hash<std::thread::id>{}(std::this_thread::get_id()));
            while (true) {
                int j = next_row.fetch_add(1);
                if (j >= image_height) break;
                row_fn(j, rng);
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < num_threads; t++) pool.emplace_back(worker);
        for (auto& t : pool) t.join();
    }

    // Draws `light_candidates` cheap (light, point-on-light) candidates
    // from `rec` and streams them through a fresh reservoir - the RIS
    // candidate-generation step shared by plain per-vertex direct lighting
    // (sample_direct_lighting(), every non-primary bounce) and the primary
    // vertex's phase A (trace_primary() - spatial reuse combines this
    // reservoir with neighbors' before anyone pays for a shadow ray).
    reservoir<light_sample> build_initial_reservoir(const hit_record& rec,
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

            // Resolve the actual world-space point on the light this
            // direction hits - cheap (one sphere, not the whole scene) and
            // needed so the candidate can be re-evaluated from a
            // *different* shading point later (spatial/temporal reuse),
            // since a direction alone is only meaningful from the origin
            // it was drawn at.
            hit_record light_hit;
            if (!light->hit(ray(rec.p, light_dir), interval(0.001, std::numeric_limits<double>::infinity()),
                             light_hit)) {
                continue;  // shouldn't happen - random() targets the visible disc - but guard anyway
            }

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

            res.update(light_sample{idx, light_hit.p, unshadowed}, weight, p_hat, rng);
        }
        return res;
    }

    // Pays for exactly one shadow ray - on `res`'s winner only - and turns
    // it into the final RIS direct-lighting estimate f(y) * W(y). Shared by
    // plain per-vertex direct lighting and the primary vertex's phase B
    // (after spatial reuse has already merged neighbors in), since
    // resolving a reservoir's winner is the same operation either way.
    color resolve_reservoir(const reservoir<light_sample>& res, const hit_record& rec,
                             const hittable& world) const {
        if (res.M == 0 || res.p_hat_y <= 0) return color(0, 0, 0);  // every candidate was degenerate

        const light_sample& winner = res.y;
        vec3 to_light = winner.light_point - rec.p;
        double dist_to_light = to_light.length();
        if (dist_to_light < 1e-6) return color(0, 0, 0);  // degenerate; shouldn't happen
        vec3 direction = to_light / dist_to_light;

        ray shadow_ray(rec.p, direction);
        hit_record shadow_rec;
        if (world.hit(shadow_ray, interval(0.001, dist_to_light - 0.001), shadow_rec)) {
            return color(0, 0, 0);  // something else sits between the surface and the light
        }

        // RIS estimator: L_direct ~= f(y) * W_y. f(y) is the true,
        // shadow-verified contribution - here that's just `unshadowed`
        // again, since we've now confirmed nothing blocks it; W_y is the
        // reservoir's own bookkeeping resolving to the unbiased weight.
        return winner.unshadowed * res.W();
    }

    // Resampled importance sampling (RIS) for direct lighting at a
    // non-primary vertex: draw `light_candidates` candidates, stream them
    // through a reservoir, resolve the winner with one shadow ray. Used by
    // every bounce past the primary vertex - see trace_primary()/
    // finish_pixel() for the primary vertex's version, which additionally
    // spatially reuses neighboring pixels' reservoirs before resolving.
    color sample_direct_lighting(const hit_record& rec, const hittable& world,
                                  const std::vector<std::shared_ptr<sphere>>& lights,
                                  const color& brdf_albedo, std::mt19937& rng) const {
        reservoir<light_sample> res = build_initial_reservoir(rec, lights, brdf_albedo, rng);
        return resolve_reservoir(res, rec, world);
    }

    // Spatial reuse: starting from this pixel's own initial reservoir,
    // folds in up to `spatial_neighbors` already-built reservoirs from
    // other pixels within `spatial_radius` - free extra candidates, since
    // each neighbor already paid for its own RIS draws this sample. A
    // neighbor is skipped if its surface looks meaningfully different
    // (normal or depth) from this pixel's, since reusing a reservoir across
    // a silhouette or depth edge (e.g. a floor pixel's chosen candidate
    // reused on a wall pixel behind it) biases the result, rather than just
    // adding a harmless extra sample the way a same-surface neighbor does.
    reservoir<light_sample> spatially_combined_reservoir(int i, int j, const primary_vertex& pv,
                                                           const std::vector<primary_vertex>& buffer,
                                                           const std::vector<std::shared_ptr<sphere>>& lights,
                                                           std::mt19937& rng) const {
        reservoir<light_sample> merged = pv.res;
        if (spatial_neighbors <= 0) return merged;

        for (int n = 0; n < spatial_neighbors; n++) {
            int di = static_cast<int>(random_double(rng, -spatial_radius, spatial_radius + 1));
            int dj = static_cast<int>(random_double(rng, -spatial_radius, spatial_radius + 1));
            if (di == 0 && dj == 0) continue;

            int ni = i + di, nj = j + dj;
            if (ni < 0 || ni >= image_width || nj < 0 || nj >= image_height) continue;

            const primary_vertex& neighbor = buffer[static_cast<size_t>(nj) * image_width + ni];
            if (neighbor.k != primary_vertex::kind::has_reservoir || neighbor.res.M == 0) continue;

            // Reject neighbors whose surface doesn't look like this
            // pixel's own: a normal-angle check catches different-object/
            // different-orientation neighbors (e.g. across a silhouette),
            // and a relative-depth check catches neighbors much nearer or
            // farther along the view ray (e.g. across a foreground/
            // background edge) even when normals happen to roughly agree.
            if (dot(pv.rec.normal, neighbor.rec.normal) < 0.9) continue;
            double t_here = pv.rec.t, t_there = neighbor.rec.t;
            if (std::fabs(t_here - t_there) > 0.1 * std::max(t_here, t_there)) continue;

            light_sample rescored = neighbor.res.y.rescored(pv.rec, pv.brdf_albedo, lights);
            double p_hat_here = luminance(rescored.unshadowed);
            merged.combine(rescored, p_hat_here, neighbor.res.M, neighbor.res.W(), rng);
        }

        return merged;
    }

    // Phase A: trace pixel (i, j)'s primary ray for this sample and, for
    // any diffuse hit with lights to sample, build its *initial* reservoir
    // (candidates scored, none resolved/shadow-tested yet - phase B does
    // that, after spatial reuse has had a chance to fold neighbors in).
    primary_vertex trace_primary(int i, int j, const hittable& world,
                                  const std::vector<std::shared_ptr<sphere>>& lights,
                                  std::mt19937& rng) const {
        primary_vertex pv;

        if (max_depth <= 0) {
            pv.k = primary_vertex::kind::terminated;
            pv.result = color(0, 0, 0);  // matches the old ray_color()'s depth<=0 guard: nothing traced at all
            return pv;
        }

        ray r = get_ray(i, j, rng);
        hit_record rec;
        if (!world.hit(r, interval(0.001, std::numeric_limits<double>::infinity()), rec)) {
            vec3 unit_direction = unit_vector(r.direction());
            double a = 0.5 * (unit_direction.y() + 1.0);
            pv.k = primary_vertex::kind::background;
            pv.result = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
            return pv;
        }

        color emitted = rec.mat->emitted();  // primary ray always counts emission - nothing upstream could have NEE'd it
        ray scattered;
        color attenuation;
        if (!rec.mat->scatter(r, rec, attenuation, scattered, rng)) {
            pv.k = primary_vertex::kind::terminated;
            pv.result = emitted;
            return pv;
        }

        pv.rec = rec;
        pv.emitted_light = emitted;
        pv.scattered = scattered;
        pv.brdf_albedo = attenuation;
        pv.scatter_specular = rec.mat->is_specular();

        if (pv.scatter_specular || lights.empty()) {
            pv.k = primary_vertex::kind::no_direct;  // nothing for phase B to spatially resolve
            return pv;
        }

        pv.res = build_initial_reservoir(rec, lights, attenuation, rng);
        pv.k = primary_vertex::kind::has_reservoir;
        return pv;
    }

    // Phase B: for pixel (i, j)'s primary vertex, spatially combine in
    // neighbors' reservoirs (if it has one), resolve the merged winner with
    // one shadow ray, and finish the path exactly like ray_color() always
    // did for everything past the primary vertex.
    color finish_pixel(int i, int j, const std::vector<primary_vertex>& buffer, const hittable& world,
                        const std::vector<std::shared_ptr<sphere>>& lights, std::mt19937& rng) const {
        const primary_vertex& pv = buffer[static_cast<size_t>(j) * image_width + i];

        if (pv.k == primary_vertex::kind::background || pv.k == primary_vertex::kind::terminated) {
            return pv.result;
        }

        color direct(0, 0, 0);
        if (pv.k == primary_vertex::kind::has_reservoir) {
            reservoir<light_sample> merged = spatially_combined_reservoir(i, j, pv, buffer, lights, rng);
            direct = resolve_reservoir(merged, pv.rec, world);
        }

        color indirect = pv.brdf_albedo *
                          ray_color(pv.scattered, max_depth - 1, world, lights, rng, pv.scatter_specular);

        return pv.emitted_light + direct + indirect;
    }

    // The heart of the path tracer past the primary vertex: intersect,
    // shade, recurse. Each bounce multiplies in the surface's attenuation;
    // hitting nothing terminates the path with a soft sky gradient acting
    // as the light source (a simple constant/environment-light
    // approximation). The primary vertex itself is handled separately by
    // trace_primary()/finish_pixel() above, so that its direct-lighting
    // reservoir can be spatially combined with neighboring pixels' before
    // being resolved - something a single recursive function can't do,
    // since it needs every pixel's initial reservoir to already exist
    // first (see render()'s two-phase-per-sample structure).
    //
    // `count_emission` guards against double-counting a light: it's true
    // for a ray following a specular (mirror/glass) bounce, since that
    // vertex couldn't have explicitly sampled the light via NEE. It's false
    // for a ray following a diffuse bounce, because direct lighting already
    // accounted for whatever light is visible from that vertex - letting
    // the indirect ray *also* add emitted() if it happens to land on a
    // light would count that light twice.
    color ray_color(const ray& r, int depth, const hittable& world,
                     const std::vector<std::shared_ptr<sphere>>& lights, std::mt19937& rng,
                     bool count_emission) const {
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
