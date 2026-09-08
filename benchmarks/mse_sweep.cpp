// benchmarks/mse_sweep.cpp - measures direct-lighting quality (MSE vs. a
// high-spp reference) for three camera.h configurations, across scenes with
// different light counts and at different samples-per-pixel budgets:
//
//   baseline   light_candidates=1, spatial_neighbors=0
//              mathematically identical to plain uniform-pick NEE, before
//              RIS/reservoir sampling existed - see camera.h's header comment.
//   reservoir  light_candidates=N (default 4), spatial_neighbors=0
//              RIS only: each shading point scores N candidates and keeps
//              the best one, still one shadow ray per hit.
//   spatial    light_candidates=N, spatial_neighbors=K (default 4)
//              RIS + spatial reuse: the primary vertex additionally folds
//              in K neighboring pixels' already-built reservoirs before
//              resolving - see spatially_combined_reservoir() in camera.h.
//
// Scenes: same "random spheres + a ring of small lights" layout as the
// project's default scene (src/main.cpp), but the ring's light count is the
// swept axis - more/smaller lights is exactly the case uniform-pick NEE
// gets noisiest on and RIS/spatial reuse are meant to help with.
//
// This binary renders ONE data point per invocation - the reference
// (--ref) or one test config (--test) - and is driven repeatedly by
// mse_sweep.py, which handles the light-count/spp/repeats loops, caches
// the reference per light count, and turns the resulting CSV into charts.
// Same split as main.cpp's --bench/--scene modes, for the same reason:
// keep the C++ side a dumb, deterministic single-shot renderer and put
// orchestration/statistics/plotting in Python.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/bvh.h"
#include "../src/camera.h"
#include "../src/hittable_list.h"
#include "../src/material.h"
#include "../src/sphere.h"
#include "../src/vec3.h"

// A "many small lights" scene: a field of diffuse/metal/glass spheres (same
// spirit as the project's main scene) plus a ring of `light_count` small
// emissive spheres. `light_count` is the axis this harness sweeps - bigger
// rings mean a bigger pool for uniform-pick NEE to search blindly, which is
// exactly where RIS/reservoir sampling (and then spatial reuse on top)
// should show a growing advantage.
hittable_list build_lit_scene(std::vector<std::shared_ptr<sphere>>& lights_out, unsigned int seed,
                               int light_count) {
    hittable_list world;
    lights_out.clear();
    std::mt19937 rng(seed);

    auto ground = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground));

    for (int a = -6; a < 6; a++) {
        for (int b = -6; b < 6; b++) {
            double choose = random_double(rng);
            point3 center(a + 0.9 * random_double(rng), 0.2, b + 0.9 * random_double(rng));
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> mat;
                if (choose < 0.7) {
                    mat = std::make_shared<lambertian>(color::random(rng) * color::random(rng));
                } else if (choose < 0.9) {
                    mat = std::make_shared<metal>(color::random(rng, 0.5, 1), random_double(rng, 0, 0.4));
                } else {
                    mat = std::make_shared<dielectric>(1.5);
                }
                world.add(std::make_shared<sphere>(center, 0.2, mat));
            }
        }
    }

    for (int i = 0; i < light_count; i++) {
        double angle = (2.0 * pi * i) / std::max(1, light_count);
        double r = 6.0;
        point3 pos(r * std::cos(angle), 2.5 + 1.5 * ((i % 3) - 1), r * std::sin(angle));
        color c = color::random(rng, 3.0, 7.0);
        auto light_mat = std::make_shared<diffuse_light>(c);
        auto light = std::make_shared<sphere>(pos, 0.3, light_mat);
        world.add(light);
        lights_out.push_back(light);
    }

    return hittable_list(std::make_shared<bvh_node>(world));
}

camera make_camera(int width, int spp, int depth, int candidates, int spatial_neighbors, int spatial_radius) {
    camera cam;
    cam.aspect_ratio = 1.5;
    cam.image_width = width;
    cam.samples_per_pixel = spp;
    cam.max_depth = depth;
    cam.light_candidates = candidates;
    cam.spatial_neighbors = spatial_neighbors;
    cam.spatial_radius = spatial_radius;
    cam.vfov = 40;
    cam.lookfrom = point3(0, 3, 9);
    cam.lookat = point3(0, 1, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0;
    return cam;
}

double mse_vs_reference(const std::vector<uint8_t>& test, const std::vector<uint8_t>& ref) {
    double sum = 0;
    for (size_t i = 0; i < test.size(); i++) {
        double d = static_cast<double>(test[i]) - static_cast<double>(ref[i]);
        sum += d * d;
    }
    return sum / test.size();
}

void write_ppm(const std::string& path, int w, int h, const std::vector<uint8_t>& px) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "error: cannot write %s\n", path.c_str());
        std::exit(1);
    }
    std::fprintf(f, "P6\n%d %d\n255\n", w, h);
    std::fwrite(px.data(), 1, px.size(), f);
    std::fclose(f);
}

std::vector<uint8_t> read_ppm(const std::string& path, int& w, int& h) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "error: cannot read %s (render the reference first with --ref)\n", path.c_str());
        std::exit(1);
    }
    char magic[3] = {0};
    int maxval = 0;
    if (std::fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4 || std::strcmp(magic, "P6") != 0) {
        std::fprintf(stderr, "error: %s is not a well-formed P6 PPM\n", path.c_str());
        std::exit(1);
    }
    std::fgetc(f);  // single whitespace byte separating the header from pixel data
    std::vector<uint8_t> px(static_cast<size_t>(w) * h * 3);
    size_t n = std::fread(px.data(), 1, px.size(), f);
    std::fclose(f);
    if (n != px.size()) {
        std::fprintf(stderr, "error: %s is truncated\n", path.c_str());
        std::exit(1);
    }
    return px;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
                  "Usage:\n"
                  "  %s --ref  <width> <depth> <ref_spp> <ref_candidates> <threads> <seed> "
                  "<light_count> <out_ppm>\n"
                  "  %s --test <width> <depth> <spp> <candidates> <spatial_neighbors> "
                  "<spatial_radius> <threads> <seed> <light_count> <config_label> <ref_ppm>\n"
                  "\n"
                  "--ref renders a high-spp ground truth (spatial reuse forced off - see the\n"
                  "comment at its call site) and saves it as a PPM. --test renders one config,\n"
                  "compares it against that PPM by mean squared error, and prints one CSV line\n"
                  "to stdout:\n"
                  "  light_count,spp,config,candidates,spatial_neighbors,spatial_radius,seed,mse,seconds\n"
                  "\n"
                  "Normally driven by mse_sweep.py - see benchmarks/README.md.\n",
                  prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    std::string mode = argv[1];

    if (mode == "--ref") {
        if (argc < 10) {
            print_usage(argv[0]);
            return 1;
        }
        int width = std::atoi(argv[2]);
        int depth = std::atoi(argv[3]);
        int ref_spp = std::atoi(argv[4]);
        int ref_candidates = std::atoi(argv[5]);
        int threads = std::atoi(argv[6]);
        unsigned int seed = static_cast<unsigned int>(std::strtoul(argv[7], nullptr, 10));
        int light_count = std::atoi(argv[8]);
        std::string out_ppm = argv[9];

        std::vector<std::shared_ptr<sphere>> lights;
        hittable_list world = build_lit_scene(lights, seed, light_count);

        // Spatial reuse stays off for the reference: its Jacobian-skip is a
        // documented, deliberate small-bias simplification (see the "Add
        // spatial reuse..." commit and spatially_combined_reservoir()'s
        // comment in camera.h) - ground truth shouldn't inherit that bias.
        // A generous candidate count is fine here since plain RIS is
        // unbiased regardless of width; it just converges faster.
        camera cam = make_camera(width, ref_spp, depth, ref_candidates, /*spatial_neighbors=*/0,
                                  /*spatial_radius=*/0);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> pixels = cam.render(world, lights, threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::fprintf(stderr, "reference: light_count=%d width=%d ref_spp=%d -> %s (%.1fs)\n", light_count,
                      width, ref_spp, out_ppm.c_str(), std::chrono::duration<double>(t1 - t0).count());
        write_ppm(out_ppm, width, cam.height(), pixels);
        return 0;
    }

    if (mode == "--test") {
        if (argc < 13) {
            print_usage(argv[0]);
            return 1;
        }
        int width = std::atoi(argv[2]);
        int depth = std::atoi(argv[3]);
        int spp = std::atoi(argv[4]);
        int candidates = std::atoi(argv[5]);
        int spatial_neighbors = std::atoi(argv[6]);
        int spatial_radius = std::atoi(argv[7]);
        int threads = std::atoi(argv[8]);
        unsigned int seed = static_cast<unsigned int>(std::strtoul(argv[9], nullptr, 10));
        int light_count = std::atoi(argv[10]);
        std::string config_label = argv[11];
        std::string ref_ppm = argv[12];

        std::vector<std::shared_ptr<sphere>> lights;
        // Same seed as the --ref call that built ref_ppm -> identical scene
        // geometry, so the only thing differing between test and reference
        // is the direct-lighting technique and sample count being measured.
        hittable_list world = build_lit_scene(lights, seed, light_count);
        camera cam = make_camera(width, spp, depth, candidates, spatial_neighbors, spatial_radius);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> pixels = cam.render(world, lights, threads);
        auto t1 = std::chrono::high_resolution_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();

        int ref_w = 0, ref_h = 0;
        std::vector<uint8_t> ref_pixels = read_ppm(ref_ppm, ref_w, ref_h);
        if (ref_w != width || ref_h != cam.height()) {
            std::fprintf(stderr, "error: reference %s is %dx%d, test render is %dx%d - can't compare\n",
                          ref_ppm.c_str(), ref_w, ref_h, width, cam.height());
            return 1;
        }

        double mse = mse_vs_reference(pixels, ref_pixels);
        std::fprintf(stderr, "  light_count=%-3d spp=%-4d %-10s candidates=%d spatial_neighbors=%d -> mse=%.4f (%.1fs)\n",
                      light_count, spp, config_label.c_str(), candidates, spatial_neighbors, mse, seconds);
        std::printf("%d,%d,%s,%d,%d,%d,%u,%.6f,%.3f\n", light_count, spp, config_label.c_str(), candidates,
                     spatial_neighbors, spatial_radius, seed, mse, seconds);
        return 0;
    }

    print_usage(argv[0]);
    return 1;
}
