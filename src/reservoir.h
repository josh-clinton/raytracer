// reservoir.h - weighted reservoir sampling (WRS) for resampled importance
// sampling (RIS).
//
// The problem this solves: you want to pick one sample out of several
// candidates, with probability proportional to how "good" each one is (its
// resampling weight), without needing to store all the candidates or make
// more than one pass over them. That's exactly what a streaming weighted
// reservoir does - see the "Weighted Reservoir Sampling" section of
// learning/restir_reference.html for the full derivation and a worked
// numeric example.
//
// Deliberately generic: this class knows nothing about lights, shading, or
// BRDFs - it just knows how to stream weighted candidates of some caller-
// supplied Sample type and keep one. camera.h::sample_direct_lighting()
// supplies `light_sample` as that type today. Spatial and temporal reuse
// (the next two project steps) will reuse this exact same class - merging
// two reservoirs together is just feeding one reservoir's chosen sample
// into the other as a single candidate, via this same update() method.
#pragma once

#include <random>

#include "vec3.h"

template <typename Sample>
class reservoir {
public:
    Sample y{};          // the candidate currently being kept
    double w_sum = 0.0;   // running total of every resampling weight streamed through so far
    int M = 0;            // how many candidates this reservoir represents (its own draws, plus
                           // anything absorbed from a merged-in reservoir - see spatial/temporal reuse)
    double p_hat_y = 0.0; // cached target-function value of `y`, so W() below never needs to
                           // re-derive it from scratch

    // Streams one candidate through the reservoir. `weight` is its
    // resampling weight (target_function(candidate) / source_pdf(candidate)
    // - see the RIS section of the reference page); `p_hat_candidate` is
    // that same target-function value, cached for later. The caller does
    // the scoring - this class only does the (provably correct) streaming
    // selection.
    void update(const Sample& candidate, double weight, double p_hat_candidate, std::mt19937& rng) {
        w_sum += weight;
        M += 1;
        if (weight > 0 && random_double(rng) < weight / w_sum) {
            y = candidate;
            p_hat_y = p_hat_candidate;
        }
    }

    // The unbiased contribution weight: multiply this by the *true* value
    // of `y` (e.g. its shadow-ray-verified contribution, not the cheap
    // unshadowed estimate used to score it) to get the final Monte Carlo
    // estimate. Returns 0 for an empty/degenerate reservoir - nothing was
    // ever kept, so there's nothing to weight.
    double W() const {
        if (M == 0 || p_hat_y <= 0) return 0.0;
        return w_sum / (M * p_hat_y);
    }
};
