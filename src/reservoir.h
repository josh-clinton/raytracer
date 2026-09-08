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
// supplies `light_sample` as that type today. Spatial reuse (camera.h's
// spatially_combined_reservoir(), shipped 2026-09) and temporal reuse (next
// project step) both extend this exact same class via combine() below -
// merging two reservoirs is just feeding one's already-chosen sample into
// the other as a single weighted candidate.
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

    // Merges another reservoir's already-chosen winner into this one as a
    // single weighted candidate - the streaming combine rule generalized
    // resampled importance sampling uses to merge reservoirs built from
    // *different* domains (Bitterli et al., "Spatiotemporal Reservoir
    // Resampling", 2020, eq. 6). `other`'s winner stands in for all
    // `other_M` candidates `other` already resampled down to one, so its
    // combine weight is `p_hat_here * other_W * other_M` rather than a
    // plain resampling weight.
    //
    // `rescored_sample` and `p_hat_here` are `other`'s winner and target-
    // function value, both re-evaluated in *this* reservoir's own domain by
    // the caller first (see light_sample::rescored() in camera.h) - a
    // target-function value computed at a different shading point isn't
    // valid here; only the underlying world-space sample (e.g. a point on
    // a light) carries over as-is. `other_M`/`other_W` are simply
    // `other.M`/`other.W()` - passed as plain values rather than the whole
    // reservoir so this stays generic and doesn't need to know `other`'s
    // domain either.
    //
    // Deliberately skips the Jacobian correction full generalized MIS uses
    // when the two domains differ substantially (e.g. very different
    // surface orientation or distance) - fine for the nearby, similarly-
    // oriented neighbors spatial reuse restricts itself to (see camera.h's
    // normal/depth reject checks before this is ever called), but would
    // introduce bias for reservoirs from more different domains and is a
    // known simplification, not an oversight.
    void combine(const Sample& rescored_sample, double p_hat_here, int other_M, double other_W,
                 std::mt19937& rng) {
        if (other_M <= 0 || p_hat_here <= 0) return;
        double weight = p_hat_here * other_W * other_M;
        update(rescored_sample, weight, p_hat_here, rng);
        M += other_M - 1;  // update() already counted 1 candidate; correct up to other_M
    }
};
