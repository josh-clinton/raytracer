// color.h - converts a linear-space radiance value into an 8-bit output
// pixel, applying gamma correction so the image doesn't look too dark.
#pragma once

#include <cmath>
#include <cstdint>

#include "interval.h"
#include "vec3.h"

// Linear -> gamma-2 space (approximates sRGB well enough for this project).
inline double linear_to_gamma(double linear_component) {
    if (linear_component > 0) return std::sqrt(linear_component);
    return 0;
}

// Writes one pixel's RGB bytes into an output buffer (row-major, 3 bytes/px).
inline void write_color(uint8_t* out, const color& pixel_color) {
    double r = linear_to_gamma(pixel_color.x());
    double g = linear_to_gamma(pixel_color.y());
    double b = linear_to_gamma(pixel_color.z());

    static const interval intensity(0.000, 0.999);
    out[0] = static_cast<uint8_t>(256 * intensity.clamp(r));
    out[1] = static_cast<uint8_t>(256 * intensity.clamp(g));
    out[2] = static_cast<uint8_t>(256 * intensity.clamp(b));
}
