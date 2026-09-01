// ray.h - a parametric ray: P(t) = origin + t * direction.
//
// Every camera ray and every scattered/reflected/refracted ray in the
// renderer is one of these. Rendering is fundamentally "shoot rays, find
// where they hit, decide what color comes back."
#pragma once

#include "vec3.h"

class ray {
public:
    ray() {}
    ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}

    const point3& origin() const { return orig; }
    const vec3& direction() const { return dir; }

    point3 at(double t) const { return orig + t * dir; }

private:
    point3 orig;
    vec3 dir;
};
