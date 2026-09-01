// aabb.h - axis-aligned bounding box, the acceleration primitive behind the
// BVH in bvh.h. A tight AABB lets us reject a whole subtree of geometry with
// one cheap slab test instead of testing every primitive against every ray.
#pragma once

#include "interval.h"
#include "ray.h"
#include "vec3.h"

class aabb {
public:
    interval x, y, z;

    aabb() : x(interval::empty), y(interval::empty), z(interval::empty) {}
    aabb(const interval& ix, const interval& iy, const interval& iz) : x(ix), y(iy), z(iz) {}

    // An empty box - the identity element for aabb(box0, box1) merging,
    // used when folding a list of objects' boxes into one enclosing box.
    static aabb empty_box() { return aabb(interval::empty, interval::empty, interval::empty); }

    // Box that just encloses two points (e.g. a sphere's -r and +r corners).
    aabb(const point3& a, const point3& b) {
        x = (a.x() <= b.x()) ? interval(a.x(), b.x()) : interval(b.x(), a.x());
        y = (a.y() <= b.y()) ? interval(a.y(), b.y()) : interval(b.y(), a.y());
        z = (a.z() <= b.z()) ? interval(a.z(), b.z()) : interval(b.z(), a.z());
    }

    aabb(const aabb& box0, const aabb& box1) {
        x = interval(std::fmin(box0.x.min, box1.x.min), std::fmax(box0.x.max, box1.x.max));
        y = interval(std::fmin(box0.y.min, box1.y.min), std::fmax(box0.y.max, box1.y.max));
        z = interval(std::fmin(box0.z.min, box1.z.min), std::fmax(box0.z.max, box1.z.max));
    }

    const interval& axis_interval(int n) const {
        if (n == 1) return y;
        if (n == 2) return z;
        return x;
    }

    // Classic "slab test": intersect the ray with each axis-aligned pair of
    // planes and shrink [t_min, t_max] each time; if the interval ever goes
    // empty, the ray misses the box.
    bool hit(const ray& r, interval ray_t) const {
        const point3& origin = r.origin();
        const vec3& direction = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const interval& ax = axis_interval(axis);
            double adinv = 1.0 / direction[axis];

            double t0 = (ax.min - origin[axis]) * adinv;
            double t1 = (ax.max - origin[axis]) * adinv;
            if (t0 > t1) std::swap(t0, t1);

            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;
            if (ray_t.max <= ray_t.min) return false;
        }
        return true;
    }

    int longest_axis() const {
        if (x.size() > y.size()) return x.size() > z.size() ? 0 : 2;
        return y.size() > z.size() ? 1 : 2;
    }
};
