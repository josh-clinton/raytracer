// hittable.h - the interface every intersectable object implements.
//
// This is the abstraction that lets the renderer treat a single sphere, a
// list of spheres, and a whole BVH subtree identically: "given a ray and a
// valid t-range, can you find the closest hit?"
#pragma once

#include <memory>

#include "aabb.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"

class material;

// Filled in by hit() when a ray intersects something. Carries everything
// downstream shading code needs: hit point, surface normal (always facing
// the incoming ray), the material to shade with, and the ray parameter t.
struct hit_record {
    point3 p;
    vec3 normal;
    std::shared_ptr<material> mat;
    double t;
    bool front_face;

    // Ensures `normal` always points against the incoming ray, and records
    // whether we hit the outside or inside of the surface (matters for
    // dielectrics, which behave differently entering vs. exiting glass).
    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual ~hittable() = default;

    virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
    virtual aabb bounding_box() const = 0;
};
