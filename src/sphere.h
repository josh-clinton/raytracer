// sphere.h - the one primitive this renderer supports.
//
// Ray-sphere intersection is the classic first case to implement because
// it reduces to a quadratic equation, but the surrounding machinery
// (hit_record, materials, bounding boxes) is exactly what you'd reuse for
// triangles, so the design generalizes.
#pragma once

#include "hittable.h"
#include "onb.h"
#include "vec3.h"

class sphere : public hittable {
public:
    sphere(const point3& center, double radius, std::shared_ptr<material> mat)
        : center(center), radius(std::fmax(0, radius)), mat(mat) {
        vec3 rvec(radius, radius, radius);
        bbox = aabb(center - rvec, center + rvec);
    }

    // Solve |P(t) - C|^2 = r^2 for t, where P(t) = origin + t*dir.
    // Expanding gives a quadratic a*t^2 + b*t + c = 0 (written here with the
    // usual half-b optimization to save a couple of multiplies).
    bool hit(const ray& r, interval ray_t, hit_record& rec) const override {
        vec3 oc = center - r.origin();
        double a = r.direction().length_squared();
        double h = dot(r.direction(), oc);
        double c = oc.length_squared() - radius * radius;

        double discriminant = h * h - a * c;
        if (discriminant < 0) return false;

        double sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies within the acceptable t range.
        double root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = mat;

        return true;
    }

    aabb bounding_box() const override { return bbox; }

    const std::shared_ptr<material>& get_material() const { return mat; }

    // --- Light sampling (solid-angle importance sampling) ------------------
    // Used by next-event estimation (camera.h::sample_direct_lighting): given
    // a shading point `origin`, sample a direction toward this sphere,
    // weighted uniformly over the solid angle it actually subtends, instead
    // of over its full surface area. A point on the far side of the sphere
    // (invisible from `origin`) would never contribute anyway, so sampling
    // uniformly over the *visible* cone puts every sample where it can
    // matter - this is the same solid-angle sphere sampling technique used
    // in Shirley's "Ray Tracing: The Rest of Your Life".

    // Solid-angle pdf (w.r.t. direction) of sampling this sphere from
    // `origin` - used to weight a light sample's contribution:
    // contribution = emitted * brdf * cos(theta) / pdf.
    double pdf_value(const point3& origin) const {
        double dist_sq = (center - origin).length_squared();
        if (dist_sq <= radius * radius) return 0;  // origin is inside/on the sphere

        double cos_theta_max = std::sqrt(1 - radius * radius / dist_sq);
        double solid_angle = 2 * pi * (1 - cos_theta_max);
        return 1.0 / solid_angle;
    }

    // Samples a *direction* (unit length) from `origin` toward a point drawn
    // uniformly over the solid angle this sphere subtends - not a uniform
    // point on the sphere's surface, which would waste samples on the half
    // `origin` can't see.
    vec3 random(const point3& origin, std::mt19937& rng) const {
        vec3 direction = center - origin;
        double dist_sq = direction.length_squared();
        onb uvw(direction);
        return uvw.transform(random_to_sphere(radius, dist_sq, rng));
    }

private:
    point3 center;
    double radius;
    std::shared_ptr<material> mat;
    aabb bbox;
};
