// sphere.h - the one primitive this renderer supports.
//
// Ray-sphere intersection is the classic first case to implement because
// it reduces to a quadratic equation, but the surrounding machinery
// (hit_record, materials, bounding boxes) is exactly what you'd reuse for
// triangles, so the design generalizes.
#pragma once

#include "hittable.h"
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

private:
    point3 center;
    double radius;
    std::shared_ptr<material> mat;
    aabb bbox;
};
