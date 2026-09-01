// material.h - defines how a surface scatters light.
//
// Each material answers one question: given an incoming ray that hit this
// surface, what outgoing ray (if any) should we trace next, and how much
// does it attenuate the light carried back to the camera? Chaining these
// scatter events together *is* the Monte Carlo path tracing algorithm.
#pragma once

#include <random>

#include "hittable.h"
#include "vec3.h"

struct hit_record;

class material {
public:
    virtual ~material() = default;

    virtual bool scatter(const ray& /*r_in*/, const hit_record& /*rec*/, color& /*attenuation*/,
                          ray& /*scattered*/, std::mt19937& /*rng*/) const {
        return false;
    }
};

// Ideal matte surface. Scatters uniformly-ish about the normal (normal +
// random unit vector), which approximates a cosine-weighted distribution -
// the standard trick from Lambertian BRDF importance sampling.
class lambertian : public material {
public:
    explicit lambertian(const color& albedo) : albedo(albedo) {}

    bool scatter(const ray& /*r_in*/, const hit_record& rec, color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        vec3 scatter_direction = rec.normal + random_unit_vector(rng);

        // Guard against a degenerate scatter direction if the random unit
        // vector exactly cancels the normal.
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

    color albedo;
};

// Reflective surface (mirror-like). `fuzz` jitters the reflected ray to
// approximate a rough/brushed metal instead of a perfect mirror.
class metal : public material {
public:
    metal(const color& albedo, double fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        reflected = reflected + fuzz * random_unit_vector(rng);
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return dot(scattered.direction(), rec.normal) > 0;
    }

    color albedo;
    double fuzz;
};

// Refractive surface (glass, water). Uses Snell's law plus Schlick's
// approximation for the reflectance-vs-angle trade-off, and randomly
// chooses reflect-vs-refract per sample - another small Monte Carlo choice.
class dielectric : public material {
public:
    explicit dielectric(double refraction_index) : refraction_index(refraction_index) {}

    bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,
                 std::mt19937& rng) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double ri = rec.front_face ? (1.0 / refraction_index) : refraction_index;

        vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = std::fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_double(rng)) {
            direction = reflect(unit_direction, rec.normal);
        } else {
            direction = refract(unit_direction, rec.normal, ri);
        }

        scattered = ray(rec.p, direction);
        return true;
    }

    double refraction_index;

private:
    // Schlick's approximation for the Fresnel reflectance term.
    static double reflectance(double cosine, double refraction_index) {
        double r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * std::pow((1 - cosine), 5);
    }
};
