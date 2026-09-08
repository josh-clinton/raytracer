// vec3.h - a minimal 3D vector class used for points, directions, and colors.
//
// This is the workhorse type of the whole renderer: camera rays, surface
// normals, and RGB colors are all just vec3s with different meanings.
#pragma once

#include <cmath>
#include <iostream>
#include <random>

// Shared math constant. Used by sphere.h's solid-angle light sampling
// (a sphere light's pdf is 1 / (2*pi*(1-cos_theta_max))) and anywhere else
// that needs it, so it isn't redefined ad hoc in multiple files.
constexpr double pi = 3.14159265358979323846;

class vec3 {
public:
    double e[3];

    vec3() : e{0, 0, 0} {}
    vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}

    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }

    vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }

    vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    vec3& operator/=(double t) { return *this *= 1 / t; }

    double length() const { return std::sqrt(length_squared()); }

    double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    // True if the vector is very close to zero in all dimensions.
    // Used to avoid degenerate scatter directions (normal + random == 0).
    bool near_zero() const {
        const double eps = 1e-8;
        return std::fabs(e[0]) < eps && std::fabs(e[1]) < eps && std::fabs(e[2]) < eps;
    }

    static vec3 random(std::mt19937& rng);
    static vec3 random(std::mt19937& rng, double min, double max);
};

// point3 and color are just aliases for vec3 - they carry no extra behavior,
// but make function signatures read clearly (a color isn't a direction).
using point3 = vec3;
using color = vec3;

// --- Vector arithmetic -------------------------------------------------

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

// Component-wise product: used to tint light by an object's albedo.
inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3 operator*(const vec3& v, double t) { return t * v; }

inline vec3 operator/(const vec3& v, double t) { return (1 / t) * v; }

inline double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3 unit_vector(const vec3& v) { return v / v.length(); }

// --- Random vectors (needed for Monte Carlo sampling) ------------------

inline double random_double(std::mt19937& rng, double min = 0.0, double max = 1.0) {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return min + (max - min) * dist(rng);
}

inline vec3 vec3::random(std::mt19937& rng) {
    return vec3(random_double(rng), random_double(rng), random_double(rng));
}

inline vec3 vec3::random(std::mt19937& rng, double min, double max) {
    return vec3(random_double(rng, min, max), random_double(rng, min, max),
                random_double(rng, min, max));
}

// Rejection-sampled random point inside the unit sphere. Used to build a
// cosine-weighted-ish diffuse scatter direction (Lambertian reflectance).
inline vec3 random_in_unit_sphere(std::mt19937& rng) {
    while (true) {
        vec3 p = vec3::random(rng, -1, 1);
        if (p.length_squared() < 1) return p;
    }
}

inline vec3 random_unit_vector(std::mt19937& rng) {
    return unit_vector(random_in_unit_sphere(rng));
}

// Random point in the unit disk, used for camera defocus blur (depth of field).
inline vec3 random_in_unit_disk(std::mt19937& rng) {
    while (true) {
        vec3 p(random_double(rng, -1, 1), random_double(rng, -1, 1), 0);
        if (p.length_squared() < 1) return p;
    }
}

// Reflect v about a surface with normal n (used by the Metal material).
inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

// Snell's-law refraction of unit vector uv through a surface with normal n
// and ratio of refractive indices etai_over_etat (used by Dielectric).
inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    double cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

// Uniformly samples a direction, in a local frame where +z points at a
// sphere's center, within the cone that sphere subtends (radius, at
// distance_squared away). This concentrates every sample on the visible
// disc of the sphere instead of wasting samples on directions that can't
// possibly hit it - the standard solid-angle technique for sampling a
// spherical light (see sphere.h's pdf_value()/random()).
inline vec3 random_to_sphere(double radius, double distance_squared, std::mt19937& rng) {
    double r1 = random_double(rng);
    double r2 = random_double(rng);
    double z = 1 + r2 * (std::sqrt(1 - radius * radius / distance_squared) - 1);

    double phi = 2 * pi * r1;
    double sin_theta = std::sqrt(1 - z * z);
    double x = std::cos(phi) * sin_theta;
    double y = std::sin(phi) * sin_theta;

    return vec3(x, y, z);
}
