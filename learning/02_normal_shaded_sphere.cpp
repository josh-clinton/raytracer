// normal_shaded_sphere.cpp
//
// Step 2 on top of simplest_raytracer.cpp. Only two things changed:
//   1. hit_sphere() now also returns *where* the ray hit (not just yes/no
//      as a t value) - we need the exact 3D hit point to compute a normal.
//   2. ray_color() no longer returns solid red for a hit. It computes the
//      surface normal at the hit point and colors the pixel BY that
//      normal's direction - which is what makes the sphere look like a
//      3D ball instead of a flat red disc.
//
// Build:   g++ -std=c++17 -O2 normal_shaded_sphere.cpp -o normal_shaded_sphere
// Run:     ./normal_shaded_sphere > image.ppm

#include <cmath>
#include <cstdio>

struct vec3 {
    double x, y, z;

    vec3 operator+(const vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    vec3 operator-(const vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    vec3 operator*(double t)      const { return {x * t, y * t, z * t}; }

    double dot(const vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    double length()            const { return std::sqrt(dot(*this)); }
    vec3   normalized()        const { double len = length(); return {x / len, y / len, z / len}; }
};

struct ray {
    vec3 origin;
    vec3 direction;

    vec3 at(double t) const { return origin + direction * t; }
};

// Unchanged from step 1 - still just solving the quadratic for where the
// ray's line touches the sphere's surface.
double hit_sphere(const vec3& center, double radius, const ray& r) {
    vec3 oc = r.origin - center;
    double a = r.direction.dot(r.direction);
    double b = 2.0 * oc.dot(r.direction);
    double c = oc.dot(oc) - radius * radius;

    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return -1.0;
    }
    return (-b - std::sqrt(discriminant)) / (2.0 * a);
}

// -----------------------------------------------------------------------
// NEW IDEA: the surface normal.
//
// At any point P on a sphere's surface, the normal (the direction
// pointing straight "outward", perpendicular to the surface) is simply
// the direction from the center to P:
//
//     normal = (P - center) / radius
//
// (dividing by radius just makes it length 1, i.e. a "unit" vector).
// This one vector is THE fundamental quantity in all of shading - every
// material in the full project (Lambertian, metal, glass) starts by
// asking "what's the normal here?" and goes from there. A flat-shaded
// sphere has no visual depth because it never asks this question; a
// normal-shaded sphere immediately looks like a solid 3D object because
// the color now changes smoothly across the surface, matching its curve.
// -----------------------------------------------------------------------
vec3 ray_color(const ray& r) {
    vec3 sphere_center = {0, 0, -1};
    double sphere_radius = 0.5;

    double t = hit_sphere(sphere_center, sphere_radius, r);
    if (t > 0.0) {
        vec3 hit_point = r.at(t);                                  // NEW: where exactly did we hit?
        vec3 normal = (hit_point - sphere_center) * (1.0 / sphere_radius);  // NEW: outward direction there

        // normal's x/y/z each range over [-1, 1] (it's a unit vector).
        // RGB values need to be in [0, 1], so remap: 0.5*(n + 1).
        // This is a common convention for *visualizing* normals as color -
        // it's not physically what a camera would see, but it's the
        // simplest way to make the normal itself visible, and it's
        // instantly recognizable to anyone who's touched a graphics
        // debugger (normals rendered this way always look like this
        // characteristic red/green/blue-tinted sphere).
        return vec3{normal.x + 1, normal.y + 1, normal.z + 1} * 0.5;
    }

    vec3 unit_direction = r.direction.normalized();
    double a = 0.5 * (unit_direction.y + 1.0);
    vec3 white = {1.0, 1.0, 1.0};
    vec3 sky_blue = {0.5, 0.7, 1.0};
    return white * (1.0 - a) + sky_blue * a;
}

int main() {
    const int image_width = 400;
    const int image_height = 225;

    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width) / image_height);
    double focal_length = 1.0;

    vec3 camera_center = {0, 0, 0};

    vec3 viewport_u = {viewport_width, 0, 0};
    vec3 viewport_v = {0, -viewport_height, 0};

    vec3 pixel_delta_u = viewport_u * (1.0 / image_width);
    vec3 pixel_delta_v = viewport_v * (1.0 / image_height);

    vec3 viewport_upper_left = camera_center - vec3{0, 0, focal_length}
                              - viewport_u * 0.5 - viewport_v * 0.5;
    vec3 pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

    std::printf("P3\n%d %d\n255\n", image_width, image_height);

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            vec3 pixel_center = pixel00_loc + pixel_delta_u * i + pixel_delta_v * j;
            vec3 ray_direction = pixel_center - camera_center;
            ray r{camera_center, ray_direction};

            vec3 color = ray_color(r);

            int ir = int(255.999 * color.x);
            int ig = int(255.999 * color.y);
            int ib = int(255.999 * color.z);
            std::printf("%d %d %d\n", ir, ig, ib);
        }
    }
    return 0;
}
