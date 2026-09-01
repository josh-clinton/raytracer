// simplest_raytracer.cpp
//
// The absolute minimum ray tracer: one file, one sphere, ONE ray per pixel,
// no materials, no bounces. It answers exactly one question per pixel:
// "does the ray through this pixel hit the sphere, yes or no?" - and colors
// it accordingly. Everything else in a real ray tracer (materials, bounces,
// many samples per pixel, acceleration structures) is refinement on top of
// this one idea. Compile and run this first; the bigger project is what you
// get once you start asking "ok, but what color *exactly*, and what happens
// after it hits?"
//
// Build:   g++ -std=c++17 -O2 simplest_raytracer.cpp -o simplest_raytracer
// Run:     ./simplest_raytracer > image.ppm
// (On Windows/MSVC:  cl /EHsc /std:c++17 simplest_raytracer.cpp)

#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------
// STEP 0: a 3D vector. This is the only "data structure" we need. A point
// in space, a direction, and an RGB color are all just three numbers - we
// reuse the same type for all three so we don't need three separate types.
// ---------------------------------------------------------------------
struct vec3 {
    double x, y, z;

    vec3 operator+(const vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    vec3 operator-(const vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    vec3 operator*(double t)      const { return {x * t, y * t, z * t}; }

    double dot(const vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    double length()            const { return std::sqrt(dot(*this)); }
    vec3   normalized()        const { double len = length(); return {x / len, y / len, z / len}; }
};

// ---------------------------------------------------------------------
// STEP 1: a ray. Just a starting point (origin) and a direction. Walking
// along it means computing origin + t * direction for increasing t.
// ---------------------------------------------------------------------
struct ray {
    vec3 origin;
    vec3 direction;

    vec3 at(double t) const { return origin + direction * t; }
};

// ---------------------------------------------------------------------
// STEP 2: does this ray hit a given sphere?
//
// A point P is ON the sphere's surface when |P - center| = radius, i.e.
//   |P - center|^2 = radius^2
// Substitute P = ray.at(t) = origin + t*direction and expand - you get a
// plain quadratic equation in t:  a*t^2 + b*t + c = 0
// (This is the ONLY piece of "3D math" in this whole file. Everything
// else is bookkeeping.)
//
// Returns the smallest positive t where the ray hits the sphere, or -1.0
// if it misses entirely.
// ---------------------------------------------------------------------
double hit_sphere(const vec3& center, double radius, const ray& r) {
    vec3 oc = r.origin - center;
    double a = r.direction.dot(r.direction);
    double b = 2.0 * oc.dot(r.direction);
    double c = oc.dot(oc) - radius * radius;

    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return -1.0;  // no real solutions -> the line never touches the sphere
    }
    // Smaller root = the nearer of the two intersection points
    // (a ray typically enters and exits a sphere - we want where it enters).
    return (-b - std::sqrt(discriminant)) / (2.0 * a);
}

// ---------------------------------------------------------------------
// STEP 3: given a ray, what color do we see?
//
// This is the "no bounces" version: check the one sphere in our world.
// Hit it -> solid red. Miss it -> a simple top-to-bottom sky gradient
// (blue at the top, white at the horizon), just so the image isn't blank.
// A real renderer replaces "solid red" with "ask the material what
// happens next and trace another ray" - that's the very next lesson.
// ---------------------------------------------------------------------
vec3 ray_color(const ray& r) {
    vec3 sphere_center = {0, 0, -1};
    double sphere_radius = 0.5;

    double t = hit_sphere(sphere_center, sphere_radius, r);
    if (t > 0.0) {
        return {1.0, 0.0, 0.0};  // hit -> solid red, no shading at all yet
    }

    // Missed the sphere: shade the background by how "up" the ray points.
    vec3 unit_direction = r.direction.normalized();
    double a = 0.5 * (unit_direction.y + 1.0);  // remap y from [-1,1] to [0,1]
    vec3 white = {1.0, 1.0, 1.0};
    vec3 sky_blue = {0.5, 0.7, 1.0};
    return white * (1.0 - a) + sky_blue * a;  // linear blend (a "lerp")
}

int main() {
    // ---------------------------------------------------------------
    // STEP 4: set up a camera and a virtual "viewport" rectangle floating
    // in front of it. Every pixel corresponds to one point on that
    // rectangle; the ray for that pixel goes from the camera through it.
    // ---------------------------------------------------------------
    const int image_width = 400;
    const int image_height = 225;  // 16:9

    double viewport_height = 2.0;
    double viewport_width = viewport_height * (double(image_width) / image_height);
    double focal_length = 1.0;  // distance from camera to the viewport

    vec3 camera_center = {0, 0, 0};

    // Vectors that span the viewport rectangle, left-to-right and top-to-bottom.
    vec3 viewport_u = {viewport_width, 0, 0};
    vec3 viewport_v = {0, -viewport_height, 0};  // negative: image rows go top->down

    vec3 pixel_delta_u = viewport_u * (1.0 / image_width);
    vec3 pixel_delta_v = viewport_v * (1.0 / image_height);

    // World-space position of the upper-left corner of the viewport, then
    // of pixel (0,0) specifically (nudged half a pixel in from the corner).
    vec3 viewport_upper_left = camera_center - vec3{0, 0, focal_length}
                              - viewport_u * 0.5 - viewport_v * 0.5;
    vec3 pixel00_loc = viewport_upper_left + (pixel_delta_u + pixel_delta_v) * 0.5;

    // ---------------------------------------------------------------
    // STEP 5: the actual render loop. One ray per pixel - no antialiasing,
    // no randomness, no averaging. It'll look a little jagged around the
    // sphere's edge; that jaggedness is exactly what shooting *many*
    // jittered rays per pixel (instead of one) fixes.
    // ---------------------------------------------------------------
    std::printf("P3\n%d %d\n255\n", image_width, image_height);  // PPM header (ASCII this time, for readability)

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
