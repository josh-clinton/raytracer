# Learning materials

The `src/` renderer at the repo root is the finished project. Everything in
here is the work that got there — kept on purpose, not cleaned up, so the
progression is visible.

- **`01_simplest_raytracer.cpp`** — the absolute minimum: one sphere, one ray
  per pixel, no materials, no bounces. Just "does this ray hit the sphere,
  yes or no."
- **`02_normal_shaded_sphere.cpp`** — one step up: colors each pixel by the
  surface normal at the hit point, which is what makes the sphere first look
  like a solid 3D object instead of a flat disc.
- **`portfolio.html`** — a write-up of the finished project (what it does, the
  concepts it demonstrates, sample renders).
- **`formula_reference.html`** — every formula the renderer implements, in
  standard math notation, with a hoverable symbol legend and a worked numeric
  example + source-code pointer for each one.
- **`bvh_diagram.html`** — a diagrammed explanation of the bounding-volume
  hierarchy acceleration structure: what an AABB is, how the tree nests, and
  why walking it beats scanning every object.

The three `.html` files are self-contained — open any of them directly in a
browser, no build step or server needed.
