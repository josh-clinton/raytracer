# Ray Tracing & Neural Rendering: A Learning Roadmap

Built while preparing to apply to AMD's *Software Development Engineer -
Ray Tracing / Neural Rendering* role. This maps the fundamentals I've
covered (with a working project, see `README.md`) and lays out the path to
the more advanced topics the role actually centers on — real-time
spatiotemporal resampling (ReSTIR) and neural rendering.

## 1. Foundations (covered — see the accompanying path tracer)

**Vectors, rays, and the camera.** A ray is `origin + t * direction`; a
camera is just a rule for turning a pixel coordinate into a ray. Everything
downstream — intersection, shading, sampling — operates on this one
primitive.

**Ray-primitive intersection.** Ray-sphere intersection reduces to a
quadratic. The same pattern (parametrize, substitute, solve) extends to
triangles (Möller–Trumbore), planes, and analytic quadrics. Production
ray tracers spend enormous engineering effort here because it's the
innermost loop of the whole algorithm.

**The rendering equation & Monte Carlo integration.** Global illumination
is defined by an integral (outgoing radiance = emitted + integral of
incoming radiance × BRDF × cosine term over the hemisphere) with no
closed-form solution for arbitrary scenes. Path tracing solves it by
random sampling: trace a ray, at each bounce randomly pick a new direction
weighted by the surface's BRDF, and average many such paths per pixel.
More samples → less noise, at a cost that's the central tension in the
whole field.

**BRDFs / materials.** Lambertian (ideal diffuse, cosine-weighted
scatter), specular/metal (mirror reflection + roughness), and dielectric
(Snell's law refraction + Fresnel reflectance via Schlick's approximation)
cover most surfaces you'll see in a first renderer. Physically based
rendering (PBR) generalizes this to microfacet models (GGX/Trowbridge-Reitz)
used throughout modern game and film rendering.

**Acceleration structures.** A BVH (bounding volume hierarchy) turns
per-ray intersection cost from O(N) objects to roughly O(log N) by
recursively partitioning space. This is conceptually the same structure
hardware ray tracing units (AMD RDNA ray accelerators, NVIDIA RT cores)
traverse — the difference is that hardware built it into silicon and
exposed it through APIs (DXR, Vulkan RT) instead of a CPU tree walk. My
project measures a **6.24× speedup** from adding a BVH on a 487-object
scene — small numbers here, but the scaling only gets more dramatic as
scene complexity grows.

**Resources used:**
- *Ray Tracing in One Weekend* series (Peter Shirley) — the standard first
  path for exactly this material; my project follows its scope (not its
  code) closely.
- *Physically Based Rendering: From Theory to Implementation* (Pharr,
  Jakob, Humphreys) — the deeper reference for everything above, freely
  readable at pbr-book.org.
- Scratchapixel.com — good visual/geometric intuition for intersection math.

## 2. Real-time ray tracing (next — what the role actually does day to day)

Offline path tracers can burn thousands of samples per pixel over minutes.
Real-time ray tracing has a budget of **1-4 samples per pixel at 60+ fps**,
so the entire field is about getting a usably noisy image with almost no
samples and then cleaning it up:

- **Spatiotemporal resampling (ReSTIR).** Instead of drawing fresh
  light-sampling candidates every frame, ReSTIR reuses and resamples
  "good" light samples from neighboring pixels (spatial) and previous
  frames (temporal) via weighted reservoir sampling, dramatically reducing
  variance per sample. This is the single most important recent algorithm
  in real-time GI and is named explicitly in the job posting — understanding
  the original paper (Bitterli et al., *Spatiotemporal Reservoir Resampling
  for Real-Time Ray Tracing*, SIGGRAPH 2020) and its follow-ups (ReSTIR GI,
  ReSTIR PT) is the highest-leverage reading for this role.
- **Denoising.** Even with ReSTIR, 1-4 spp images are noisy; a
  spatiotemporal or learned denoiser (e.g. SVGF, or a small neural network)
  filters that noise using auxiliary buffers (normals, depth, albedo) to
  preserve detail edges.
- **Hardware ray tracing APIs.** DirectX Raytracing (DXR) and Vulkan Ray
  Tracing expose BVH build/traversal, ray generation, closest-hit/miss
  shaders as a programmable pipeline on top of fixed-function traversal
  hardware. Porting my CPU path tracer's structure (ray gen → intersect →
  shade → scatter) onto this pipeline, in HLSL/GLSL compute or ray shaders,
  is the natural next project.

**Planned next project increment:** port the scatter loop to a compute
shader (or DXR hit-shader pipeline) and implement a minimal ReSTIR direct-
lighting pass on a simple multi-light scene, comparing noise levels at
equal sample counts with and without spatiotemporal reuse.

## 3. Neural rendering (the other half of the role)

"Neural rendering" spans a few distinct ideas worth telling apart:

- **Neural scene representations (NeRF).** A small MLP is trained to map a
  5D input (3D position + 2D viewing direction) to color + density; novel
  views are synthesized by volume-rendering rays through the trained
  network. Mildenhall et al., *NeRF: Representing Scenes as Neural
  Radiance Fields for View Synthesis* (ECCV 2020), is the foundational
  paper.
- **3D Gaussian Splatting.** Replaces the implicit MLP with an explicit set
  of anisotropic 3D Gaussians (position, covariance, color, opacity)
  optimized directly and rendered by fast differentiable rasterization —
  the current state of the art for real-time novel-view synthesis. Kerbl
  et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*
  (SIGGRAPH 2023).
- **Neural denoising / neural supersampling.** Applying learned models
  inside the classic rendering pipeline rather than replacing it — e.g. a
  small network that denoises a 1-spp path-traced frame, or upsamples a
  lower-resolution render (the category AMD's FSR and NVIDIA's DLSS occupy
  as products). This is likely the most directly relevant neural-rendering
  work for a rendering-team SDE role, since it plugs straight into the ray
  tracing pipeline built in part 1-2.
- **AI-assisted rendering code.** The posting explicitly lists experience
  with Cursor/Claude Code/Codex for rendering code as a plus — this project
  itself (built with Claude Code / Claude in Cowork) is a small existing
  data point for that.

**Planned next project increment:** train a tiny NeRF (or reproduce a
minimal Gaussian Splatting demo) on a synthetic multi-view dataset rendered
*by this same path tracer* — reusing the camera/ray infrastructure already
built, which is a natural bridge between the two halves of the role.

## 4. How this maps to the job posting's "Preferred Experience" bullets

| JD bullet | Status after this project | Next step |
|---|---|---|
| Expert C/C++ | Demonstrated: multithreaded C++17, RAII, polymorphic material/hittable interfaces, zero warnings under `-Wall -Wextra` | Keep building in C++; add GPU-side HLSL/GLSL |
| Real-time rendering, ray tracing, PBR, Monte Carlo, GI | Fundamentals demonstrated end-to-end | Move from offline (CPU, many spp) to real-time (GPU, 1-4 spp + reuse) |
| GPU-accelerated algorithms, modern graphics APIs, shader languages | Not yet started | DXR/Vulkan RT port of this renderer |
| Software engineering, data structures, algorithms, performance optimization | Demonstrated: BVH, measured 6.24× speedup, multithreading | Profile with a real GPU profiler (RGP, Nsight) once ported |
| Profiling, debugging, optimizing graphics workloads | Started: CPU timing + benchmark mode | GPU timeline profiling after GPU port |
| ReSTIR, path tracing, sampling, real-time ray tracing research | Path tracing + sampling demonstrated; ReSTIR not yet implemented | Read Bitterli et al. 2020; implement minimal ReSTIR DI |
| Neural rendering / AI coding tools | Not yet started; tool experience (this project) is a data point | Small NeRF/Gaussian Splatting reproduction |

## 5. Suggested reading order

1. *Ray Tracing in One Weekend* (+ *...the Next Week*, *...Rest of Your Life*) — Shirley
2. *PBR Book*, chapters on Monte Carlo integration and BSDFs — Pharr/Jakob/Humphreys
3. Bitterli et al., *Spatiotemporal Reservoir Resampling* (ReSTIR), SIGGRAPH 2020
4. Mildenhall et al., *NeRF*, ECCV 2020
5. Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023
6. AMD GPUOpen ray tracing samples/documentation, and the DXR or Vulkan RT
   spec, once ready to move off CPU
