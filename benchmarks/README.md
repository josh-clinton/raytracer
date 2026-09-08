# Benchmark sweep

`sweep.py` runs the renderer's `--bench` mode across four axes - image size,
object count, camera orbit angle, and material mix - measuring the BVH's
speedup over a flat linear scan at each point. Each axis is swept one value
at a time while the other three stay at a fixed baseline, so what changes
between runs is isolated to the one thing being measured.

```bash
# from the repo root, after building the project:
cd benchmarks
python3 sweep.py                    # full sweep - several minutes
python3 sweep.py --quick            # fast smoke test, low quality, ~1 minute
python3 sweep.py --repeats 5        # more repeats per config = tighter error bars
```

Needs `matplotlib`, `pandas`, and `Pillow` (`pip install matplotlib pandas pillow`).

## What it does

For each configuration, `sweep.py` shells out to the same `raytracer`
binary you'd use to render normally, in two special CLI modes added for
this:

- **`--bench`** builds the scene twice from the same seed - once flat, once
  BVH-wrapped - times both renders, and prints one CSV row (object count,
  every config parameter, both times, and the ratio). This is what the
  sweep is actually measuring.
- **`--scene`** renders one real (BVH-accelerated) image for a given
  config, used to grab a handful of representative preview images per
  dimension - not just timing numbers.

Each axis holds the other three at a shared baseline (300px wide, 32
samples/pixel, depth 8, `grid_radius` 11 → ~487 objects, 20° field of view,
13° camera angle, the original 80/15/5 diffuse/metal/glass mix) and sweeps
just one:

| Axis | What varies | Held at baseline |
|---|---|---|
| `image_size` | render width 100→600px | everything else |
| `object_count` | `grid_radius` 2→14 (≈16→650 objects) | everything else |
| `camera_angle` | orbit angle 0°→315° around the scene | everything else |
| `material_mix` | diffuse/metal/glass split, five presets | everything else |

Every config runs `--repeats` times (default 3) with a different random
seed each time, so the charts show mean ± standard deviation rather than a
single noisy sample.

## Output

```
results/raw_results.csv    every individual run, one row each
results/summary.csv        mean/std per swept value, across dimensions
charts/*.png                one two-panel chart per axis (render time + speedup)
gallery/*.png                a few actual rendered scenes from interesting configs
RESULTS.md                   the charts + a speedup table, generated fresh each run
```

`RESULTS.md` and everything under `results/`, `charts/`, and `gallery/` are
regenerated on every run - they're not meant to be hand-edited, and results
will differ by machine (this is timing-sensitive, so don't expect your
numbers to match anyone else's exactly, including the ones checked into
this repo from whatever machine last ran it).

# MSE sweep (direct-lighting quality, not speed)

A separate, unrelated sweep: `mse_sweep.py` (driving `mse_sweep.cpp`)
measures direct-lighting *quality* - MSE against a high-spp reference - for
`camera.h`'s three direct-lighting configurations, across scenes with
different light counts and at different samples-per-pixel budgets:

- **baseline** - `light_candidates=1`, `spatial_neighbors=0`: mathematically
  the pre-RIS behavior, a single blind uniform light pick per shading point.
- **reservoir** - `light_candidates=4`, `spatial_neighbors=0`: RIS/weighted
  reservoir sampling only.
- **spatial** - `light_candidates=4`, `spatial_neighbors=4`: RIS plus
  spatial reuse of neighboring pixels' reservoirs.

```bash
# from the repo root, after building the project (see above):
cd benchmarks
python3 mse_sweep.py --quick                                          # fast smoke test
python3 mse_sweep.py --light-counts 1,4,12,32 --spp 4,16,64 --repeats 3   # a real run, a few minutes
```

Each light count gets its own scene (a field of diffuse/metal/glass
spheres, same as the project's main scene, plus a ring of that many small
emissive spheres) and its own reference render (spatial reuse forced off,
so ground truth doesn't inherit its documented small-bias simplification -
see `spatially_combined_reservoir()` in `camera.h`). Every test render at
that light count reuses the same scene seed as its reference, so geometry
is pixel-identical across configs; repeats still get independent Monte
Carlo noise, since `camera.h` seeds its per-thread RNG from system entropy,
not from the scene seed.

## Output

```
results/mse_raw.csv         every individual run (one row per repeat)
results/mse_summary.csv     mean/std MSE per (light_count, spp, config)
results/ref_cache/*.ppm     cached reference renders, reused across runs
charts/mse_vs_spp.png       one panel per light count, MSE vs. spp
charts/mse_vs_lights.png    one panel per spp, MSE vs. light count
MSE_RESULTS.md              the charts + a summary table, generated fresh each run
```

`results/ref_cache/` is keyed by light count, width, and ref spp, so a
second run with the same settings skips re-rendering references - delete
that folder to force a fresh reference. As with `sweep.py`, everything
under `results/`, `charts/`, and `MSE_RESULTS.md` is regenerated on every
run and will differ by machine and by scene RNG draw.

### Reading the result

At 1 light, all three configs land on top of each other - there's only one
light to pick, so neither RIS nor spatial reuse has anything to do. As
light count grows, `reservoir`'s MSE advantage over `baseline` grows with
it (a 3-4 spot check: roughly even at 1-4 lights, ~1.1-1.3x lower MSE at 12
lights, ~1.3-2x lower at 32 lights across the sampled spp range) - the
larger the light pool, the worse a blind uniform pick gets, and the more
RIS's scored candidates pay off. `spatial`'s further improvement over
`reservoir` alone is real but modest and noisier in this scene, consistent
with the spatial-reuse commit's own note that a dense field of small,
differently-shaded spheres trips the normal/depth reject checks often,
correctly refusing to reuse across nearby discontinuities.
