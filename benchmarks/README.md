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
