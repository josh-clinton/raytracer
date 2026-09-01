#!/usr/bin/env python3
"""benchmarks/sweep.py

Runs the raytracer's --bench mode across four axes - image size, object
count, camera orbit angle, and material mix - and produces:

  benchmarks/results/raw_results.csv     every individual run
  benchmarks/results/summary.csv         mean/std speedup per swept value
  benchmarks/charts/<dimension>.png      one chart per axis
  benchmarks/gallery/<name>.png          a handful of actual rendered scenes
  benchmarks/RESULTS.md                  a short written summary

Each axis is swept one value at a time while holding the other three at a
fixed baseline (see BASELINE below), so what changes between runs is
isolated to the one thing being measured.

Usage:
    python3 benchmarks/sweep.py                  # full sweep (several minutes)
    python3 benchmarks/sweep.py --quick           # fast smoke test, low quality
    python3 benchmarks/sweep.py --binary ../build/raytracer --repeats 5
"""
import argparse
import pathlib
import statistics
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Same two colors used throughout this project's other diagrams - validated
# for colorblind-safe separation with dataviz's palette validator
# (ΔE 21.2 protan / 26.7 normal-vision), not just picked to look nice:
#   linear scan  = amber  #c06a2c  (the "expensive" baseline)
#   BVH          = blue   #2f6fb5  (the accelerated version)
COLOR_FLAT = "#c06a2c"
COLOR_BVH = "#2f6fb5"
GRID_COLOR = "#dddddd"

CSV_COLUMNS = [
    "width", "height", "spp", "depth", "grid_radius", "object_count", "vfov",
    "cam_angle_deg", "diffuse_frac", "metal_frac", "glass_frac", "seed",
    "flat_seconds", "bvh_seconds", "speedup",
]

BASELINE = dict(width=300, spp=32, depth=8, grid_radius=11, vfov=20.0, cam_angle=13.0,
                 diffuse_pct=80, metal_pct=15)

MATERIAL_MIXES = {
    (100, 0): "100% diffuse",
    (0, 100): "100% metal",
    (0, 0): "100% glass",
    (33, 33): "even mix",
    (80, 15): "default (80/15/5)",
}

SWEEPS = {
    "image_size": dict(
        title="Speedup vs. image size",
        xlabel="image width (px)",
        values=[100, 150, 200, 300, 450, 600],
        apply=lambda cfg, v: cfg.update(width=v),
        label=lambda v: str(v),
        numeric=lambda v: v,
    ),
    "object_count": dict(
        title="Speedup vs. object count",
        xlabel="objects in scene",
        values=[2, 4, 6, 8, 11, 14],
        apply=lambda cfg, v: cfg.update(grid_radius=v),
        label=lambda v: f"r={v}",
        numeric=None,  # filled in from the actual measured object_count column
    ),
    "camera_angle": dict(
        title="Speedup vs. camera orbit angle",
        xlabel="camera angle (degrees)",
        values=[0, 45, 90, 135, 180, 225, 270, 315],
        apply=lambda cfg, v: cfg.update(cam_angle=v),
        label=lambda v: f"{v}°",
        numeric=lambda v: v,
    ),
    "material_mix": dict(
        title="Speedup vs. material mix",
        xlabel=None,
        values=list(MATERIAL_MIXES.keys()),
        apply=lambda cfg, v: cfg.update(diffuse_pct=v[0], metal_pct=v[1]),
        label=lambda v: MATERIAL_MIXES[v],
        numeric=None,  # categorical - bar chart, not a line
    ),
}


def run_bench(binary, cfg, seed):
    args = [
        str(binary), "--bench", str(cfg["width"]), str(cfg["spp"]), str(cfg["depth"]),
        str(cfg["grid_radius"]), str(cfg["vfov"]), str(cfg["cam_angle"]),
        str(cfg["diffuse_pct"]), str(cfg["metal_pct"]), str(seed),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    last_line = result.stdout.strip().splitlines()[-1]
    return dict(zip(CSV_COLUMNS, last_line.split(",")))


def render_scene(binary, cfg, seed, out_png):
    ppm_path = out_png.with_suffix(".ppm")
    args = [
        str(binary), "--scene", str(cfg["width"]), str(cfg["spp"]), str(cfg["depth"]),
        str(cfg["grid_radius"]), str(cfg["vfov"]), str(cfg["cam_angle"]),
        str(cfg["diffuse_pct"]), str(cfg["metal_pct"]), str(seed), str(ppm_path),
    ]
    subprocess.run(args, capture_output=True, text=True, check=True)
    Image.open(ppm_path).save(out_png)
    ppm_path.unlink()


def sweep_dimension(binary, name, spec, repeats, seed_base):
    rows = []
    for i, val in enumerate(spec["values"]):
        cfg = dict(BASELINE)
        spec["apply"](cfg, val)
        speedups = []
        for r in range(repeats):
            seed = seed_base + i * 1000 + r
            row = run_bench(binary, cfg, seed)
            row["dimension"] = name
            row["swept_label"] = spec["label"](val)
            row["swept_order"] = i
            rows.append(row)
            speedups.append(float(row["speedup"]))
        print(f"  [{name:14s}] {spec['label'](val):<20s} "
              f"mean speedup {statistics.mean(speedups):5.2f}x  (n={repeats})")
    return rows


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#444444", labelsize=9)


def plot_line_dimension(df, name, spec, out_dir):
    agg = df.groupby("swept_order").agg(
        swept_label=("swept_label", "first"),
        x=("swept_order", "first"),
        object_count=("object_count", "mean"),
        flat_mean=("flat_seconds", "mean"), flat_std=("flat_seconds", "std"),
        bvh_mean=("bvh_seconds", "mean"), bvh_std=("bvh_seconds", "std"),
        speedup_mean=("speedup", "mean"), speedup_std=("speedup", "std"),
    ).reset_index(drop=True)

    if spec["numeric"] is not None:
        xvals = [spec["numeric"](v) for v in spec["values"]]
    else:
        xvals = agg["object_count"].tolist()  # object_count sweep: use the real measured count

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(spec["title"], fontsize=13, fontweight="bold", color="#14181d")

    ax1.errorbar(xvals, agg["flat_mean"], yerr=agg["flat_std"], color=COLOR_FLAT,
                 marker="o", markersize=6, linewidth=2, capsize=3, label="linear scan (no BVH)")
    ax1.errorbar(xvals, agg["bvh_mean"], yerr=agg["bvh_std"], color=COLOR_BVH,
                 marker="o", markersize=6, linewidth=2, capsize=3, label="BVH")
    ax1.set_yscale("log")
    ax1.set_ylabel("render time (s, log scale)", fontsize=9, color="#5b6570")
    ax1.set_xlabel(spec["xlabel"] or "object count", fontsize=9, color="#5b6570")
    ax1.set_title("render time", fontsize=10, color="#5b6570")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    style_axes(ax1)

    ax2.errorbar(xvals, agg["speedup_mean"], yerr=agg["speedup_std"], color=COLOR_BVH,
                 marker="o", markersize=6, linewidth=2, capsize=3)
    ax2.set_ylabel("speedup (×)", fontsize=9, color="#5b6570")
    ax2.set_xlabel(spec["xlabel"] or "object count", fontsize=9, color="#5b6570")
    ax2.set_title("BVH speedup", fontsize=10, color="#5b6570")
    last_x, last_y = xvals[-1], agg["speedup_mean"].iloc[-1]
    ax2.annotate(f"{last_y:.1f}×", (last_x, last_y), textcoords="offset points",
                 xytext=(6, 6), fontsize=9, color=COLOR_BVH, fontweight="bold")
    style_axes(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path, agg


def plot_material_mix(df, spec, out_dir):
    agg = df.groupby("swept_order").agg(
        swept_label=("swept_label", "first"),
        flat_mean=("flat_seconds", "mean"), flat_std=("flat_seconds", "std"),
        bvh_mean=("bvh_seconds", "mean"), bvh_std=("bvh_seconds", "std"),
        speedup_mean=("speedup", "mean"), speedup_std=("speedup", "std"),
    ).reset_index(drop=True)

    labels = agg["swept_label"].tolist()
    x = range(len(labels))
    width = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(spec["title"], fontsize=13, fontweight="bold", color="#14181d")

    ax1.bar([i - width / 2 for i in x], agg["flat_mean"], width, yerr=agg["flat_std"],
            color=COLOR_FLAT, label="linear scan (no BVH)", capsize=3)
    ax1.bar([i + width / 2 for i in x], agg["bvh_mean"], width, yerr=agg["bvh_std"],
            color=COLOR_BVH, label="BVH", capsize=3)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("render time (s)", fontsize=9, color="#5b6570")
    ax1.set_title("render time", fontsize=10, color="#5b6570")
    ax1.legend(frameon=False, fontsize=8.5)
    style_axes(ax1)

    bars = ax2.bar(list(x), agg["speedup_mean"], yerr=agg["speedup_std"], color=COLOR_BVH,
                    capsize=3, width=0.55)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylabel("speedup (×)", fontsize=9, color="#5b6570")
    ax2.set_title("BVH speedup", fontsize=10, color="#5b6570")
    for b, v, e in zip(bars, agg["speedup_mean"], agg["speedup_std"].fillna(0)):
        ax2.annotate(f"{v:.1f}×", (b.get_x() + b.get_width() / 2, v + e), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=8.5, color=COLOR_BVH, fontweight="bold")
    style_axes(ax2)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = out_dir / "material_mix.png"
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    return out_path, agg


def write_results_md(summaries, out_path, repeats):
    lines = [
        "# Benchmark sweep results",
        "",
        f"Generated by `benchmarks/sweep.py` ({repeats} repeat(s) per configuration; "
        "each repeat uses a different random seed, so numbers below are means).",
        "",
    ]
    for name, (chart_path, agg) in summaries.items():
        title = SWEEPS[name]["title"]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({chart_path.relative_to(chart_path.parents[1])})")
        lines.append("")
        lines.append("| " + " | ".join(agg["swept_label"]) + " |")
        lines.append("|" + "---|" * len(agg))
        lines.append("| " + " | ".join(f"{v:.2f}×" for v in agg["speedup_mean"]) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=None, help="path to the raytracer binary (default: ../build/raytracer)")
    parser.add_argument("--repeats", type=int, default=3, help="runs per configuration (default: 3)")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--quick", action="store_true",
                         help="fast smoke test: fewer values, 1 repeat, low quality")
    parser.add_argument("--skip-gallery", action="store_true", help="skip rendering showcase images")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    binary = pathlib.Path(args.binary) if args.binary else here.parent / "build" / "raytracer"
    if not binary.exists():
        sys.exit(f"raytracer binary not found at {binary}\n"
                  f"Build it first: mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .")

    repeats = args.repeats
    global BASELINE
    if args.quick:
        repeats = 1
        BASELINE = dict(BASELINE, spp=12, depth=6)
        for spec in SWEEPS.values():
            spec["values"] = spec["values"][:3] if spec is not SWEEPS["material_mix"] else spec["values"]
        SWEEPS["image_size"]["values"] = [100, 200, 300]
        SWEEPS["object_count"]["values"] = [3, 6, 11]
        SWEEPS["camera_angle"]["values"] = [0, 120, 240]

    results_dir = here / "results"
    charts_dir = here / "charts"
    gallery_dir = here / "gallery"
    for d in (results_dir, charts_dir, gallery_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Using binary: {binary}")
    print(f"Repeats per config: {repeats}  |  quick mode: {args.quick}\n")

    all_rows = []
    for name, spec in SWEEPS.items():
        print(f"Sweeping {name} ({spec['title']})...")
        all_rows += sweep_dimension(binary, name, spec, repeats, args.seed_base)
        print()

    df = pd.DataFrame(all_rows)
    for col in ["width", "height", "spp", "depth", "grid_radius", "object_count", "swept_order", "seed"]:
        df[col] = df[col].astype(int)
    for col in ["vfov", "cam_angle_deg", "diffuse_frac", "metal_frac", "glass_frac",
                "flat_seconds", "bvh_seconds", "speedup"]:
        df[col] = df[col].astype(float)
    df.to_csv(results_dir / "raw_results.csv", index=False)
    print(f"Wrote {len(df)} raw rows -> {results_dir / 'raw_results.csv'}")

    summaries = {}
    for name, spec in SWEEPS.items():
        dim_df = df[df["dimension"] == name]
        if name == "material_mix":
            summaries[name] = plot_material_mix(dim_df, spec, charts_dir)
        else:
            summaries[name] = plot_line_dimension(dim_df, name, spec, charts_dir)
        print(f"Wrote {summaries[name][0]}")

    all_summaries = pd.concat(
        [agg.assign(dimension=name) for name, (_, agg) in summaries.items()], ignore_index=True)
    all_summaries.to_csv(results_dir / "summary.csv", index=False)

    write_results_md(summaries, here / "RESULTS.md", repeats)
    print(f"\nWrote {here / 'RESULTS.md'}")

    if not args.skip_gallery:
        print("\nRendering gallery showcase images...")
        gallery_spp = 24 if args.quick else 80
        gallery_depth = 8 if args.quick else 12
        showcase = [
            ("wide_field", dict(BASELINE, width=500 if not args.quick else 200, grid_radius=14, spp=gallery_spp, depth=gallery_depth), 42),
            ("all_glass", dict(BASELINE, diffuse_pct=0, metal_pct=0, spp=gallery_spp, depth=gallery_depth), 7),
            ("all_metal", dict(BASELINE, diffuse_pct=0, metal_pct=100, spp=gallery_spp, depth=gallery_depth), 7),
            ("angle_225", dict(BASELINE, cam_angle=225, spp=gallery_spp, depth=gallery_depth), 99),
        ]
        for label, cfg, seed in showcase:
            out_png = gallery_dir / f"{label}.png"
            render_scene(binary, cfg, seed, out_png)
            print(f"  wrote {out_png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
