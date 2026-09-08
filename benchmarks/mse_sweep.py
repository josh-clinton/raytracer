#!/usr/bin/env python3
"""benchmarks/mse_sweep.py

Measures direct-lighting quality (MSE vs. a high-spp reference) for three
camera.h configurations - baseline (pre-RIS uniform-pick NEE), reservoir
(RIS only), and spatial (RIS + spatial reuse) - across scenes with
different light counts and at different samples-per-pixel budgets. Produces:

  benchmarks/results/mse_raw.csv       every individual run (one per repeat)
  benchmarks/results/mse_summary.csv   mean/std MSE per (light_count, spp, config)
  benchmarks/charts/mse_vs_spp.png     one panel per light count, MSE vs. spp
  benchmarks/charts/mse_vs_lights.png  one panel per spp, MSE vs. light count
  benchmarks/MSE_RESULTS.md            a short written summary

For each light count, one reference image is rendered once (spatial reuse
forced off - see mse_sweep.cpp) and reused for every spp/config/repeat at
that light count, so what's being measured is purely the technique and
sample count, not different ground truths. Test renders reuse the same
scene seed as their light count's reference (so geometry matches pixel for
pixel) but pick up independent Monte Carlo noise each repeat, since
camera.h's per-thread RNG is seeded from system entropy, not from that seed
- see mse_sweep.cpp's header comment.

Usage:
    python3 benchmarks/mse_sweep.py                  # full sweep - can take a while
    python3 benchmarks/mse_sweep.py --quick           # fast smoke test
    python3 benchmarks/mse_sweep.py --light-counts 1,4,12,32 --spp 4,16,64,128
    python3 benchmarks/mse_sweep.py --binary ../build/mse_sweep --repeats 5
"""
import argparse
import os
import pathlib
import statistics
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Three configs compared at equal spp, at every (light_count, spp) point -
# see mse_sweep.cpp's header comment for exactly what each one does.
#   baseline   light_candidates=1, spatial_neighbors=0 - pre-RIS behavior
#   reservoir  light_candidates=N, spatial_neighbors=0 - RIS only
#   spatial    light_candidates=N, spatial_neighbors=K - RIS + spatial reuse
# Colors follow this project's existing amber/blue pair (colorblind-validated
# - see sweep.py) for baseline/reservoir, plus a third, distinguishable green
# for spatial; markers also differ per line so the distinction doesn't rely
# on color alone.
CONFIGS = {
    "baseline": dict(color="#c06a2c", marker="o", label="baseline (1 candidate, no spatial reuse)"),
    "reservoir": dict(color="#2f6fb5", marker="s", label="reservoir (RIS)"),
    "spatial": dict(color="#3c9a5f", marker="^", label="reservoir + spatial reuse"),
}
GRID_COLOR = "#dddddd"

CSV_COLUMNS = [
    "light_count", "spp", "config", "candidates", "spatial_neighbors",
    "spatial_radius", "seed", "mse", "seconds",
]


def scene_seed(seed_base, light_count):
    # One fixed seed per light count, shared by the reference and every test
    # render at that light count, so they all render the identical scene
    # geometry - the only thing allowed to differ is the lighting technique.
    return seed_base + light_count


def render_reference(binary, results_dir, cfg, light_count):
    ppm_path = results_dir / "ref_cache" / f"ref_L{light_count}_w{cfg['width']}_spp{cfg['ref_spp']}.ppm"
    ppm_path.parent.mkdir(parents=True, exist_ok=True)
    if ppm_path.exists():
        return ppm_path
    args = [
        str(binary), "--ref", str(cfg["width"]), str(cfg["depth"]), str(cfg["ref_spp"]),
        str(cfg["ref_candidates"]), str(cfg["threads"]), str(scene_seed(cfg["seed_base"], light_count)),
        str(light_count), str(ppm_path),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    print(f"  {result.stderr.strip()}")
    return ppm_path


def render_test(binary, cfg, light_count, spp, config_name, ref_ppm):
    spec = cfg["config_specs"][config_name]
    args = [
        str(binary), "--test", str(cfg["width"]), str(cfg["depth"]), str(spp),
        str(spec["candidates"]), str(spec["spatial_neighbors"]), str(cfg["spatial_radius"]),
        str(cfg["threads"]), str(scene_seed(cfg["seed_base"], light_count)), str(light_count),
        config_name, str(ref_ppm),
    ]
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    last_line = result.stdout.strip().splitlines()[-1]
    return dict(zip(CSV_COLUMNS, last_line.split(",")))


def run_sweep(binary, results_dir, cfg, light_counts, spp_values, repeats):
    rows = []
    for light_count in light_counts:
        print(f"light_count={light_count}: rendering reference ({cfg['ref_spp']} spp)...")
        ref_ppm = render_reference(binary, results_dir, cfg, light_count)

        for spp in spp_values:
            for config_name in CONFIGS:
                mses = []
                for rep in range(repeats):
                    row = render_test(binary, cfg, light_count, spp, config_name, ref_ppm)
                    row["repeat"] = rep
                    rows.append(row)
                    mses.append(float(row["mse"]))
                print(f"  spp={spp:<5d} {config_name:<10s} mse mean={statistics.mean(mses):8.4f}"
                      + (f"  std={statistics.stdev(mses):.4f}" if repeats > 1 else "") + f"  (n={repeats})")
    return rows


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#444444", labelsize=8.5)


def plot_mse_vs_spp(df, light_counts, spp_values, out_path):
    n = len(light_counts)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Direct-lighting MSE vs. reference, by samples/pixel", fontsize=13,
                 fontweight="bold", color="#14181d")

    for ax, light_count in zip(axes, light_counts):
        sub = df[df["light_count"] == light_count]
        for config_name, style in CONFIGS.items():
            csub = sub[sub["config"] == config_name]
            agg = csub.groupby("spp")["mse"].agg(["mean", "std"]).reindex(spp_values)
            ax.errorbar(spp_values, agg["mean"], yerr=agg["std"].fillna(0), color=style["color"],
                        marker=style["marker"], markersize=5.5, linewidth=1.8, capsize=3,
                        label=style["label"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(spp_values)
        ax.set_xticklabels([str(v) for v in spp_values])
        ax.set_title(f"{light_count} light{'s' if light_count != 1 else ''}", fontsize=10, color="#5b6570")
        ax.set_xlabel("samples/pixel (log scale)", fontsize=8.5, color="#5b6570")
        if ax is axes[0]:
            ax.set_ylabel("MSE vs. reference (log scale)", fontsize=8.5, color="#5b6570")
        style_axes(ax)

    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_mse_vs_lights(df, light_counts, spp_values, out_path):
    n = len(spp_values)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle("Direct-lighting MSE vs. reference, by light count", fontsize=13,
                 fontweight="bold", color="#14181d")

    for ax, spp in zip(axes, spp_values):
        sub = df[df["spp"] == spp]
        for config_name, style in CONFIGS.items():
            csub = sub[sub["config"] == config_name]
            agg = csub.groupby("light_count")["mse"].agg(["mean", "std"]).reindex(light_counts)
            ax.errorbar(light_counts, agg["mean"], yerr=agg["std"].fillna(0), color=style["color"],
                        marker=style["marker"], markersize=5.5, linewidth=1.8, capsize=3,
                        label=style["label"])
        ax.set_yscale("log")
        ax.set_xticks(light_counts)
        ax.set_title(f"{spp} spp", fontsize=10, color="#5b6570")
        ax.set_xlabel("lights in scene", fontsize=8.5, color="#5b6570")
        if ax is axes[0]:
            ax.set_ylabel("MSE vs. reference (log scale)", fontsize=8.5, color="#5b6570")
        style_axes(ax)

    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def write_results_md(summary, light_counts, spp_values, repeats, out_path, chart1, chart2):
    lines = [
        "# Direct-lighting MSE sweep results",
        "",
        f"Generated by `benchmarks/mse_sweep.py` ({repeats} repeat(s) per configuration; "
        "each repeat is an independent Monte Carlo noise draw on identical scene geometry - "
        "see the script's docstring). Lower MSE is better (closer to the high-spp reference).",
        "",
        f"![MSE vs. samples per pixel]({chart1.name})",
        "",
        f"![MSE vs. light count]({chart2.name})",
        "",
        "## Mean MSE by configuration",
        "",
        "| lights | spp | baseline | reservoir | +spatial | reservoir vs. baseline | +spatial vs. reservoir |",
        "|---|---|---|---|---|---|---|",
    ]
    for light_count in light_counts:
        for spp in spp_values:
            row = summary[(summary["light_count"] == light_count) & (summary["spp"] == spp)]
            vals = {name: row[row["config"] == name]["mse_mean"].iloc[0] for name in CONFIGS
                    if not row[row["config"] == name].empty}
            if len(vals) < 3:
                continue
            base, res, spa = vals["baseline"], vals["reservoir"], vals["spatial"]
            res_ratio = base / res if res > 0 else float("inf")
            spa_ratio = res / spa if spa > 0 else float("inf")
            lines.append(f"| {light_count} | {spp} | {base:.4f} | {res:.4f} | {spa:.4f} "
                         f"| {res_ratio:.2f}x lower | {spa_ratio:.2f}x lower |")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default=None, help="path to the mse_sweep binary (default: ../build/mse_sweep)")
    parser.add_argument("--light-counts", default="1,4,12,32", help="comma-separated light counts to sweep")
    parser.add_argument("--spp", default="4,16,64,128", help="comma-separated samples-per-pixel budgets to sweep")
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--ref-spp", type=int, default=512)
    parser.add_argument("--ref-candidates", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=4, help="light_candidates for the reservoir/spatial configs")
    parser.add_argument("--spatial-neighbors", type=int, default=4)
    parser.add_argument("--spatial-radius", type=int, default=20)
    parser.add_argument("--threads", type=int, default=0, help="0 = let the binary pick (hardware_concurrency)")
    parser.add_argument("--repeats", type=int, default=3, help="independent renders per (light_count, spp, config)")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="fast smoke test: fewer values, 1 repeat, tiny image")
    args = parser.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    binary = pathlib.Path(args.binary) if args.binary else here.parent / "build" / "mse_sweep"
    if not binary.exists():
        sys.exit(f"mse_sweep binary not found at {binary}\n"
                  f"Build it first: mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .")

    light_counts = [int(v) for v in args.light_counts.split(",")]
    spp_values = [int(v) for v in args.spp.split(",")]
    repeats = args.repeats
    width, depth, ref_spp = args.width, args.depth, args.ref_spp

    if args.quick:
        light_counts = light_counts[:2] if len(light_counts) > 2 else light_counts
        spp_values = spp_values[:2] if len(spp_values) > 2 else spp_values
        repeats = 1
        width = min(width, 120)
        depth = min(depth, 6)
        ref_spp = min(ref_spp, 96)

    threads = args.threads if args.threads > 0 else (os.cpu_count() or 4)

    cfg = dict(
        width=width, depth=depth, ref_spp=ref_spp, ref_candidates=args.ref_candidates,
        threads=threads, seed_base=args.seed_base, spatial_radius=args.spatial_radius,
        config_specs={
            "baseline": dict(candidates=1, spatial_neighbors=0),
            "reservoir": dict(candidates=args.candidates, spatial_neighbors=0),
            "spatial": dict(candidates=args.candidates, spatial_neighbors=args.spatial_neighbors),
        },
    )

    results_dir = here / "results"
    charts_dir = here / "charts"
    for d in (results_dir, charts_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Using binary: {binary}")
    print(f"light_counts={light_counts}  spp={spp_values}  repeats={repeats}  "
          f"width={width}  ref_spp={ref_spp}  threads={threads}\n")

    rows = run_sweep(binary, results_dir, cfg, light_counts, spp_values, repeats)

    df = pd.DataFrame(rows)
    for col in ["light_count", "spp", "candidates", "spatial_neighbors", "spatial_radius", "seed", "repeat"]:
        df[col] = df[col].astype(int)
    for col in ["mse", "seconds"]:
        df[col] = df[col].astype(float)
    raw_path = results_dir / "mse_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nWrote {len(df)} raw rows -> {raw_path}")

    summary = df.groupby(["light_count", "spp", "config"]).agg(
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        seconds_mean=("seconds", "mean"),
    ).reset_index()
    summary_path = results_dir / "mse_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    chart1 = charts_dir / "mse_vs_spp.png"
    plot_mse_vs_spp(df, light_counts, spp_values, chart1)
    print(f"Wrote {chart1}")

    chart2 = charts_dir / "mse_vs_lights.png"
    plot_mse_vs_lights(df, light_counts, spp_values, chart2)
    print(f"Wrote {chart2}")

    results_md = here / "MSE_RESULTS.md"
    write_results_md(summary, light_counts, spp_values, repeats, results_md, chart1, chart2)
    print(f"Wrote {results_md}")

    print("\nDone.")


if __name__ == "__main__":
    main()
