"""Run every SCALE-Sim config in configs/ on every layer of the selected topologies.

Order: layer by layer. For each layer, all 27 configs are run (in parallel),
then a per-layer summary is written and a "done" message is printed.

Output layout:
    results/<Model>/<Layer>/topology.csv              single-layer topology used
    results/<Model>/<Layer>/<config>/COMPUTE_REPORT.csv
    results/<Model>/<Layer>/<config>/BANDWIDTH_REPORT.csv
    results/<Model>/<Layer>/<config>/DETAILED_ACCESS_REPORT.csv
    results/<Model>/<Layer>/logs/<config>.log          SCALE-Sim console output
    results/<Model>/<Layer>/summary.csv                27 rows, one per config
    results/all_results.csv                            every layer x every config

Runs that already have a COMPUTE_REPORT.csv are skipped, so the script can be
stopped and restarted safely.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\run_all.py              # 4 workers (default)
    .venv\\Scripts\\python.exe scripts\\run_all.py --workers 4
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

MODELS = {
    "AlexNet":   ROOT / "topologies" / "alexnet_selected.csv",
    "ResNet-18": ROOT / "topologies" / "resnet18_selected.csv",
}

CFG_RE = re.compile(r"^(os|ws|is)_(\d+)x(\d+)_sram(\d+)kB$")


# --------------------------------------------------------------------------- helpers
def read_topology(path):
    """Return (header_line, [(layer_name, row_line), ...])."""
    lines = [l.rstrip("\n") for l in path.read_text().splitlines() if l.strip()]
    header, rows = lines[0], lines[1:]
    return header, [(r.split(",")[0].strip(), r) for r in rows]


def read_report(path):
    """Read a one-layer SCALE-Sim report CSV into a dict {column: value}."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f, skipinitialspace=True))
    header = [h.strip() for h in rows[0] if h.strip()]
    values = [v.strip() for v in rows[1] if v.strip()]
    return dict(zip(header, values))


def run_one(cfg_path, topo_path, layer_dir):
    """Run SCALE-Sim for one config on one single-layer topology."""
    name = cfg_path.stem
    out_dir = layer_dir / name
    if (out_dir / "COMPUTE_REPORT.csv").exists():
        return name, "skipped", 0.0

    log_path = layer_dir / "logs" / f"{name}.log"
    cmd = [sys.executable, "-m", "scalesim.scale",
           "-c", str(cfg_path), "-t", str(topo_path),
           "-p", str(layer_dir), "-i", "conv"]
    t0 = time.time()
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    status = "ok" if rc == 0 and (out_dir / "COMPUTE_REPORT.csv").exists() else f"FAILED (rc={rc})"
    return name, status, time.time() - t0


def summarise(model, layer, layer_dir, cfgs):
    """Collect the 27 runs of one layer into a list of flat dicts."""
    rows = []
    for cfg in cfgs:
        run_dir = layer_dir / cfg.stem
        if not (run_dir / "COMPUTE_REPORT.csv").exists():
            continue
        df, h, w, s = CFG_RE.match(cfg.stem).groups()
        comp = read_report(run_dir / "COMPUTE_REPORT.csv")
        bw = read_report(run_dir / "BANDWIDTH_REPORT.csv")
        det = read_report(run_dir / "DETAILED_ACCESS_REPORT.csv")
        dram_rd = float(det["DRAM IFMAP Reads"]) + float(det["DRAM Filter Reads"])
        dram_wr = float(det["DRAM OFMAP Writes"])
        rows.append({
            "model": model, "layer": layer, "config": cfg.stem,
            "dataflow": df, "array_h": int(h), "array_w": int(w), "sram_kB": int(s),
            "total_cycles": comp["Total Cycles"],
            "stall_cycles": comp["Stall Cycles"],
            "overall_util_pct": comp["Overall Util %"],
            "mapping_eff_pct": comp["Mapping Efficiency %"],
            "compute_util_pct": comp["Compute Util %"],
            "ifmap_sram_bw": bw["Avg IFMAP SRAM BW"],
            "filter_sram_bw": bw["Avg FILTER SRAM BW"],
            "ofmap_sram_bw": bw["Avg OFMAP SRAM BW"],
            "ifmap_dram_bw": bw["Avg IFMAP DRAM BW"],
            "filter_dram_bw": bw["Avg FILTER DRAM BW"],
            "ofmap_dram_bw": bw["Avg OFMAP DRAM BW"],
            "dram_ifmap_reads": det["DRAM IFMAP Reads"],
            "dram_filter_reads": det["DRAM Filter Reads"],
            "dram_ofmap_writes": det["DRAM OFMAP Writes"],
            "dram_total_accesses": int(dram_rd + dram_wr),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    # Limited by RAM, not CPU: heavy runs (e.g. FC6, OS, 64x16) can need 2-3+ GB each.
    ap.add_argument("--workers", type=int, default=4,
                    help="number of SCALE-Sim runs in parallel (default: 4, sized for 16 GB RAM)")
    args = ap.parse_args()

    cfgs = sorted(CONFIG_DIR.glob("*.cfg"))
    assert cfgs, f"No configs found in {CONFIG_DIR}"
    n_cfg = len(cfgs)

    total_layers = sum(len(read_topology(p)[1]) for p in MODELS.values())
    print(f"{n_cfg} configs x {total_layers} layers = {n_cfg * total_layers} runs, "
          f"{args.workers} in parallel\n")

    all_rows, layer_no, t_start = [], 0, time.time()

    for model, topo_file in MODELS.items():
        header, layers = read_topology(topo_file)

        for layer, row in layers:
            layer_no += 1
            layer_dir = RESULTS_DIR / model / layer
            (layer_dir / "logs").mkdir(parents=True, exist_ok=True)

            # single-layer topology file for this layer
            topo_path = layer_dir / "topology.csv"
            topo_path.write_text(header + "\n" + row + "\n")

            print(f"[{layer_no}/{total_layers}] {model} {layer}: running {n_cfg} configs ...")
            t_layer, failed = time.time(), []

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_one, c, topo_path, layer_dir) for c in cfgs]
                for i, fut in enumerate(as_completed(futures), 1):
                    name, status, secs = fut.result()
                    print(f"    ({i:2d}/{n_cfg}) {name:22s} {status:8s} {secs:7.1f}s")
                    if status.startswith("FAILED"):
                        failed.append(name)

            rows = summarise(model, layer, layer_dir, cfgs)
            write_csv(layer_dir / "summary.csv", rows)
            all_rows.extend(rows)

            mins = (time.time() - t_layer) / 60
            if failed:
                print(f"==> {model} {layer}: {n_cfg - len(failed)}/{n_cfg} configs done, "
                      f"{len(failed)} FAILED ({', '.join(failed)}) - see {layer_dir / 'logs'}")
            else:
                print(f"==> {model} {layer}: ALL {n_cfg} CONFIGS DONE in {mins:.1f} min. "
                      f"Results saved to {layer_dir}")
            print()

    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(RESULTS_DIR / "all_results.csv", all_rows)
    print(f"ALL DONE in {(time.time() - t_start) / 60:.1f} min. "
          f"Combined results: {RESULTS_DIR / 'all_results.csv'}")


if __name__ == "__main__":
    main()
