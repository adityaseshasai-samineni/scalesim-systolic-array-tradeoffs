# SCALE-Sim sweep: dataflow × array shape × SRAM size

This branch extends the [pilot](README.md) with the full sweep. It covers 7 conv and FC layers from two DNNs, each simulated in 27 hardware configurations, for **189 SCALE-Sim 2.0.2 runs** in CALC mode. The three results are compute utilization, runtime, and off-chip (DRAM) traffic.

## Setup

| Knob | Values |
| --- | --- |
| Dataflow | output stationary (OS), weight stationary (WS), input stationary (IS) |
| Array shape | 32×32 (square), 16×64 (wide), 64×16 (tall), all 1,024 PEs |
| SRAM per operand | 16, 64, 256 KiB (IFMAP = FILTER = OFMAP) |
| Mode | `InterfaceBandwidth: CALC`, 1-byte words |

Fixing the PE count isolates the effect of array shape from array size.

The layers come from AlexNet (torchvision variant) and ResNet-18. Each one is shown in SCALE-Sim's matrix-multiply (GEMM) view: Npx = output pixels, K = R·S·C (dot-product length), M = filters.

| Model | Layer | Output | Npx | K | M | MACs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AlexNet | Conv2 | 27×27 | 729 | 1,600 | 192 | 224.0 M |
| AlexNet | Conv5 | 13×13 | 169 | 2,304 | 256 | 99.7 M |
| AlexNet | FC6 | 1×1 | 1 | 9,216 | 4,096 | 37.7 M |
| AlexNet | FC8 | 1×1 | 1 | 4,096 | 1,000 | 4.1 M |
| ResNet-18 | Conv2_1a | 56×56 | 3,136 | 576 | 64 | 115.6 M |
| ResNet-18 | Conv5_1b | 7×7 | 49 | 4,608 | 512 | 115.6 M |
| ResNet-18 | FC | 1×1 | 1 | 512 | 1,000 | 0.5 M |

The topology files list each conv input at its padded size, so every layer produces its real output size. FC layers are written as 1×1 convolutions at batch 1.

## Best configuration per layer

| Layer | Utilization (max) | Runtime (min cycles) | DRAM traffic (min) |
| --- | --- | --- | --- |
| AlexNet Conv2 | OS 32×32, 95.4% | OS 32×32, 229,355 | OS 64×16 256 KiB, 0.51 MB |
| AlexNet Conv5 | OS 16×64, 92.9% | OS 16×64, 104,807 | OS 64×16 256 KiB, 0.79 MB |
| AlexNet FC6 | OS 16×64, 6.2% | OS 16×64, 594,815 | OS, 64 or 256 KiB (any shape), 37.76 MB |
| AlexNet FC8 | OS 16×64, 6.0% | OS 16×64, 66,783 | OS (any shape or SRAM), 4.10 MB |
| ResNet-18 Conv2_1a | WS 16×64 or 32×32, 97.1% | WS 16×64 or 32×32, 116,279 | OS 16×64 256 KiB, 0.46 MB |
| ResNet-18 Conv5_1b | OS 32×32, 75.5% | OS 32×32, 149,439 | OS 64×16 256 KiB, 2.43 MB |
| ResNet-18 FC | IS 64×16, 5.5% | IS 64×16, 9,135 | OS (any shape or SRAM), 0.51 MB |

## Findings

- **Which layer shapes favour which dataflow.**
  - **WS** wins only when Npx (output pixels) is large. That's because WS streams the pixels through fixed weights. ResNet Conv2_1a, with 3,136 pixels, reaches 97.1% utilization.
  - **OS** wins on layers with few pixels and a long K.
  - ResNet Conv2_1a and Conv5_1b do exactly the same work (115.6 M MACs), but WS runs Conv5_1b 2.8× slower, because Conv5_1b has only 49 pixels.
- **FC layers at batch 1 waste the array.** No FC layer exceeds 6.2% utilization under any dataflow.
  - **WS:** 100% mapping efficiency but about 1% utilization, because each weight tile is used for only one cycle.
  - **OS and IS:** the batch dimension (size 1) is mapped onto rows (OS) or columns (IS), so only one row or column does any work.
- **Array shape matters even with the PE count fixed.**
  - For FC layers, shape alone changes runtime by up to 4.0×: wide arrays are best for OS, tall arrays for IS.
  - The tall 64×16 array is usually the slowest, because pipeline fill/drain grows with 2H + W.
- **SRAM affects only off-chip traffic.**
  - In CALC mode, utilization and runtime are identical for all three SRAM sizes.
  - With OS, going from 16 to 256 KiB cuts conv-layer DRAM traffic by up to 27×, because each output is written only once.
  - WS and IS keep sending partial sums off-chip whenever K is larger than the array height.
- **FC layers are memory-bound.** Every weight is used once, so DRAM traffic stays at roughly the weight size (37.8 MB for FC6) for every dataflow, shape and SRAM size. The fastest FC configurations need 57–64 bytes/cycle (57–64 GB/s at 1 GHz).

**Cross-check with the pilot.** The pilot ran AlexNet FC8 as a matrix multiply (`-i gemm`). This sweep runs the same layer as a 1×1 convolution. On a 32×32 array with 256 KiB SRAM, both give identical reports: OS 133,055, WS 389,119 and IS 140,031 cycles.

## Files

| Path | Contents |
| --- | --- |
| [configs/](configs) | 27 configs named `<dataflow>_<H>x<W>_sram<kB>kB.cfg` |
| [topologies/](topologies) | `alexnet_selected.csv`, `resnet18_selected.csv` |
| [scripts/make_configs.py](scripts/make_configs.py) | Generates the 27 configs |
| [scripts/run_all.py](scripts/run_all.py) | Runs every config on every layer, one layer at a time, in parallel; resumable |
| [scripts/make_plots.py](scripts/make_plots.py) | Builds the results PDF |
| [results/all_results.csv](results/all_results.csv) | All 189 runs in one table |
| [results/scalesim_results.pdf](results/scalesim_results.pdf) | One heatmap per result, with the best configuration marked ★ |
| `results/<Model>/<Layer>/` | Per-run SCALE-Sim reports, `summary.csv`, the single-layer `topology.csv`, and logs |

## Reproduce

From the repository root, with Python 3.11 (shown for Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install scalesim==2.0.2 "numpy<2" "pandas<2.3" "matplotlib<3.10"
.venv\Scripts\python.exe scripts\make_configs.py
.venv\Scripts\python.exe scripts\run_all.py --workers 4
.venv\Scripts\python.exe scripts\make_plots.py           # 3-page PDF
.venv\Scripts\python.exe scripts\make_plots.py --full    # detailed 8-page PDF
```

SCALE-Sim 2.0.2 crashes with NumPy 2.x, so NumPy is pinned below 2.

The full sweep took about 64 minutes with 4 workers on a 14-core machine with 16 GB of RAM. Memory is the limit, not CPU: the heaviest runs (AlexNet FC6) need 2–3 GB each.

## Notes

- In CALC mode SCALE-Sim estimates the bandwidth needed for stall-free execution, so stall cycles are always 0.
- The `Avg DRAM BW` columns in SCALE-Sim's own reports are averaged over each stream's active window only. The bandwidth figures above use total DRAM accesses ÷ total cycles instead.
