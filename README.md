# SCALE-Sim systolic array tradeoffs

This project studies how the **dataflow** of a systolic-array accelerator determines data reuse:
output stationary (OS), weight stationary (WS) and input stationary (IS). It uses SCALE-Sim 2.0.2 to model
convolution and fully connected (FC) layers from two DNNs. It then measures how dataflow, array shape,
array size and on-chip SRAM size affect three things: **compute utilization**, **runtime** and
**off-chip (DRAM) traffic and bandwidth demand**.

In total there are **567 simulations**: 7 layers × 27 hardware configurations × 3 array sizes.

## 1. Study design

| Knob | Values |
| --- | --- |
| Dataflow | OS, WS, IS |
| Array size (PEs) | 256, 1,024, 4,096 |
| Array shape, same PE count | square, wide, tall: 16×16 / 8×32 / 32×8, 32×32 / 16×64 / 64×16, 64×64 / 32×128 / 128×32 |
| SRAM per operand | 16, 64, 256 KiB (IFMAP = FILTER = OFMAP) |
| Bandwidth mode | `InterfaceBandwidth: CALC`, 1-byte words |

Each array size has 27 configurations (3 dataflows × 3 shapes × 3 SRAM sizes). Within one size the PE count
is fixed, so the shape effect is separated from the size effect.

The layers come from AlexNet (torchvision variant) and ResNet-18. SCALE-Sim maps each layer to a matrix multiply
(GEMM): Npx = output pixels, K = R·S·C (dot-product length), M = filters.

| Model | Layer | Output | Npx | K | M | MACs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| AlexNet | Conv2 | 27×27 | 729 | 1,600 | 192 | 224.0 M |
| AlexNet | Conv5 | 13×13 | 169 | 2,304 | 256 | 99.7 M |
| AlexNet | FC6 | 1×1 | 1 | 9,216 | 4,096 | 37.7 M |
| AlexNet | FC8 | 1×1 | 1 | 4,096 | 1,000 | 4.1 M |
| ResNet-18 | Conv2_1a | 56×56 | 3,136 | 576 | 64 | 115.6 M |
| ResNet-18 | Conv5_1b | 7×7 | 49 | 4,608 | 512 | 115.6 M |
| ResNet-18 | FC | 1×1 | 1 | 512 | 1,000 | 0.5 M |

The topology files give each conv input at its padded size, so every layer produces its real output size.
FC layers are written as 1×1 convolutions at batch 1.

## 2. Step-by-step process

All commands run from the repository root and are shown for Windows PowerShell with Python 3.11. On Linux or macOS,
use `.venv/bin/python`.

**Step 1: install SCALE-Sim 2.0.2.**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install scalesim==2.0.2 "numpy<2" "pandas<2.3" "matplotlib<3.10"
```

SCALE-Sim 2.0.2 crashes with NumPy 2.x, so NumPy is pinned below 2. pandas 2.3+ needs NumPy 2, so pandas is
pinned too.

**Step 2: describe the layers.** The topology files are
[topologies/alexnet_selected.csv](topologies/alexnet_selected.csv) and
[topologies/resnet18_selected.csv](topologies/resnet18_selected.csv). Each row is
`name, IFMAP H, IFMAP W, filter H, filter W, channels, filters, stride,`. The trailing comma is required.
[layer_table.csv](layer_table.csv) lists the GEMM dimensions used in the plots.

**Step 3: generate the hardware configs.** [scripts/make_configs.py](scripts/make_configs.py) writes 27 configs
per array size into `configs_pe256/`, `configs_pe1024/` and `configs_pe4096/`. Each is named
`<dataflow>_<H>x<W>_sram<kB>kB.cfg`.

```powershell
.venv\Scripts\python.exe scripts\make_configs.py
```

**Step 4: run the simulations.** [scripts/run_all.py](scripts/run_all.py) splits each model into single-layer
topologies, runs all 27 configs of a layer in parallel and prints a "done" line after each layer. It writes
`results_peN/`. It is resumable: runs that already have a `COMPUTE_REPORT.csv` are skipped.

```powershell
.venv\Scripts\python.exe -u scripts\run_all.py --pes 1024 --workers 4
.venv\Scripts\python.exe -u scripts\run_all.py --pes 4096 --workers 4
.venv\Scripts\python.exe -u scripts\run_all.py --pes 256  --workers 3
```

Memory is the limit, not CPU. On a 14-core machine with 16 GB of RAM the three sweeps took 64, 42 and 186 minutes.
The heaviest runs (AlexNet FC6, especially on the 256-PE arrays) need 2–5 GB each. Two 256-PE FC6 runs ran out of
memory with 3 workers and were rerun one at a time.

**Step 5: build the plots.** [scripts/make_plots.py](scripts/make_plots.py) writes
`results_peN/scalesim_results.pdf`, a 4-page report per array size:

```powershell
.venv\Scripts\python.exe scripts\make_plots.py --pes 1024          # also 256, 4096
.venv\Scripts\python.exe scripts\make_plots.py --pes 1024 --full   # detailed 8-page version
```

| Page | Content |
| --- | --- |
| 1 | Compute utilization of all 27 configs × 7 layers (heatmap), best marked ★ |
| 2 | Runtime: fastest config in million cycles, the others as "× slower" |
| 3 | DRAM traffic: least traffic in MB, the others as "× more" |
| 4 | DRAM bandwidth demand vs. three DRAM bandwidth levels, with a slowdown estimate |

## 3. Results

Utilization and runtime are the same for every SRAM size in CALC mode (see Notes), so those columns name only
dataflow and shape.

### Best configuration per layer

**256 PEs**

| Layer | Utilization (max) | Runtime (min cycles) | DRAM traffic (min) |
| --- | --- | --- | --- |
| AlexNet Conv2 | OS 16×16, 97.2% | OS 16×16, 899,759 | OS 32×8 256 KiB, 0.51 MB |
| AlexNet Conv5 | OS 16×16, 94.8% | OS 16×16, 410,783 | OS 32×8 256 KiB, 0.70 MB |
| AlexNet FC6 | OS 8×32, 12.4% | OS 8×32, 1,184,511 | OS, 64 or 256 KiB, any shape, 37.76 MB |
| AlexNet FC8 | OS 8×32, 12.1% | OS 8×32, 132,287 | OS, any shape or SRAM, 4.10 MB |
| ResNet-18 Conv2_1a | WS 16×16 or 8×32, 98.6% | WS 16×16 or 8×32, 458,207 | OS 8×32 256 KiB, 0.69 MB |
| ResNet-18 Conv5_1b | OS 8×32, 86.8% | OS 8×32, 520,351 | OS 32×8 256 KiB, 2.86 MB |
| ResNet-18 FC | IS 32×8, 11.7% | IS 32×8, 17,119 | OS, any shape or SRAM, 0.51 MB |

**1,024 PEs**

| Layer | Utilization (max) | Runtime (min cycles) | DRAM traffic (min) |
| --- | --- | --- | --- |
| AlexNet Conv2 | OS 32×32, 95.4% | OS 32×32, 229,355 | OS 64×16 256 KiB, 0.51 MB |
| AlexNet Conv5 | OS 16×64, 92.9% | OS 16×64, 104,807 | OS 64×16 256 KiB, 0.79 MB |
| AlexNet FC6 | OS 16×64, 6.2% | OS 16×64, 594,815 | OS, 64 or 256 KiB, any shape, 37.76 MB |
| AlexNet FC8 | OS 16×64, 6.0% | OS 16×64, 66,783 | OS, any shape or SRAM, 4.10 MB |
| ResNet-18 Conv2_1a | WS 16×64 or 32×32, 97.1% | WS 16×64 or 32×32, 116,279 | OS 16×64 256 KiB, 0.46 MB |
| ResNet-18 Conv5_1b | OS 32×32, 75.5% | OS 32×32, 149,439 | OS 64×16 256 KiB, 2.43 MB |
| ResNet-18 FC | IS 64×16, 5.5% | IS 64×16, 9,135 | OS, any shape or SRAM, 0.51 MB |

**4,096 PEs**

| Layer | Utilization (max) | Runtime (min cycles) | DRAM traffic (min) |
| --- | --- | --- | --- |
| AlexNet Conv2 | OS 64×64, 88.0% | OS 64×64, 62,135 | OS 128×32 256 KiB, 0.57 MB |
| AlexNet Conv5 | OS 64×64, 83.5% | OS 64×64, 29,159 | OS 128×32 256 KiB, 1.09 MB |
| AlexNet FC6 | OS 32×128, 3.1% | OS 32×128, 299,967 | OS, 64 or 256 KiB, any shape, 37.76 MB |
| AlexNet FC8 | OS 32×128, 2.9% | OS 32×128, 34,031 | OS, any shape or SRAM, 4.10 MB |
| ResNet-18 Conv2_1a | WS 64×64, 94.3% | WS 64×64, 29,933 | OS 64×64 256 KiB, 0.46 MB |
| ResNet-18 Conv5_1b | OS 64×64, 74.5% | OS 64×64, 37,871 | OS 64×64 or 128×32 256 KiB, 2.43 MB |
| ResNet-18 FC | IS 128×32, 2.4% | IS 128×32, 5,143 | OS, any shape or SRAM, 0.51 MB |

### Array size: speedup vs. utilization

The fastest config of each layer at each size:

| Layer | Cycles at 256 PEs | Cycles at 1,024 PEs | Cycles at 4,096 PEs | Speedup, 256 → 4,096 | Best utilization, 256 / 1,024 / 4,096 |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet Conv2 | 899,759 | 229,355 | 62,135 | 14.5× | 97.2 / 95.4 / 88.0% |
| AlexNet Conv5 | 410,783 | 104,807 | 29,159 | 14.1× | 94.8 / 92.9 / 83.5% |
| AlexNet FC6 | 1,184,511 | 594,815 | 299,967 | 3.9× | 12.4 / 6.2 / 3.1% |
| AlexNet FC8 | 132,287 | 66,783 | 34,031 | 3.9× | 12.1 / 6.0 / 2.9% |
| ResNet-18 Conv2_1a | 458,207 | 116,279 | 29,933 | 15.3× | 98.6 / 97.1 / 94.3% |
| ResNet-18 Conv5_1b | 520,351 | 149,439 | 37,871 | 13.7× | 86.8 / 75.5 / 74.5% |
| ResNet-18 FC | 17,119 | 9,135 | 5,143 | 3.3× | 11.7 / 5.5 / 2.4% |

Going from 256 to 4,096 PEs multiplies the PE count by 16. Conv layers get 14–15× faster. FC layers get only 3–4×
faster, because at batch 1 most of the extra PEs have no work.

### DRAM bandwidth demand

Each config's average demand is DRAM traffic ÷ cycles, in bytes per cycle (= GB/s at 1 GHz). Page 4 of each PDF
compares it with three DRAM bandwidth levels per size:

| PEs | Levels (B/cycle) | Configs that fit at low / mid / high | Demand of the fastest configs | Fastest configs at the mid level |
| --- | --- | --- | --- | --- |
| 256 | 8 / 16 / 32 | 30% / 44% / 84% | 0.6 – 39.5 | at least 2.5× slower |
| 1,024 | 16 / 32 / 64 | 30% / 49% / 90% | 2.5 – 78.0 | at least 2.4× slower |
| 4,096 | 32 / 64 / 128 | 32% / 55% / 98% | 9.1 – 126.8 | at least 2.0× slower |

The slowdown estimate assumes runtime ≥ traffic ÷ bandwidth. Real DRAM traffic is bursty, so these are lower bounds.

### Findings

- **Which layer shapes favour which dataflow.**
  - **WS** wins only when Npx (output pixels) is large. It keeps the weights fixed and streams pixels through them, so it
    needs many pixels per weight tile. ResNet Conv2_1a (3,136 pixels) reaches 94–99% utilization with WS at every size.
  - **OS** wins on layers with few pixels and a long K (AlexNet Conv2/Conv5, ResNet Conv5_1b) and on most FC layers.
  - **IS** wins only on the smallest FC layer (ResNet-18 FC).
  - ResNet Conv2_1a and Conv5_1b do the same work (115.6 M MACs), but WS runs Conv5_1b 1.9× (256 PEs), 2.8× (1,024)
    and 4.6× (4,096) slower, because Conv5_1b has only 49 pixels.
- **FC layers at batch 1 waste the array.**
  - No FC layer exceeds 12.4% utilization at 256 PEs, 6.2% at 1,024 or 3.1% at 4,096.
  - WS is up to 8.9× slower than the best dataflow on FC layers, because each weight tile is used for one cycle.
- **Array shape matters even at a fixed PE count.**
  - For FC layers, shape alone changes runtime by up to 4.0× at every size: OS needs a wide array, IS a tall one.
- **Bigger arrays are faster but less utilized.**
  - Utilization drops with size for every layer.
  - The speedup from 16× more PEs is 14–15× for conv layers but only 3–4× for FC layers.
- **SRAM affects only DRAM traffic** (in CALC mode).
  - With OS, going from 16 to 256 KiB cuts conv-layer traffic by up to 53× (256 PEs), 27× (1,024) and 11× (4,096),
    because each output is written only once.
  - WS and IS keep sending partial sums off-chip whenever K is larger than the array height.
- **FC layers are memory-bound.**
  - Every weight is used once, so DRAM traffic stays at about the weight size (37.8 MB for FC6) in every configuration
    (spread ≤ 1.16×).
  - The fastest FC configs need 31–33 B/cycle (256 PEs), 57–64 (1,024) and 100–127 (4,096).

### Cross-check with the pilot

The earlier pilot ([results/pilot](results/pilot), inputs and Slurm script in [experiments](experiments)) ran AlexNet
fc8 as a GEMM on a 32×32 array with 256 KiB SRAM. This sweep runs the same layer as a 1×1 convolution and produces
identical reports: OS 133,055, WS 389,119 and IS 140,031 cycles.

## 4. Files

| Path | Contents |
| --- | --- |
| [topologies/](topologies) | Layer definitions for the two models |
| [layer_table.csv](layer_table.csv) | GEMM view of each layer (Npx, K, M, MACs), used by the plots |
| [config_table.csv](config_table.csv) | The 27 configurations of the 1,024-PE sweep |
| `configs_pe256/`, `configs_pe1024/`, `configs_pe4096/` | 27 SCALE-Sim configs per array size |
| `results_peN/all_results.csv` | All 189 runs of one array size in one table |
| `results_peN/scalesim_results.pdf` | The 4-page report for that size |
| `results_peN/<Model>/<Layer>/` | Per-run SCALE-Sim reports, `summary.csv`, the single-layer `topology.csv`, and logs |
| [scripts/](scripts) | `make_configs.py`, `run_all.py`, `make_plots.py` |
| [results/pilot](results/pilot), [experiments](experiments) | The earlier pilot (9 runs) |

## 5. Notes and limitations

- **CALC mode.** SCALE-Sim estimates the DRAM bandwidth needed for stall-free execution, so stall cycles are always 0.
  Runtime and utilization depend only on dataflow and array shape. SRAM size changes only DRAM traffic and bandwidth
  demand. Across all 567 runs the three SRAM sizes give identical cycles.
- **USER mode was not used.** USER mode fixes the DRAM bandwidth, but in SCALE-Sim 2.0.2 its read buffer
  (`scalesim/memory/read_buffer.py`) is marked "TODO: Verification Pending". In a test it re-fetched every WS filter
  weight about 38× (16 B/cycle, 256 KiB SRAM), which made WS look 17–73× slower than CALC for reasons that have
  nothing to do with the hardware. The bandwidth question is therefore answered from CALC demand (page 4) instead.
- **Bandwidth metric.** SCALE-Sim's own `Avg DRAM BW` columns average each stream over its active window only. The
  figures here use total DRAM accesses ÷ total cycles.

## References

- [SCALE-Sim ISPASS 2020 paper](https://github.com/scalesim-project/SCALE-Sim#detailed-documentation)
- [SCALE-Sim v2 documentation](https://scale-sim-project.readthedocs.io/en/latest/)
