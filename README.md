# SCALE-Sim systolic array tradeoffs

This P15 architecture project studies how dataflow, array geometry, and SRAM capacity affect dense DNN layers in SCALE-Sim.

The completed pilot compares output-, weight-, and input-stationary dataflows on AlexNet conv1, batch-one fc8, and a small validation GEMM. The AlexNet runs use a 32×32 array and configure 256 KiB per operand SRAM. All nine runs use SCALE-Sim 2.0.2 in CALC mode.

| Workload | OS cycles | WS cycles | IS cycles |
| --- | ---: | ---: | ---: |
| Tiny GEMM | 35 | 29 | 33 |
| AlexNet conv1 | 121,124 | 112,283 | 216,599 |
| AlexNet fc8 | 133,055 | 389,119 | 140,031 |

WS was fastest on conv1, while OS had the lowest reported DRAM traffic. OS was fastest on batch-one fc8. CALC mode estimates bandwidth demand for stall-free execution. The array shape, SRAM capacity, and fixed bandwidth sweeps are pending.

The nine runs' configurations and original compute, bandwidth, and detailed access reports are in [results/pilot](results/pilot). The inputs and Slurm script are in [experiments](experiments).

From the repository root on a Slurm system with Python 3.11, run:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
sbatch experiments/pilot.sbatch
```

References include the [SCALE-Sim ISPASS 2020 paper](https://github.com/scalesim-project/SCALE-Sim#detailed-documentation) and the [SCALE-Sim v2 docs](https://scale-sim-project.readthedocs.io/en/latest/).
