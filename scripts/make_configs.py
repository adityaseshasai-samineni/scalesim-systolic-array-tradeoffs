"""Generate the 27 SCALE-Sim 2.0.2 config files for the sweep:
3 dataflows x 3 array shapes (1024 PEs each) x 3 SRAM sizes.

Usage (from project root):  .venv\\Scripts\\python.exe scripts\\make_configs.py
"""
from itertools import product
from pathlib import Path

DATAFLOWS = ["os", "ws", "is"]
SHAPES = [(32, 32), (16, 64), (64, 16)]   # (ArrayHeight, ArrayWidth), all 1024 PEs
SRAM_KB = [16, 64, 256]                   # same size for IFMAP, FILTER and OFMAP SRAM

TEMPLATE = """[general]
run_name = {name}

[architecture_presets]
ArrayHeight:    {h}
ArrayWidth:     {w}
IfmapSramSzkB:  {s}
FilterSramSzkB: {s}
OfmapSramSzkB:  {s}
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 10
Dataflow : {df}
MemoryBanks:   1

[run_presets]
InterfaceBandwidth: CALC
"""

out_dir = Path(__file__).resolve().parent.parent / "configs"
out_dir.mkdir(exist_ok=True)

for df, (h, w), s in product(DATAFLOWS, SHAPES, SRAM_KB):
    name = f"{df}_{h}x{w}_sram{s}kB"
    (out_dir / f"{name}.cfg").write_text(TEMPLATE.format(name=name, h=h, w=w, s=s, df=df))
    print(name)
