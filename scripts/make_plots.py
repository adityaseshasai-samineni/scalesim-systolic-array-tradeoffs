"""Build the results PDF from results/all_results.csv.

Three results, each shown as (a) an overview heatmap of every run and (b) per-layer bars,
with the best configuration of every layer highlighted (thick outline + star).

  1. Compute utilization   (higher is better)   - 9 configs per layer (SRAM has no effect)
  2. Runtime (cycles)      (lower is better)    - 9 configs per layer (SRAM has no effect)
  3. Off-chip traffic      (lower is better)    - all 27 configs per layer
     + average off-chip bandwidth demand (bytes/cycle)

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\make_plots.py [--png-dir DIR]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "all_results.csv"
LAYERS = ROOT / "layer_table.csv"
OUT_PDF = ROOT / "results" / "scalesim_results.pdf"

# ---------------------------------------------------------------- palette (reference dataviz palette, light)
DF_ORDER = ["os", "ws", "is"]
DF_NAME = {"os": "Output Stationary (OS)", "ws": "Weight Stationary (WS)", "is": "Input Stationary (IS)"}
DF_SHORT = {"os": "OS", "ws": "WS", "is": "IS"}
DF_COLOR = {"os": "#2a78d6", "ws": "#eb6834", "is": "#1baf7a"}   # categorical slots 1-3
SHAPES = ["32x32", "16x64", "64x16"]
SHAPE_LBL = {"32x32": "32×32", "16x64": "16×64", "64x16": "64×16"}
SRAMS = [16, 64, 256]
SRAM_TINT = {16: 0.40, 64: 0.70, 256: 1.00}                      # lightness carries SRAM size

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})
A4 = (11.69, 8.27)  # landscape


def tint(hex_color, t):
    """Mix a color with white: t=1 -> full color, t<1 -> lighter."""
    r, g, b = to_rgb(hex_color)
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


# ---------------------------------------------------------------- data
def load():
    d = pd.read_csv(RESULTS)
    lt = pd.read_csv(LAYERS)
    d["arr"] = d.array_h.astype(str) + "x" + d.array_w.astype(str)
    d["traffic_MB"] = d.dram_total_accesses / 1e6            # 1-byte words -> bytes
    d["bw_Bpc"] = d.dram_total_accesses / d.total_cycles      # bytes/cycle (= GB/s at 1 GHz)
    d["Mcycles"] = d.total_cycles / 1e6
    d = d.merge(lt[["model", "layer", "layer_type", "Npx", "K", "M"]], on=["model", "layer"])
    order = list(zip(lt.model, lt.layer))
    d["lidx"] = [order.index((m, l)) for m, l in zip(d.model, d.layer)]
    d["row_lbl"] = [f"{m} {l}\nNpx={n:,}  K={k:,}  M={mm:,}"
                    for m, l, n, k, mm in zip(d.model, d.layer, d.Npx, d.K, d.M)]
    d["title_lbl"] = d.model + " " + d.layer
    return d, order


def col_keys(with_sram):
    keys = []
    for df in DF_ORDER:
        for s in SHAPES:
            if with_sram:
                keys += [(df, s, k) for k in SRAMS]
            else:
                keys.append((df, s))
    return keys


def matrix(d, value, with_sram):
    """rows = layers (table order), cols = config keys. SRAM-invariant metrics use the 64 kB runs."""
    src = d if with_sram else d[d.sram_kB == 64]
    keys = col_keys(with_sram)
    rows = []
    for li in sorted(src.lidx.unique()):
        s = src[src.lidx == li]
        idx = (list(zip(s.dataflow, s.arr, s.sram_kB)) if with_sram else list(zip(s.dataflow, s.arr)))
        lut = dict(zip(idx, s[value]))
        rows.append([lut[k] for k in keys])
    labels = src.sort_values("lidx").drop_duplicates("lidx").row_lbl.tolist()
    return np.array(rows, dtype=float), keys, labels


def best_mask(m, higher_better):
    b = m.max(axis=1, keepdims=True) if higher_better else m.min(axis=1, keepdims=True)
    return np.isclose(m, b, rtol=1e-9, atol=0)


# ---------------------------------------------------------------- drawing helpers
def page_title(fig, title, subtitle):
    fig.text(0.04, 0.955, title, fontsize=17, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.912, subtitle, fontsize=9.5, color=INK2, va="top")


def outline_best(ax, x, y, w=1.0, h=1.0):
    """Double outline (black outer, white inner) - visible on light and dark cells."""
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False, ec=INK, lw=3.2, zorder=5))
    ax.add_patch(Rectangle((x - w / 2 + 0.06, y - h / 2 + 0.06), w - 0.12, h - 0.12,
                           fill=False, ec="white", lw=1.1, zorder=6))


def heatmap(ax, score, text, best, keys, row_labels, with_sram, cbar_label, fig, fontsize=8.5):
    """score in [0,1] drives color (darker = better); text is what is printed in the cell."""
    n_r, n_c = score.shape
    im = ax.imshow(score, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    for r in range(n_r):
        for c in range(n_c):
            color = "white" if score[r, c] > 0.55 else INK
            t = (("★\n" if with_sram else "★ ") if best[r, c] else "") + text[r, c]
            ax.text(c, r, t, ha="center", va="center", fontsize=fontsize, color=color,
                    weight="bold" if best[r, c] else "normal", zorder=7)
            if best[r, c]:
                outline_best(ax, c, r)
    # x tick labels: shape (and SRAM), dataflow group labels above
    if with_sram:
        ax.set_xticks(range(n_c))
        ax.set_xticklabels([f"{k[2]}" for k in keys], fontsize=7, color=INK2)
        for gi, (df, s) in enumerate([(df, s) for df in DF_ORDER for s in SHAPES]):
            ax.annotate(SHAPE_LBL[s], xy=(gi * 3 + 1, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -22), textcoords="offset points", ha="center", va="top",
                        fontsize=8.5, weight="bold", color=INK)
            if gi % 3:
                ax.axvline(gi * 3 - 0.5, color="white", lw=2)
        ax.annotate("SRAM kB", xy=(0, 0), xycoords="axes fraction", xytext=(-6, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color=MUTED)
        ax.annotate("array", xy=(0, 0), xycoords="axes fraction", xytext=(-6, -22),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color=MUTED)
        group_w = 9
    else:
        ax.set_xticks(range(n_c))
        ax.set_xticklabels([SHAPE_LBL[k[1]] for k in keys], fontsize=8.5, color=INK)
        group_w = 3
    for gi, df in enumerate(DF_ORDER):
        x0 = gi * group_w - 0.5
        ax.text(x0 + group_w / 2, -0.75, DF_NAME[df], ha="center", va="bottom", fontsize=10,
                weight="bold", color=DF_COLOR[df])
        ax.plot([x0 + 0.1, x0 + group_w - 0.1], [-0.62, -0.62], color=DF_COLOR[df], lw=3,
                solid_capstyle="round", clip_on=False)
        if gi:
            ax.axvline(x0, color="white", lw=4)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for r in range(1, n_r):
        ax.axhline(r - 0.5, color="white", lw=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(cbar_label, color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, color=AXIS)


def legend_dataflows(fig, y=0.06, extra=None):
    handles = [Patch(color=DF_COLOR[df], label=DF_NAME[df]) for df in DF_ORDER]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, y - 0.04))


def style_bar_ax(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="y", length=0, labelsize=7.5)
    ax.tick_params(axis="x", length=0)


def label_best(ax, xs, vals, best, fmt, top, fontsize):
    """One label for the best value; ties get a single label centred over the tied bars."""
    bx = np.array(xs)[best]
    n = int(best.sum())
    t = "★ " + fmt(vals[best][0]) + (f"  ({n} tied)" if n > 1 else "")
    ax.text(bx.mean(), vals[best][0] + top * 0.035, t, ha="center", va="bottom",
            fontsize=fontsize, weight="bold", color=INK, zorder=4)


def bars_9(ax, s, value, higher_better, fmt, ylabel):
    """9 bars: 3 dataflow groups x 3 shapes. Best bar(s): black outline + star label."""
    s = s[s.sram_kB == 64]
    xs, vals, cols, ticks = [], [], [], []
    for gi, df in enumerate(DF_ORDER):
        for si, sh in enumerate(SHAPES):
            x = gi * 3.6 + si
            v = s[(s.dataflow == df) & (s.arr == sh)][value].iloc[0]
            xs.append(x); vals.append(v); cols.append(DF_COLOR[df]); ticks.append(SHAPE_LBL[sh])
    vals = np.array(vals)
    best = np.isclose(vals, vals.max() if higher_better else vals.min(), rtol=1e-9)
    ax.bar(xs, vals, width=0.82, color=cols, edgecolor=[INK if b else "none" for b in best],
           linewidth=[2.2 if b else 0 for b in best], zorder=3)
    top = vals.max()
    label_best(ax, xs, vals, best, fmt, top, 7.5)
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, rotation=90, fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=7.5)
    style_bar_ax(ax)


def bars_27(ax, s, value, fmt, ylabel):
    """27 bars: dataflow (hue) x shape (cluster) x SRAM (lightness). Best (min): outline + star."""
    xs, vals, cols, cx, cl = [], [], [], [], []
    for gi, df in enumerate(DF_ORDER):
        for si, sh in enumerate(SHAPES):
            base = gi * 12.6 + si * 4.0
            for ki, kb in enumerate(SRAMS):
                v = s[(s.dataflow == df) & (s.arr == sh) & (s.sram_kB == kb)][value].iloc[0]
                xs.append(base + ki); vals.append(v); cols.append(tint(DF_COLOR[df], SRAM_TINT[kb]))
            cx.append(base + 1); cl.append(SHAPE_LBL[sh])
    vals = np.array(vals)
    best = np.isclose(vals, vals.min(), rtol=1e-9)
    ax.bar(xs, vals, width=0.86, color=cols, edgecolor=[INK if b else "none" for b in best],
           linewidth=[1.6 if b else 0 for b in best], zorder=3)
    top = vals.max()
    label_best(ax, xs, vals, best, fmt, top, 7.5)
    ax.set_ylim(0, top * 1.25)
    ax.set_xticks(cx)
    ax.set_xticklabels(cl, fontsize=6.8)
    ax.set_ylabel(ylabel, fontsize=7.5)
    style_bar_ax(ax)


def small_multiples(fig, d, order, draw, nrows, ncols, top=0.81, bottom=0.14, wspace=0.32, hspace=0.75):
    gs = fig.add_gridspec(nrows, ncols, left=0.06, right=0.98, top=top, bottom=bottom,
                          wspace=wspace, hspace=hspace)
    for i, (m, l) in enumerate(order):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        s = d[(d.model == m) & (d.layer == l)]
        r = s.iloc[0]
        draw(ax, s)
        ax.set_title(f"{m} {l}", fontsize=9.5, weight="bold", loc="left", pad=14)
        ax.text(0, 1.02, f"Npx={r.Npx:,}  K={r.K:,}  M={r.M:,}", transform=ax.transAxes,
                fontsize=7, color=MUTED, va="bottom")
    return gs


# ---------------------------------------------------------------- pages
def page_cover(pdf, d, order):
    fig = plt.figure(figsize=A4)
    page_title(fig, "Dataflow, array shape and SRAM trade-offs in a systolic array",
               "SCALE-Sim 2.0.2  ·  189 simulations = 7 layers × 27 configurations  ·  "
               "CALC (bandwidth-estimate) mode, 1-byte words")
    txt = [
        ("Layers (2 DNNs)", "AlexNet: Conv2, Conv5, FC6, FC8     ResNet-18: Conv2_1a, Conv5_1b, FC   (FC at batch 1)"),
        ("Dataflows", "OS - output stationary,  WS - weight stationary,  IS - input stationary"),
        ("Array shapes", "32×32 (square),  16×64 (wide),  64×16 (tall)  - all 1,024 PEs"),
        ("SRAM per buffer", "16 kB,  64 kB,  256 kB  (IFMAP = FILTER = OFMAP)"),
        ("Layer GEMM view", "Npx = output pixels,  K = dot-product length (R·S·C),  M = filters / outputs"),
    ]
    y = 0.84
    for k, v in txt:
        fig.text(0.06, y, k, fontsize=10, weight="bold", color=INK)
        fig.text(0.24, y, v, fontsize=10, color=INK2)
        y -= 0.04

    fig.text(0.06, 0.6, "How to read this report", fontsize=12, weight="bold", color=INK)
    how = [
        "• Three results, each with an overview heatmap (every run) and per-layer bar charts.",
        "• The best configuration of every layer is marked  ★  and outlined in black (ties are all marked).",
        "• Heatmap color is always 'darker = better' (score relative to that layer's best).",
        "• Utilization and runtime are identical for all SRAM sizes in CALC mode (verified: 0.0 difference),",
        "   so those results show 9 configs per layer (3 dataflows × 3 shapes, 64 kB runs).",
        "• Off-chip traffic depends on all three knobs, so it shows all 27 configs per layer.",
    ]
    for i, h in enumerate(how):
        fig.text(0.06, 0.56 - i * 0.035, h, fontsize=9.5, color=INK2)

    # best-per-layer summary table
    fig.text(0.06, 0.32, "Best configuration per layer", fontsize=12, weight="bold", color=INK)
    rows = []
    b64 = d[d.sram_kB == 64]
    for m, l in order:
        s64 = b64[(b64.model == m) & (b64.layer == l)]
        s = d[(d.model == m) & (d.layer == l)]
        u = s64[np.isclose(s64.overall_util_pct, s64.overall_util_pct.max(), rtol=1e-9)]
        t = s64[np.isclose(s64.total_cycles, s64.total_cycles.min(), rtol=1e-9)]
        f = s[np.isclose(s.dram_total_accesses, s.dram_total_accesses.min(), rtol=1e-9)]
        def ties(names):
            names = list(dict.fromkeys(names))
            return names[0] if len(names) == 1 else (
                " / ".join(names) if len(names) == 2 else f"{names[0]} (+{len(names) - 1} tied)")
        fmt = lambda x: ties([f"{DF_SHORT[a]} {SHAPE_LBL[b]}" for a, b in zip(x.dataflow, x.arr)])
        fmt3 = lambda x: ties([f"{DF_SHORT[a]} {SHAPE_LBL[b]} {k} kB"
                               for a, b, k in zip(x.dataflow, x.arr, x.sram_kB)])
        rows.append([f"{m} {l}", f"{fmt(u)}  ({u.overall_util_pct.iloc[0]:.1f}%)",
                     f"{fmt(t)}  ({t.total_cycles.iloc[0]:,} cyc)",
                     f"{fmt3(f)}  ({f.traffic_MB.iloc[0]:.2f} MB)"])
    ax = fig.add_axes([0.06, 0.03, 0.9, 0.27]); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["Layer", "1. Utilization (max)", "2. Runtime (min)",
                                            "3. Off-chip traffic (min)"],
                  loc="upper left", cellLoc="left", colLoc="left", colWidths=[0.16, 0.26, 0.26, 0.32])
    tb.auto_set_font_size(False); tb.set_fontsize(8.3); tb.scale(1, 1.45)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f3f2ee")
    pdf.savefig(fig); return fig


def page_heat_util(pdf, d):
    m, keys, labels = matrix(d, "overall_util_pct", with_sram=False)
    best = best_mask(m, True)
    text = np.vectorize(lambda v: f"{v:.1f}%")(m)
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 1 · Compute utilization",
               "Overall utilization = MACs ÷ (cycles × 1,024 PEs). Higher is better. "
               "★ = best config for that layer. Same for every SRAM size (64 kB runs shown).")
    ax = fig.add_axes([0.2, 0.12, 0.72, 0.68])
    heatmap(ax, m / 100, text, best, keys, labels, False, "utilization (0-100%)", fig, fontsize=9)
    fc_max = d[d.layer_type == "FC"].overall_util_pct.max()
    fig.text(0.04, 0.04, "Reading: WS wins only when Npx (output pixels) is large (ResNet Conv2_1a); "
             f"OS wins on layers with fewer pixels and long K; at batch 1 no FC layer exceeds {fc_max:.1f}%.",
             fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def page_bars_util(pdf, d, order):
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 1 · Compute utilization per layer",
               "9 configs per layer (dataflow × array shape). ★ + black outline = best. Same y-scale (0-100%) in every panel.")
    small_multiples(fig, d, order, lambda ax, s: (
        bars_9(ax, s, "overall_util_pct", True, lambda v: f"{v:.1f}%", "utilization %"),
        ax.set_ylim(0, 115)), 2, 4)
    legend_dataflows(fig)
    pdf.savefig(fig); return fig


def page_heat_runtime(pdf, d):
    m, keys, labels = matrix(d, "total_cycles", with_sram=False)
    best = best_mask(m, False)
    score = m.min(axis=1, keepdims=True) / m                       # 1 = fastest
    text = np.where(best, np.vectorize(lambda v: f"{v / 1e6:.3f}M")(m),
                    np.vectorize(lambda r: f"{r:.2f}×")(m / m.min(axis=1, keepdims=True)))
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 2 · Runtime",
               "★ cell = fastest config (its cycle count, in millions). Other cells = how many times slower than ★. "
               "Same for every SRAM size.")
    ax = fig.add_axes([0.2, 0.12, 0.72, 0.68])
    heatmap(ax, score, text, best, keys, labels, False, "speed relative to best (1 = fastest)", fig, fontsize=9)
    fc = np.array([lbl.split("\n")[0].split()[-1].startswith("FC") for lbl in labels])
    shape_gap = max((m[fc][:, gi * 3:gi * 3 + 3].max(axis=1) / m[fc][:, gi * 3:gi * 3 + 3].min(axis=1)).max()
                    for gi in (0, 2))
    ws_slow = (m[fc][:, 3:6] / m[fc].min(axis=1, keepdims=True)).max()
    fig.text(0.04, 0.04, f"Reading: for FC layers the array shape alone changes runtime up to {shape_gap:.1f}× "
             f"(OS: wide best, IS: tall best); WS is up to {ws_slow:.1f}× slower than the best on FC layers at batch 1.",
             fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def page_bars_runtime(pdf, d, order):
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 2 · Runtime per layer (million cycles)",
               "9 configs per layer. Lower is better. ★ + black outline = fastest. Each panel has its own y-scale.")
    small_multiples(fig, d, order, lambda ax, s: bars_9(
        ax, s, "Mcycles", False, lambda v: f"{v:.3f}M", "million cycles"), 2, 4)
    legend_dataflows(fig)
    pdf.savefig(fig); return fig


def page_heat27_util(pdf, d):
    m, keys, labels = matrix(d, "overall_util_pct", with_sram=True)
    best = best_mask(m, True)
    text = np.vectorize(lambda v: f"{v:.1f}")(m)
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 1 · Compute utilization - all 27 configs",
               "Cell values in %. Utilization = MACs ÷ (cycles × 1,024 PEs). ★ cell = highest utilization. "
               "Higher is better.")
    ax = fig.add_axes([0.17, 0.17, 0.76, 0.62])
    heatmap(ax, m / 100, text, best, keys, labels, True, "utilization (0-100%)", fig, fontsize=6.3)
    fc_max = d[d.layer_type == "FC"].overall_util_pct.max()
    fig.text(0.04, 0.05, "Reading: WS wins only when Npx (output pixels) is large (ResNet Conv2_1a); OS wins on "
             f"layers with fewer pixels and long K; at batch 1 no FC layer exceeds {fc_max:.1f}%.",
             fontsize=9, color=INK2)
    fig.text(0.04, 0.025, "SRAM size does not change utilization in SCALE-Sim's CALC mode, so the 16 / 64 / 256 kB "
             "cells of each array repeat (ties are all marked ★).", fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def page_heat27_runtime(pdf, d):
    m, keys, labels = matrix(d, "total_cycles", with_sram=True)
    best = best_mask(m, False)
    ratio = m / m.min(axis=1, keepdims=True)
    text = np.where(best, np.vectorize(lambda v: f"{v / 1e6:.3f}")(m),
                    np.vectorize(lambda r: f"{r:.2f}×")(ratio))
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 2 · Runtime - all 27 configs",
               "★ cell = fastest, in million cycles. Other cells = times slower than ★. Lower is better.")
    ax = fig.add_axes([0.17, 0.17, 0.76, 0.62])
    heatmap(ax, 1 / ratio, text, best, keys, labels, True, "speed (1 = fastest)", fig, fontsize=5.8)
    fc = d[d.layer_type == "FC"].copy()
    fc["slow"] = fc.total_cycles / fc.groupby("layer").total_cycles.transform("min")
    shape_gap = fc[fc.dataflow != "ws"].groupby(["layer", "dataflow", "sram_kB"]).total_cycles.agg(
        lambda x: x.max() / x.min()).max()
    fig.text(0.04, 0.05, f"Reading: for FC layers the array shape alone changes runtime up to {shape_gap:.1f}× "
             f"(OS: wide best, IS: tall best); WS is up to {fc[fc.dataflow == 'ws'].slow.max():.1f}× slower "
             "than the best on FC layers at batch 1.", fontsize=9, color=INK2)
    fig.text(0.04, 0.025, "SRAM size does not change runtime in SCALE-Sim's CALC mode (no stalls), so the "
             "16 / 64 / 256 kB cells of each array repeat (ties are all marked ★).", fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def page_heat_traffic(pdf, d):
    m, keys, labels = matrix(d, "traffic_MB", with_sram=True)
    best = best_mask(m, False)
    ratio = m / m.min(axis=1, keepdims=True)
    score = 1 / ratio
    text = np.where(best, np.vectorize(lambda v: f"{v:.2f}")(m), np.vectorize(
        lambda r: f"{r:.1f}×" if r < 10 else f"{r:.0f}×")(ratio))
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 3 · Off-chip (DRAM) traffic - all 27 configs",
               "Total DRAM reads + writes. ★ cell = least traffic (in MB). Other cells = times more traffic than ★. "
               "Lower is better.")
    ax = fig.add_axes([0.17, 0.17, 0.76, 0.62])
    heatmap(ax, score, text, best, keys, labels, True, "traffic efficiency (1 = least traffic)", fig, fontsize=6.3)
    conv = d[(d.layer_type == "Conv") & (d.dataflow == "os")]
    p = conv.pivot_table(index=["layer", "arr"], columns="sram_kB", values="traffic_MB")
    os_cut = (p[16] / p[256]).max()
    fc = d[d.layer_type == "FC"]
    fig.text(0.04, 0.05, f"Reading: with OS, growing SRAM 16 → 256 kB cuts conv-layer traffic by up to "
             f"{os_cut:.0f}× (each output is written only once).", fontsize=9, color=INK2)
    fig.text(0.04, 0.025, "FC layers at batch 1 stay at ≈ their weight size (every weight is used once), "
             f"so no dataflow, shape or SRAM size helps (spread ≤ "
             f"{(fc.groupby('layer').traffic_MB.max() / fc.groupby('layer').traffic_MB.min()).max():.2f}×).",
             fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def page_bars_traffic(pdf, d, order):
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 3 · Off-chip traffic per layer (MB)",
               "27 configs per layer: color = dataflow, cluster = array shape, light → dark = SRAM 16 / 64 / 256 kB. "
               "★ + black outline = least traffic.")
    small_multiples(fig, d, order, lambda ax, s: bars_27(
        ax, s, "traffic_MB", lambda v: f"{v:.2f}", "MB"), 4, 2, top=0.82, bottom=0.12, wspace=0.14, hspace=0.95)
    extra = [Patch(color=tint(INK2, SRAM_TINT[k]), label=f"{k} kB") for k in SRAMS]
    legend_dataflows(fig, y=0.06, extra=extra)
    pdf.savefig(fig); return fig


def page_heat_bw(pdf, d):
    m, keys, labels = matrix(d, "bw_Bpc", with_sram=True)
    best = best_mask(m, False)
    score = m.min(axis=1, keepdims=True) / m
    text = np.vectorize(lambda v: f"{v:.1f}")(m)
    # mark the fastest config's demand with a diamond (fastest = min cycles; ties -> all)
    cyc, _, _ = matrix(d, "total_cycles", with_sram=True)
    fastest = best_mask(cyc, False)
    text = np.where(fastest & ~best, np.char.add("◆ ", text.astype(str)), text)
    fig = plt.figure(figsize=A4)
    page_title(fig, "Result 3 · Average off-chip bandwidth demand (bytes / cycle  =  GB/s at 1 GHz)",
               "DRAM traffic ÷ runtime. ★ = lowest demand.  ◆ = demand of the fastest config(s) for that layer. "
               "Lower demand is easier on DRAM - but slow configs look 'cheap' here.")
    ax = fig.add_axes([0.17, 0.17, 0.76, 0.62])
    heatmap(ax, score, text, best, keys, labels, True, "lower demand (1 = lowest)", fig, fontsize=6.3)
    fc = np.array([lbl.split("\n")[0].split()[-1].startswith("FC") for lbl in labels])
    fast_fc = m[fc][fastest[fc]]
    a2 = d[(d.model == "AlexNet") & (d.layer == "Conv2") & (d.dataflow == "os") & (d.arr == "32x32")]
    a2 = a2.set_index("sram_kB").bw_Bpc
    fig.text(0.04, 0.05, f"Reading: the fastest FC configs (◆) need {fast_fc.min():.0f}-{fast_fc.max():.0f} B/cycle "
             "while using ≈ 6% of the PEs - FC layers at batch 1 are memory-bound.", fontsize=9, color=INK2)
    fig.text(0.04, 0.025, f"For conv layers SRAM decides the demand of the same fast config: AlexNet Conv2 OS 32×32 "
             f"needs {a2[16]:.1f} B/cycle with 16 kB but {a2[256]:.1f} B/cycle with 256 kB.",
             fontsize=9, color=INK2)
    pdf.savefig(fig); return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", help="also save each page as PNG here (for checking)")
    ap.add_argument("--full", action="store_true",
                    help="detailed 8-page version (cover, bars, bandwidth) -> scalesim_results_full.pdf")
    args = ap.parse_args()
    d, order = load()
    if args.full:
        out = OUT_PDF.with_name("scalesim_results_full.pdf")
        pages = [
            lambda p: page_cover(p, d, order),
            lambda p: page_heat_util(p, d), lambda p: page_bars_util(p, d, order),
            lambda p: page_heat_runtime(p, d), lambda p: page_bars_runtime(p, d, order),
            lambda p: page_heat_traffic(p, d), lambda p: page_bars_traffic(p, d, order),
            lambda p: page_heat_bw(p, d),
        ]
    else:  # simple: the 3 results, each as one 7 x 27 heatmap of all 189 runs
        out = OUT_PDF
        pages = [lambda p: page_heat27_util(p, d), lambda p: page_heat27_runtime(p, d),
                 lambda p: page_heat_traffic(p, d)]
    with PdfPages(out) as pdf:
        for i, mk in enumerate(pages, 1):
            fig = mk(pdf)
            if args.png_dir:
                Path(args.png_dir).mkdir(parents=True, exist_ok=True)
                fig.savefig(Path(args.png_dir) / f"page{i}.png", dpi=110)
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "SCALE-Sim dataflow / array shape / SRAM trade-offs"
    print(f"Wrote {out} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
