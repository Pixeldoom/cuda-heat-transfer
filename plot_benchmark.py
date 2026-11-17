#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_benchmark.py - визуализация результатов multi-GPU бенчмарка стенсила.
Поддерживает sweep по размеру сетки (dim) и типу данных (precision: float/double).

Вход : CSV файлы stencil_bench (новый формат:
       variant,precision,gpus,dim,iters,warmup,reps,median_ms,min_ms,mcells_s,
       speedup,efficiency_pct,max_diff)
       или старый формат (variant,gpus,dim,steps,avg_ms,mcells_s) -> precision=float.

Выход (в каталоге --outdir):
       scaling_dim<N>_<prec>.pdf/png   - панель 2x2: время, ускорение, эффективность,
                                         производительность (для каждой пары dim/precision)
       overhead_dim<N>_<prec>.pdf/png  - накладные расходы на обмен: доля и абсолютные
       compare_dim<N>.pdf/png          - float против double при одном размере сетки
       sweep_<prec>.pdf/png            - влияние размера сетки на E(p) и долю обмена
       summary_baseline.pdf/png        - T1(dim) для обеих точностей и их отношение
       aggregated_results.csv          - объединённые данные с пересчитанными метриками
       + отчёт о корректности (max_diff против эталона 1 GPU) в stdout.

Запуск:
    python plot_benchmark.py benchmark_results.csv [ещё.csv ...] [-o figures]
           [--formats pdf png] [--dpi 300] [--width 7.2] [--no-latex]
           [--dims 2048 4096] [--precisions float double]
           [--figures scaling overhead compare sweep baseline]
           [--sweep-variant auto] [--tol 0.0]

Зависимости: numpy, pandas, matplotlib.
LaTeX опционален: для кириллицы нужны babel-russian и cm-super (T2A);
при их отсутствии используется mathtext (DejaVu Serif).
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------- настройки -------------------------------

PREC_BYTES = {"float": 4, "double": 8}
PREC_ORDER = ["float", "double"]
PREC_LS = {"float": "-", "double": "--"}          # точность -> тип линии (compare)

# Стили, различимые в ч/б печати: маркер + тип линии + уровень серого + штриховка.
# ключ -> (подпись, маркер, заливка маркера, тип линии, серый, штриховка)
VARIANTS = {
    "stencil_p2p":       ("P2P-обмен гало",          "o", "black", "-",  "0.00", "///"),
    "stencil_host_halo": ("Обмен гало через хост",   "s", "white", "--", "0.30", "xxx"),
    "stencil_host_full": ("Полная копия через хост", "^", "0.55",  ":",  "0.55", "\\\\"),
    "stencil_direct":    ("P2P-обмен гало",          "o", "black", "-",  "0.00", "///"),
    "stencil_host":      ("Полная копия через хост", "^", "0.55",  ":",  "0.55", "\\\\"),
}
ORDER = ["stencil_p2p", "stencil_host_halo", "stencil_host_full",
         "stencil_direct", "stencil_host"]
PREFER = ["stencil_p2p", "stencil_direct", "stencil_host_halo",
          "stencil_host", "stencil_host_full"]

IDEAL_STYLE = dict(color="0.55", ls=(0, (1, 2)), lw=1.0)
REF_STYLE = dict(color="0.25", ls=(0, (4, 2)), lw=1.0)

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
GRAYS = ["0.00", "0.35", "0.55", "0.20", "0.45", "0.65"]
LINES = ["-", "--", ":", "-."]
HATCHES = ["///", "xxx", "\\\\\\", "...", "+++", "ooo", "|||"]

FIGURE_KINDS = ["scaling", "overhead", "compare", "sweep", "baseline"]
CREATED = []

LATEX_PREAMBLE = (
    r"\usepackage[T2A]{fontenc}" "\n"
    r"\usepackage[utf8]{inputenc}" "\n"
    r"\usepackage[russian]{babel}"
)


def setup_style(use_latex: bool) -> bool:
    """Единый стиль оформления; возвращает True, если включён usetex."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "lines.markersize": 5,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "axes.grid": True, "grid.color": "0.85",
        "grid.linewidth": 0.5, "grid.linestyle": "-",
        "legend.frameon": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    })
    if use_latex:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
        try:  # пробный рендер: проверяем LaTeX и кириллицу (T2A + babel-russian)
            fig = plt.figure(figsize=(1.0, 0.5))
            fig.text(0, 0, r"Проверка кириллицы: $T_1/T_p$")
            fig.canvas.draw()
            plt.close(fig)
        except Exception:
            plt.close("all")
            plt.rcParams["text.usetex"] = False
            plt.rcParams["text.latex.preamble"] = ""
            warnings.warn("LaTeX с кириллицей недоступен (нужны babel-russian и cm-super); "
                          "используется mathtext.")
    return plt.rcParams["text.usetex"]


def L(s: str) -> str:
    """Экранирование '%' для usetex (mathtext воспринимает его буквально)."""
    return re.sub(r"(?<!\\)%", r"\\%", s) if plt.rcParams["text.usetex"] else s


def series_style(i: int) -> dict:
    """Стиль i-й серии для графиков по размерам сетки / числу GPU."""
    return dict(marker=MARKERS[i % len(MARKERS)],
                color=GRAYS[i % len(GRAYS)],
                ls=LINES[i % len(LINES)],
                mfc="white" if i % 2 else GRAYS[i % len(GRAYS)],
                mec=GRAYS[i % len(GRAYS)])


def hatch(i: int) -> str:
    return HATCHES[i % len(HATCHES)]


# --------------------------- загрузка и агрегация ---------------------------

def normalize_precision(value) -> str:
    s = str(value).strip().lower()
    if s in ("float", "f", "single", "float32", "fp32"):
        return "float"
    if s in ("double", "d", "float64", "fp64"):
        return "double"
    raise ValueError(f"неизвестный тип данных: {value!r} (ожидается float или double)")


def load_results(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        if "median_ms" not in df and "avg_ms" in df:          # старый формат
            df = df.rename(columns={"avg_ms": "median_ms"})
        missing = {"variant", "gpus", "dim", "median_ms"} - set(df.columns)
        if missing:
            raise ValueError(f"{path}: отсутствуют колонки {sorted(missing)}")
        if "precision" not in df:                              # старые файлы - только float
            df["precision"] = "float"
        if "min_ms" not in df:
            df["min_ms"] = df["median_ms"]
        if "max_diff" not in df:
            df["max_diff"] = 0.0
        df["precision"] = df["precision"].map(normalize_precision)
        df["source"] = Path(path).name
        frames.append(df[["variant", "precision", "gpus", "dim",
                          "median_ms", "min_ms", "max_diff", "source"]])

    df = pd.concat(frames, ignore_index=True)
    df["gpus"] = pd.to_numeric(df["gpus"], errors="coerce")
    df["dim"] = pd.to_numeric(df["dim"], errors="coerce")
    for c in ("median_ms", "min_ms", "max_diff"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["gpus", "dim", "median_ms"])
    df = df[(df["gpus"] >= 1) & (df["dim"] >= 4) & (df["median_ms"] > 0)]
    df["gpus"] = df["gpus"].astype(int)
    df["dim"] = df["dim"].astype(int)
    df["min_ms"] = df[["min_ms", "median_ms"]].min(axis=1)     # защита от min > median
    return df.reset_index(drop=True)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Объединение повторных прогонов одной конфигурации из разных файлов."""
    g = (df.groupby(["dim", "precision", "variant", "gpus"], as_index=False)
           .agg(median_ms=("median_ms", "median"),
                min_ms=("min_ms", "min"),
                max_diff=("max_diff", "max"),
                n_runs=("median_ms", "size")))
    return g.sort_values(["dim", "precision", "variant", "gpus"]).reset_index(drop=True)


def add_metrics(g: pd.DataFrame):
    """Метрики внутри группы (dim, precision). Возвращает (T1, T1_min) или None."""
    base = g[g["gpus"] == 1]
    if base.empty:
        return None
    for preferred in ("stencil_single", "stencil_direct"):
        row = base[base["variant"] == preferred]
        if not row.empty:
            base = row
            break
    T1 = float(base["median_ms"].min())
    T1_min = float(base["min_ms"].min())

    dim = int(g["dim"].iloc[0])
    prec = str(g["precision"].iloc[0])
    cells = float(dim) ** 2

    g["t1_ms"] = T1
    g["speedup"] = T1 / g["median_ms"]
    g["efficiency_pct"] = 100.0 * g["speedup"] / g["gpus"]
    g["ideal_ms"] = T1 / g["gpus"]
    g["overhead_ms"] = (g["median_ms"] - g["ideal_ms"]).clip(lower=0.0)
    g["comm_fraction_pct"] = 100.0 * g["overhead_ms"] / g["median_ms"]
    g["mcells_s"] = cells / (g["median_ms"] * 1e3)
    # минимальный трафик: чтение u^n + запись u^(n+1)
    g["bw_gbs"] = 2.0 * cells * PREC_BYTES[prec] / (g["median_ms"] * 1e6)
    return T1, T1_min


def active_variants(g: pd.DataFrame):
    present = set(g["variant"])
    return [v for v in ORDER if v in present]


def pick_variant(g: pd.DataFrame, requested: str = "auto"):
    """Выбор режима обмена для сводных графиков по размерам сетки."""
    present = set(g["variant"])
    if requested and requested != "auto":
        if requested in present:
            return requested
        warnings.warn(f"режим {requested} отсутствует в данных; выбирается автоматически")
    for v in PREFER:
        if v in present and not g[(g["variant"] == v) & (g["gpus"] > 1)].empty:
            return v
    return None


def save_figure(fig, stem: Path, formats, dpi):
    for ext in formats:
        name = stem.with_suffix("." + ext)
        fig.savefig(name, dpi=dpi, bbox_inches="tight")
        CREATED.append(name.name)
    plt.close(fig)


# ------------------------------- рисунки -------------------------------

def plot_scaling(g, dim, prec, base, outdir, args):
    """Панель 2x2: время шага, ускорение, эффективность, производительность."""
    T1, T1_min = base
    cells = float(dim) ** 2
    ps = np.array(sorted(g["gpus"].unique()))
    pfit = np.linspace(1.0, float(ps.max()), 200)

    fig, axes = plt.subplots(2, 2, figsize=(args.width, 0.80 * args.width), sharex=True)
    (ax_t, ax_s), (ax_e, ax_m) = axes
    handles = {}

    for v in active_variants(g):
        sub = g[(g["variant"] == v) & (g["gpus"] > 1)].sort_values("gpus")
        if sub.empty:
            continue
        label, marker, mfc, ls, gray, _ = VARIANTS[v]
        p = np.concatenate(([1], sub["gpus"].to_numpy()))
        t = np.concatenate(([T1], sub["median_ms"].to_numpy()))
        tmin = np.concatenate(([T1_min], sub["min_ms"].to_numpy()))
        lower = np.maximum(t - tmin, 0.0)
        style = dict(ls=ls, marker=marker, color=gray, mfc=mfc, mec=gray)
        ax_t.errorbar(p, t, yerr=[lower, np.zeros_like(t)],
                      capsize=2.5, elinewidth=0.8, **style)
        ax_s.plot(p, T1 / t, **style)
        ax_e.plot(p, 100.0 * (T1 / t) / p, **style)
        ax_m.plot(p, cells / (t * 1e3), **style)
        handles[label] = Line2D([], [], label=label, **style)

    ax_t.plot(pfit, T1 / pfit, **IDEAL_STYLE)
    ax_s.plot(pfit, pfit, **IDEAL_STYLE)
    ax_e.axhline(100.0, **IDEAL_STYLE)
    ax_m.plot(pfit, pfit * cells / (T1 * 1e3), **IDEAL_STYLE)
    handles["Идеальное масштабирование"] = Line2D(
        [], [], label="Идеальное масштабирование", **IDEAL_STYLE)

    ax_t.set_yscale("log")
    ax_t.set_ylabel(L(r"Время шага, $T_p$ (мс)"))
    ax_s.set_ylabel(L(r"Ускорение, $S(p) = T_1 / T_p$"))
    ax_e.set_ylabel(L(r"Параллельная эффективность, $E(p)$ (%)"))
    ax_m.set_ylabel(L(r"Производительность ($10^6$ ячеек/с)"))
    for ax, title in zip(axes.ravel(),
                         ["(а) Время шага", "(б) Ускорение",
                          "(в) Параллельная эффективность", "(г) Производительность"]):
        ax.set_title(L(title))
    for ax in (ax_s, ax_e, ax_m):
        ax.set_ylim(bottom=0)
    ax_e.set_ylim(0, max(110.0, 1.08 * float(g["efficiency_pct"].max())))
    for ax in (ax_e, ax_m):
        ax.set_xlabel(L(r"Число GPU, $p$"))
    ax_t.set_xticks(ps)
    ax_t.set_xlim(0.8, float(ps.max()) + 0.2)

    fig.suptitle(L(rf"Сильное масштабирование, сетка {dim}$\times${dim}, {prec}"))
    if handles:
        fig.legend(handles=list(handles.values()), loc="lower center",
                   ncol=min(2, len(handles)), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    save_figure(fig, outdir / f"scaling_dim{dim}_{prec}", args.formats, args.dpi)


def plot_overhead(g, dim, prec, outdir, args):
    """Накладные расходы на обмен: доля (%) и абсолютная оценка (мс)."""
    multi = g[g["gpus"] > 1]
    variants = [v for v in active_variants(multi)
                if not multi[multi["variant"] == v].empty]
    if multi.empty or not variants:
        return
    ps = sorted(multi["gpus"].unique())
    x = np.arange(len(ps))
    n = len(variants)
    bw = 0.8 / n

    fig, (ax_f, ax_o) = plt.subplots(1, 2, figsize=(args.width, 0.45 * args.width))
    handles = {}
    for i, v in enumerate(variants):
        label, _, _, _, gray, h = VARIANTS[v]
        sub = multi[multi["variant"] == v].set_index("gpus").reindex(ps)
        frac = sub["comm_fraction_pct"].to_numpy(dtype=float)
        over = sub["overhead_ms"].to_numpy(dtype=float)
        pos = x + (i - (n - 1) / 2.0) * bw
        bar = dict(width=bw * 0.92, facecolor="white", edgecolor=gray,
                   hatch=h, linewidth=0.8)
        ok = np.isfinite(frac)
        ax_f.bar(pos[ok], frac[ok], **bar)
        ok_o = ok & np.isfinite(over) & (over > 0)
        ax_o.bar(pos[ok_o], over[ok_o], **bar)
        handles[label] = Patch(label=label, facecolor="white", edgecolor=gray,
                               hatch=h, linewidth=0.8)

    if (multi["overhead_ms"] > 0).any():
        ax_o.set_yscale("log")
    ax_f.set_ylim(0, 105)
    ax_f.set_ylabel(L("Доля обмена данными (%)"))
    ax_o.set_ylabel(L("Оценка накладных расходов (мс)"))
    ax_f.set_title(L("(а) Относительные накладные расходы"))
    ax_o.set_title(L("(б) Абсолютные накладные расходы"))
    for ax in (ax_f, ax_o):
        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in ps])
        ax.set_xlabel(L(r"Число GPU, $p$"))
        ax.grid(axis="x", visible=False)

    fig.suptitle(L(rf"Накладные расходы на обмен, сетка {dim}$\times${dim}, {prec}"))
    if handles:
        fig.legend(handles=list(handles.values()), loc="lower center",
                   ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 0.05))
    fig.text(0.5, 0.005,
             L(r"Оценка накладных расходов: $T_p - T_1/p$ (отрицательные значения обнулены)"),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    save_figure(fig, outdir / f"overhead_dim{dim}_{prec}", args.formats, args.dpi)


def plot_compare(gd, dim, outdir, args):
    """float против double при одном размере сетки: время и их отношение."""
    precs = [p for p in PREC_ORDER if p in set(gd["precision"])]
    if len(precs) < 2:
        return
    all_p = sorted(gd["gpus"].unique())
    pfit = np.linspace(1.0, float(max(all_p)), 100)

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(args.width, 0.45 * args.width))
    handles_t, handles_r = {}, {}

    for prec in precs:
        gp = gd[gd["precision"] == prec]
        T1 = float(gp["t1_ms"].iloc[0])
        for v in active_variants(gp):
            sub = gp[(gp["variant"] == v) & (gp["gpus"] > 1)].sort_values("gpus")
            if sub.empty:
                continue
            label, marker, _, _, gray, _ = VARIANTS[v]
            ls = PREC_LS[prec]
            mfc = gray if prec == "float" else "white"
            style = dict(ls=ls, marker=marker, color=gray, mfc=mfc, mec=gray)
            p = np.concatenate(([1], sub["gpus"].to_numpy()))
            t = np.concatenate(([T1], sub["median_ms"].to_numpy()))
            key = (label, prec)
            ax_t.plot(p, t, **style)
            handles_t[key] = Line2D([], [], label=f"{label}, {prec}", **style)

    ax_t.set_yscale("log")
    ax_t.set_ylabel(L(r"Время шага, $T_p$ (мс)"))
    ax_t.set_xlabel(L(r"Число GPU, $p$"))
    ax_t.set_title(L("(а) Время шага для разных типов данных"))
    ax_t.set_xticks(all_p)
    ax_t.set_xlim(0.8, float(max(all_p)) + 0.2)

    # отношение времён double/float по каждому режиму обмена
    for i, v in enumerate(active_variants(gd)):
        a = gd[(gd["precision"] == "float") & (gd["variant"] == v) & (gd["gpus"] > 1)]
        b = gd[(gd["precision"] == "double") & (gd["variant"] == v) & (gd["gpus"] > 1)]
        if a.empty or b.empty:
            continue
        a = a.set_index("gpus")["median_ms"]
        b = b.set_index("gpus")["median_ms"]
        common = sorted(a.index.intersection(b.index))
        if not common:
            continue
        ratio = b.loc[common].to_numpy() / a.loc[common].to_numpy()
        label, marker, _, _, gray, _ = VARIANTS[v]
        style = dict(ls="-", marker=marker, color=gray, mfc="white", mec=gray)
        ax_r.plot(common, ratio, **style)
        handles_r[label] = Line2D([], [], label=label, **style)

    t1f = float(gd[gd["precision"] == "float"]["t1_ms"].iloc[0])
    t1d = float(gd[gd["precision"] == "double"]["t1_ms"].iloc[0])
    ax_r.plot([1], [t1d / t1f], ls="none", marker="*", color="black", markersize=9)
    handles_r["Эталон (1 GPU)"] = Line2D([], [], ls="none", marker="*",
                                         color="black", label="Эталон (1 GPU)")
    ax_r.axhline(2.0, **REF_STYLE)
    handles_r["Двукратный трафик"] = Line2D([], [], label="Двукратный трафик", **REF_STYLE)

    ax_r.set_ylabel(L(r"Отношение $T_p^{\,\mathrm{double}} / T_p^{\,\mathrm{float}}$"))
    ax_r.set_xlabel(L(r"Число GPU, $p$"))
    ax_r.set_title(L("(б) Замедление при переходе на double"))
    ax_r.set_xticks(all_p)
    ax_r.set_xlim(0.8, float(max(all_p)) + 0.2)
    ax_r.set_ylim(bottom=0)

    fig.suptitle(L(rf"Сравнение точностей, сетка {dim}$\times${dim}"))
    hs = list(handles_t.values()) + list(handles_r.values())
    if hs:
        fig.legend(handles=hs, loc="lower center",
                   ncol=min(3, len(hs)), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))
    save_figure(fig, outdir / f"compare_dim{dim}", args.formats, args.dpi)


def plot_sweep(gp, prec, outdir, args):
    """Влияние размера сетки: E(p) для разных dim и доля обмена от dim."""
    v = pick_variant(gp, args.sweep_variant)
    if v is None:
        return
    sub = gp[gp["variant"] == v]
    dims = sorted(sub["dim"].unique())
    if len(dims) < 2:
        return
    label = VARIANTS[v][0]

    fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(args.width, 0.45 * args.width))
    handles_e, handles_f = {}, {}

    for i, dd in enumerate(dims):
        s = sub[sub["dim"] == dd].sort_values("gpus")
        p = s["gpus"].to_numpy(dtype=float)
        e = s["efficiency_pct"].to_numpy(dtype=float)
        if 1 not in p:                      # E(1) = 100% по определению
            p = np.concatenate(([1.0], p))
            e = np.concatenate(([100.0], e))
        style = series_style(i)
        ax_e.plot(p, e, **style)
        handles_e[dd] = Line2D([], [], label=rf"{dd}$\times${dd}", **style)

    ax_e.axhline(100.0, **IDEAL_STYLE)
    ax_e.set_ylim(0, max(110.0, 1.08 * float(sub["efficiency_pct"].max())))
    ax_e.set_ylabel(L(r"Параллельная эффективность, $E(p)$ (%)"))
    ax_e.set_xlabel(L(r"Число GPU, $p$"))
    ax_e.set_title(L("(а) Эффективность при разных размерах сетки"))
    all_p = sorted(sub["gpus"].unique())
    ax_e.set_xticks(all_p)
    ax_e.set_xlim(0.8, float(max(all_p)) + 0.2)

    for j, p in enumerate([p for p in all_p if p > 1]):
        s = sub[sub["gpus"] == p].sort_values("dim")
        if s.empty:
            continue
        style = series_style(j)
        ax_f.plot(s["dim"].to_numpy(dtype=float),
                  s["comm_fraction_pct"].to_numpy(dtype=float), **style)
        handles_f[p] = Line2D([], [], label=rf"$p = {p}$", **style)

    ax_f.set_xscale("log", base=2)
    ax_f.set_xticks(dims)
    ax_f.set_xticklabels([str(d) for d in dims])
    ax_f.minorticks_off()
    ax_f.set_ylim(bottom=0)
    ax_f.set_ylabel(L("Доля обмена данными (%)"))
    ax_f.set_xlabel(L(r"Размер сетки, dim (сторона квадрата)"))
    ax_f.set_title(L("(б) Доля обмена от размера задачи"))

    fig.suptitle(L(rf"Влияние размера сетки, {prec}; режим: {label}"))
    hs = list(handles_e.values()) + list(handles_f.values())
    if hs:
        fig.legend(handles=hs, loc="lower center",
                   ncol=min(4, len(hs)), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    save_figure(fig, outdir / f"sweep_{prec}", args.formats, args.dpi)


def plot_baseline(data, outdir, args):
    """T1(dim) для обеих точностей и отношение double/float (проверка memory-bound)."""
    precs = [p for p in PREC_ORDER if p in set(data["precision"])]
    if len(precs) < 2:
        return
    base = (data[data["gpus"] == 1][["dim", "precision", "t1_ms"]]
            .drop_duplicates(subset=["dim", "precision"]))
    dims = sorted(base["dim"].unique())
    if len(dims) < 2:
        return

    fig, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(args.width, 0.45 * args.width))
    handles = {}
    for i, prec in enumerate(precs):
        s = base[base["precision"] == prec].sort_values("dim")
        style = dict(ls=PREC_LS[prec], marker=MARKERS[i], color=GRAYS[i],
                     mfc="white" if i else GRAYS[i], mec=GRAYS[i])
        ax_t.plot(s["dim"].to_numpy(dtype=float), s["t1_ms"].to_numpy(dtype=float), **style)
        handles[prec] = Line2D([], [], label=prec, **style)

    s0 = base[base["precision"] == precs[0]].sort_values("dim")
    d0 = float(s0["dim"].iloc[0])
    t0 = float(s0["t1_ms"].iloc[0])
    dfit = np.linspace(dims[0], dims[-1], 100)
    ax_t.plot(dfit, t0 * (dfit / d0) ** 2, **REF_STYLE)
    handles["Квадратичный рост"] = Line2D([], [], label=r"Рост $\propto \mathrm{dim}^2$",
                                          **REF_STYLE)

    ax_t.set_xscale("log", base=2)
    ax_t.set_yscale("log")
    ax_t.set_xticks(dims)
    ax_t.set_xticklabels([str(d) for d in dims])
    ax_t.minorticks_off()
    ax_t.set_ylabel(L(r"Время шага на 1 GPU, $T_1$ (мс)"))
    ax_t.set_xlabel(L("Размер сетки, dim (сторона квадрата)"))
    ax_t.set_title(L("(а) Базовое время от размера задачи"))

    a = base[base["precision"] == "float"].set_index("dim")["t1_ms"]
    b = base[base["precision"] == "double"].set_index("dim")["t1_ms"]
    common = sorted(a.index.intersection(b.index))
    if common:
        ratio = b.loc[common].to_numpy() / a.loc[common].to_numpy()
        ax_r.plot(common, ratio, ls="-", marker="o", color="black",
                  mfc="white", mec="black")
    ax_r.axhline(2.0, **REF_STYLE)
    ax_r.set_xscale("log", base=2)
    ax_r.set_xticks(dims)
    ax_r.set_xticklabels([str(d) for d in dims])
    ax_r.minorticks_off()
    ax_r.set_ylim(0, 2.6)
    ax_r.set_ylabel(L(r"Отношение $T_1^{\,\mathrm{double}} / T_1^{\,\mathrm{float}}$"))
    ax_r.set_xlabel(L("Размер сетки, dim (сторона квадрата)"))
    ax_r.set_title(L("(б) Цена удвоения трафика памяти"))

    fig.suptitle(L("Базовая производительность одиночного GPU"))
    if handles:
        fig.legend(handles=list(handles.values()), loc="lower center",
                   ncol=min(3, len(handles)), bbox_to_anchor=(0.5, 0.02))
    fig.text(0.5, 0.005,
             L("Отношение около 2 указывает на режим, ограниченный пропускной способностью памяти"),
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    save_figure(fig, outdir / "summary_baseline", args.formats, args.dpi)


# --------------------------------- отчёт ---------------------------------

def report(g, dim, prec, tol):
    cols = ["variant", "gpus", "median_ms", "min_ms", "mcells_s", "bw_gbs",
            "speedup", "efficiency_pct", "max_diff", "n_runs"]
    print(f"\n=== Сетка {dim}x{dim}, {prec} ({PREC_BYTES[prec]} Б/ячейка) ===")
    print(g[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    bad = g[g["max_diff"] > tol]
    if not bad.empty:
        print(f"ВНИМАНИЕ: проверка корректности НЕ пройдена (max_diff > {tol:g}):")
        print(bad[["variant", "gpus", "max_diff"]].to_string(index=False))
    else:
        print(f"Проверка корректности пройдена: max_diff <= {tol:g} для всех конфигураций.")
    if (g["median_ms"] < g["ideal_ms"]).any():
        print("Примечание: сверхлинейное ускорение (вероятно, эффект кэша L2); "
              "накладные расходы обнулены.")


# ---------------------------------- main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSV-файл(ы) бенчмарка")
    ap.add_argument("-o", "--outdir", default="figures", help="каталог для рисунков")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    choices=["pdf", "png", "svg", "eps"], help="форматы рисунков")
    ap.add_argument("--dpi", type=int, default=300, help="dpi растровых форматов")
    ap.add_argument("--width", type=float, default=7.2,
                    help="ширина рисунка в дюймах (по умолчанию двухколоночный)")
    ap.add_argument("--no-latex", action="store_true", help="не использовать usetex")
    ap.add_argument("--dims", nargs="*", type=int, default=None,
                    help="строить только для указанных размеров сетки")
    ap.add_argument("--precisions", nargs="*", default=None, choices=PREC_ORDER,
                    help="строить только для указанных типов данных")
    ap.add_argument("--figures", nargs="*", default=None, choices=FIGURE_KINDS,
                    help="какие рисунки строить (по умолчанию все)")
    ap.add_argument("--sweep-variant", default="auto",
                    help="режим обмена для сводных графиков по размерам (auto)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="допустимое max_diff (внутри одной точности ожидается 0)")
    args = ap.parse_args()

    usetex = setup_style(use_latex=not args.no_latex)
    print(f"Рендеринг текста: {'LaTeX (usetex)' if usetex else 'mathtext (запасной вариант)'}")

    def want(kind):
        return args.figures is None or kind in args.figures

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv)
    if args.dims:
        df = df[df["dim"].isin(args.dims)]
    if args.precisions:
        df = df[df["precision"].isin(args.precisions)]
    if df.empty:
        print("После фильтрации данных не осталось.")
        return
    g = aggregate(df)

    parts = []
    for (dim, prec), gg in g.groupby(["dim", "precision"]):
        gg = gg.copy()
        base = add_metrics(gg)
        if base is None:
            print(f"[пропуск] dim={dim}, {prec}: нет эталонной строки с 1 GPU (gpus=1)")
            continue
        report(gg, dim, prec, args.tol)
        if want("scaling"):
            plot_scaling(gg, dim, prec, base, outdir, args)
        if want("overhead"):
            plot_overhead(gg, dim, prec, outdir, args)
        parts.append(gg)

    if not parts:
        print("Нечего строить: ни одной группы с эталоном на 1 GPU.")
        return

    data = pd.concat(parts, ignore_index=True)
    data.to_csv(outdir / "aggregated_results.csv", index=False)

    if want("compare"):
        for dim, gd in data.groupby("dim"):
            plot_compare(gd, dim, outdir, args)
    if want("sweep"):
        for prec, gp in data.groupby("precision"):
            plot_sweep(gp, prec, outdir, args)
    if want("baseline"):
        plot_baseline(data, outdir, args)

    print(f"\nСохранено в '{outdir}/': aggregated_results.csv и {len(CREATED)} файл(ов) рисунков")
    for name in CREATED:
        print("  " + name)


if __name__ == "__main__":
    main()