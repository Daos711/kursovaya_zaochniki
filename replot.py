#!/usr/bin/env python3
"""
Перерисовка графиков из сохранённых данных (без пересчёта).

Запуск:
  python replot.py                   # все варианты
  python replot.py 1 3 5             # только указанные
"""

import sys
import os
import numpy as np

from bearing_solver.variants import get_variant
from bearing_solver.postprocess import (
    plot_clearance_2d_section,
    plot_pressure_2d_section,
    plot_F_vs_epsilon,
    plot_mu_vs_epsilon,
    plot_Q_vs_epsilon,
    _plot_3d_field,
)

ALL_VARIANTS = [1, 3, 5, 11, 12, 13, 15, 17, 18, 19, 20, 23]


def replot_variant(var_num, results_root="results"):
    folder = os.path.join(results_root, f"var_{var_num:02d}")
    data_dir = os.path.join(folder, "data")
    fig_dir = os.path.join(folder, "figures")
    npz_path = os.path.join(data_dir, "fields.npz")

    if not os.path.exists(npz_path):
        print(f"[var_{var_num:02d}] fields.npz не найден — пропуск")
        return

    params = get_variant(var_num)
    dep_name = params["depression_name"]

    d = np.load(npz_path)
    result = {k: d[k] for k in d.files}
    # скалярные значения из массивов длины 1
    for key in ("F_nd_3d", "F_dep_3d", "mu_nd_3d", "mu_dep_3d",
                "Q_nd_3d", "Q_dep_3d"):
        if key in result and result[key].ndim == 1:
            result[key] = float(result[key][0])

    os.makedirs(fig_dir, exist_ok=True)

    # 2D
    fig = plot_clearance_2d_section(result, dep_name)
    fig.savefig(os.path.join(fig_dir, "clearance_2d.png"), dpi=300)

    fig = plot_pressure_2d_section(result, dep_name)
    fig.savefig(os.path.join(fig_dir, "pressure_2d.png"), dpi=300)

    # 3D
    fig = _plot_3d_field(result, "P_nd_3d", "P_dep_3d", "plasma", "P")
    fig.savefig(os.path.join(fig_dir, "pressure_3d.png"), dpi=300)

    fig = _plot_3d_field(result, "H_nd_3d", "H_dep_3d", "viridis", "H")
    fig.savefig(os.path.join(fig_dir, "clearance_3d.png"), dpi=300)

    # Зависимости от ε
    for name, plot_fn in [("load_vs_epsilon", plot_F_vs_epsilon),
                           ("friction_vs_epsilon", plot_mu_vs_epsilon),
                           ("flow_vs_epsilon", plot_Q_vs_epsilon)]:
        fig = plot_fn(result, dep_name)
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300)

    print(f"[var_{var_num:02d}] {dep_name} — графики перерисованы")


def main():
    results_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "results")

    if len(sys.argv) > 1:
        variants = [int(x) for x in sys.argv[1:]]
    else:
        variants = ALL_VARIANTS

    for v in variants:
        replot_variant(v, results_root)

    print("Готово.")


if __name__ == "__main__":
    main()
