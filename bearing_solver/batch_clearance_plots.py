"""
Пакетный скрипт: сохраняет 2D-графики зазора H(φ) при Z=0
для всех 10 типов углублений × 3 геометрии.

Запуск:
    python -m bearing_solver.batch_clearance_plots [папка]

По умолчанию сохраняет в ./clearance_plots/
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from bearing_solver.variants import get_variant, GEOMETRIES, DEPRESSION_TYPES
from bearing_solver.geometry import (
    make_grid, compute_depression_centers,
    base_clearance, create_H_with_depressions,
)


def plot_clearance_2d_section(params, epsilon=0.6, num_phi=500, num_Z=500):
    """
    Строит 2D-график H(φ) при Z=0 — сечение зазора
    с углублениями и без.
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, _, _ = make_grid(num_phi, num_Z)
    phi_c_flat, Z_c_flat = compute_depression_centers(params)

    H0 = base_clearance(epsilon, Phi_mesh)
    H_dep = create_H_with_depressions(H0, params, Phi_mesh, Z_mesh,
                                       phi_c_flat, Z_c_flat)

    Z_idx = np.argmin(np.abs(Z_1D - 0.0))

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(phi_1D, H0[Z_idx, :],
            label="Без углублений", color="blue", linewidth=1.5)
    ax.plot(phi_1D, H_dep[Z_idx, :],
            label=params["depression_name"], color="red", linewidth=1.5)
    ax.set_xlabel("φ, рад")
    ax.set_ylabel("H")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "clearance_plots"

    for geom_key in ["A", "B", "C"]:
        geom_dir = os.path.join(output_dir, f"geometry_{geom_key}")
        os.makedirs(geom_dir, exist_ok=True)

        for dep_type in range(1, 11):
            # Номер варианта: A→1-10, B→11-20, C→21-30
            geom_idx = {"A": 0, "B": 1, "C": 2}[geom_key]
            variant = geom_idx * 10 + dep_type
            params = get_variant(variant)

            print(f"Геометрия {geom_key}, тип {dep_type:2d}: "
                  f"{params['depression_name']}...", end=" ")

            fig = plot_clearance_2d_section(params)
            safe_name = (params['depression_name']
                         .replace(" ", "_").replace(".", "")
                         .replace("(", "").replace(")", ""))
            filename = f"clearance_type_{dep_type:02d}_{safe_name}.png"
            fig.savefig(os.path.join(geom_dir, filename), dpi=150)
            print("OK")

    print(f"\nВсе графики сохранены в: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
