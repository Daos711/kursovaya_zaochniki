"""
Пакетный скрипт: сохраняет 10 графиков зазора H(φ) при Z=0
(по одному на каждый тип углубления, геометрия A).

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

from bearing_solver.variants import get_variant, DEPRESSION_TYPES
from bearing_solver.geometry import (
    make_grid, compute_depression_centers,
    base_clearance, create_H_with_depressions,
)


def plot_clearance_2d_section(params, epsilon=0.6, num_phi=5000, num_Z=500):
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
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "figures"
    os.makedirs(output_dir, exist_ok=True)

    for dep_type in range(1, 11):
        variant = dep_type  # варианты 1-10 = геометрия A, типы 1-10
        params = get_variant(variant)

        print(f"Тип {dep_type:2d}: {params['depression_name']}...", end=" ")

        fig = plot_clearance_2d_section(params)
        filename = f"clearance_type_{dep_type:02d}.png"
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        print("OK")

    print(f"\nВсе графики сохранены в: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
