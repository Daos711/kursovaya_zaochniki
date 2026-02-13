"""
Постобработка: вычисление F, μ, Q и построение графиков.
Расчёт разбит на два этапа для поэтапного отображения в GUI.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from joblib import Parallel, delayed

# numpy >= 2.0 переименовал trapz → trapezoid
_trapz = getattr(np, "trapezoid", np.trapz)

from .solver import solve_reynolds_gauss_seidel_numba, compute_dP_dphi
from .geometry import (base_clearance, create_H_with_depressions,
                       make_grid, compute_depression_centers)


# -------------------------------------------------------------------
#  Вычисление F, μ, Q для одного значения ε
# -------------------------------------------------------------------

def _compute_for_epsilon(epsilon, params, phi_1D, Z_1D,
                         Phi_mesh, Z_mesh, d_phi, d_Z,
                         phi_c_flat, Z_c_flat):
    """Возвращает кортеж результатов для одного ε."""
    R = params["R"]
    c = params["c"]
    L = params["L"]
    U = params["U"]
    eta = params["eta"]

    pressure_scale = 6 * eta * U * R / c ** 2
    load_scale = pressure_scale * R * L / 2
    friction_scale = eta * U * R * L / c

    cos_phi = np.cos(Phi_mesh)
    sin_phi = np.sin(Phi_mesh)

    H0 = base_clearance(epsilon, Phi_mesh)

    # --- Без углублений ---
    H_nd = H0.copy()
    P_nd, _, _ = solve_reynolds_gauss_seidel_numba(
        H_nd, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)

    Fr_nd = _trapz(_trapz(P_nd * cos_phi, phi_1D, axis=1), Z_1D)
    Ft_nd = _trapz(_trapz(P_nd * sin_phi, phi_1D, axis=1), Z_1D)
    F_nd = np.sqrt(Fr_nd ** 2 + Ft_nd ** 2) * load_scale

    dP_nd = compute_dP_dphi(P_nd, d_phi)
    integ_nd = 1.0 / H_nd + 3 * H_nd * dP_nd
    f_nd = _trapz(_trapz(integ_nd, phi_1D, axis=1), Z_1D) * friction_scale
    mu_nd = f_nd / F_nd if F_nd > 0 else 0.0

    q_nd = H_nd - 0.5 * H_nd ** 3 * dP_nd
    Q_nd = U * c * R * _trapz(_trapz(q_nd, phi_1D, axis=1), Z_1D) * 1000

    # --- С углублениями ---
    H_dep = create_H_with_depressions(H0, params, Phi_mesh, Z_mesh,
                                       phi_c_flat, Z_c_flat)
    P_dep, _, _ = solve_reynolds_gauss_seidel_numba(
        H_dep, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)

    Fr_dep = _trapz(_trapz(P_dep * cos_phi, phi_1D, axis=1), Z_1D)
    Ft_dep = _trapz(_trapz(P_dep * sin_phi, phi_1D, axis=1), Z_1D)
    F_dep = np.sqrt(Fr_dep ** 2 + Ft_dep ** 2) * load_scale

    dP_dep = compute_dP_dphi(P_dep, d_phi)
    integ_dep = 1.0 / H_dep + 3 * H_dep * dP_dep
    f_dep = _trapz(_trapz(integ_dep, phi_1D, axis=1), Z_1D) * friction_scale
    mu_dep = f_dep / F_dep if F_dep > 0 else 0.0

    q_dep = H_dep - 0.5 * H_dep ** 3 * dP_dep
    Q_dep = U * c * R * _trapz(_trapz(q_dep, phi_1D, axis=1), Z_1D) * 1000

    return (epsilon,
            F_nd, mu_nd, Q_nd,
            F_dep, mu_dep, Q_dep)


# -------------------------------------------------------------------
#  ЭТАП 1: 3D-поля при заданном ε (давление + зазор)
# -------------------------------------------------------------------

def run_stage1_3d(params, epsilon_3d=0.6, num_phi=500, num_Z=500,
                  progress_callback=None):
    """
    Решает Рейнольдса для одного ε и возвращает 3D-поля.
    GUI может показать графики сразу после этого этапа.
    """
    phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z = make_grid(num_phi, num_Z)
    phi_c_flat, Z_c_flat = compute_depression_centers(params)

    R = params["R"]
    c = params["c"]
    L = params["L"]
    U = params["U"]
    eta = params["eta"]

    pressure_scale = 6 * eta * U * R / c ** 2
    load_scale = pressure_scale * R * L / 2
    friction_scale = eta * U * R * L / c

    cos_phi = np.cos(Phi_mesh)
    sin_phi = np.sin(Phi_mesh)

    if progress_callback:
        progress_callback("stage1_text", "Формирование зазора...")
        progress_callback("stage1_progress", 5)

    # Формирование зазора
    H0_3d = base_clearance(epsilon_3d, Phi_mesh)
    H_nd_3d = H0_3d.copy()
    H_dep_3d = create_H_with_depressions(H0_3d, params, Phi_mesh, Z_mesh,
                                          phi_c_flat, Z_c_flat)
    if progress_callback:
        progress_callback("log", "Зазор сформирован")
        progress_callback("stage1_text", "Рейнольдс без углублений...")
        progress_callback("stage1_progress", 10)

    # Решаем без углублений
    P_nd_3d, _, iter_nd = solve_reynolds_gauss_seidel_numba(
        H_nd_3d, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)
    if progress_callback:
        progress_callback("log", f"Без углублений: {iter_nd} итераций")
        progress_callback("stage1_text", "Рейнольдс с углублениями...")
        progress_callback("stage1_progress", 50)

    # Решаем с углублениями
    P_dep_3d, _, iter_dep = solve_reynolds_gauss_seidel_numba(
        H_dep_3d, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)
    if progress_callback:
        progress_callback("log", f"С углублениями: {iter_dep} итераций")
        progress_callback("stage1_text", "Вычисление F, μ, Q...")
        progress_callback("stage1_progress", 90)

    # Числовые результаты при epsilon_3d
    Fr = _trapz(_trapz(P_nd_3d * cos_phi, phi_1D, axis=1), Z_1D)
    Ft = _trapz(_trapz(P_nd_3d * sin_phi, phi_1D, axis=1), Z_1D)
    F_nd_3d = np.sqrt(Fr ** 2 + Ft ** 2) * load_scale
    dP = compute_dP_dphi(P_nd_3d, d_phi)
    integ = 1.0 / H_nd_3d + 3 * H_nd_3d * dP
    f_val = _trapz(_trapz(integ, phi_1D, axis=1), Z_1D) * friction_scale
    mu_nd_3d = f_val / F_nd_3d if F_nd_3d > 0 else 0.0
    q_i = H_nd_3d - 0.5 * H_nd_3d ** 3 * dP
    Q_nd_3d = U * c * R * _trapz(_trapz(q_i, phi_1D, axis=1), Z_1D) * 1000

    Fr = _trapz(_trapz(P_dep_3d * cos_phi, phi_1D, axis=1), Z_1D)
    Ft = _trapz(_trapz(P_dep_3d * sin_phi, phi_1D, axis=1), Z_1D)
    F_dep_3d = np.sqrt(Fr ** 2 + Ft ** 2) * load_scale
    dP = compute_dP_dphi(P_dep_3d, d_phi)
    integ = 1.0 / H_dep_3d + 3 * H_dep_3d * dP
    f_val = _trapz(_trapz(integ, phi_1D, axis=1), Z_1D) * friction_scale
    mu_dep_3d = f_val / F_dep_3d if F_dep_3d > 0 else 0.0
    q_i = H_dep_3d - 0.5 * H_dep_3d ** 3 * dP
    Q_dep_3d = U * c * R * _trapz(_trapz(q_i, phi_1D, axis=1), Z_1D) * 1000

    if progress_callback:
        progress_callback("stage1_text", "Готово")
        progress_callback("stage1_progress", 100)

    return {
        "Phi_mesh": Phi_mesh, "Z_mesh": Z_mesh,
        "phi_1D": phi_1D, "Z_1D": Z_1D,
        "d_phi": d_phi, "d_Z": d_Z,
        "phi_c_flat": phi_c_flat, "Z_c_flat": Z_c_flat,
        "H_nd_3d": H_nd_3d, "H_dep_3d": H_dep_3d,
        "P_nd_3d": P_nd_3d, "P_dep_3d": P_dep_3d,
        "F_nd_3d": F_nd_3d, "mu_nd_3d": mu_nd_3d, "Q_nd_3d": Q_nd_3d,
        "F_dep_3d": F_dep_3d, "mu_dep_3d": mu_dep_3d, "Q_dep_3d": Q_dep_3d,
    }


# -------------------------------------------------------------------
#  ЭТАП 2: Перебор по ε (графики зависимостей)
# -------------------------------------------------------------------

def run_stage2_epsilon_sweep(params, stage1_result, n_jobs=-1,
                             progress_callback=None):
    """
    Перебор по ε с поточечным отчётом прогресса.
    Использует joblib для параллелизации, но отчитывается
    после завершения каждой точки.
    """
    phi_1D = stage1_result["phi_1D"]
    Z_1D = stage1_result["Z_1D"]
    Phi_mesh = stage1_result["Phi_mesh"]
    Z_mesh = stage1_result["Z_mesh"]
    d_phi = stage1_result["d_phi"]
    d_Z = stage1_result["d_Z"]
    phi_c_flat = stage1_result["phi_c_flat"]
    Z_c_flat = stage1_result["Z_c_flat"]

    epsilon_values = np.linspace(0.05, 0.8, 15)
    n_eps = len(epsilon_values)

    if progress_callback:
        progress_callback("stage2_progress", 0)
        progress_callback("stage2_text", "Перебор по ε...")
        progress_callback("log", f"Перебор по ε ({n_eps} точек)...")

    # Последовательный расчёт с отчётом прогресса по каждой точке
    F_nd_list, mu_nd_list, Q_nd_list = [], [], []
    F_dep_list, mu_dep_list, Q_dep_list = [], [], []

    for k, eps in enumerate(epsilon_values):
        res = _compute_for_epsilon(
            eps, params, phi_1D, Z_1D, Phi_mesh, Z_mesh,
            d_phi, d_Z, phi_c_flat, Z_c_flat)
        _, f_nd, m_nd, q_nd, f_dep, m_dep, q_dep = res
        F_nd_list.append(f_nd)
        mu_nd_list.append(m_nd)
        Q_nd_list.append(q_nd)
        F_dep_list.append(f_dep)
        mu_dep_list.append(m_dep)
        Q_dep_list.append(q_dep)

        if progress_callback:
            pct = int(100 * (k + 1) / n_eps)
            progress_callback("stage2_progress", pct)
            progress_callback("stage2_text", f"ε = {eps:.3f}  ({k+1}/{n_eps})")

    if progress_callback:
        progress_callback("stage2_text", "Готово")
        progress_callback("log", "Расчёт завершён.")

    # Объединяем с результатами stage1
    result = dict(stage1_result)
    result.update({
        "epsilon_values": epsilon_values,
        "F_nd": np.array(F_nd_list),
        "F_dep": np.array(F_dep_list),
        "mu_nd": np.array(mu_nd_list),
        "mu_dep": np.array(mu_dep_list),
        "Q_nd": np.array(Q_nd_list),
        "Q_dep": np.array(Q_dep_list),
    })
    return result


# -------------------------------------------------------------------
#  Построение графиков (возвращают Figure)
# -------------------------------------------------------------------

def plot_pressure_2d_section(result, dep_name="с углублениями"):
    """2D-график давления P(φ) при Z = 0 — сечение по середине."""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    phi_1D = result["phi_1D"]
    Z_1D = result["Z_1D"]
    Z_idx = np.argmin(np.abs(Z_1D - 0.0))

    ax.plot(phi_1D, result["P_nd_3d"][Z_idx, :],
            label="Без углублений", color="blue", linewidth=1.5)
    ax.plot(phi_1D, result["P_dep_3d"][Z_idx, :],
            label=dep_name, color="red", linewidth=1.5)
    ax.set_xlabel("φ, рад")
    ax.set_ylabel("P")
    ax.set_title("Поле давления P(φ) при Z = 0")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_3d_fields(result, dep_name="с углублениями"):
    """2×2: зазор H и давление P, без и с углублениями."""
    fig = Figure(figsize=(16, 14))
    Phi = result["Phi_mesh"]
    Z = result["Z_mesh"]

    cases = [
        ("Без углублений", result["H_nd_3d"], result["P_nd_3d"]),
        (dep_name,         result["H_dep_3d"], result["P_dep_3d"]),
    ]

    for i, (title, H_case, P_case) in enumerate(cases):
        idx_H = 2 * i + 1
        idx_P = 2 * i + 2

        ax_H = fig.add_subplot(2, 2, idx_H, projection="3d")
        surf_H = ax_H.plot_surface(Phi, Z, H_case, cmap="viridis",
                                    rcount=100, ccount=100)
        fig.colorbar(surf_H, ax=ax_H, shrink=0.5, aspect=10)
        ax_H.set_xlabel("φ, рад", fontsize=10)
        ax_H.set_ylabel("Z", fontsize=10)
        ax_H.set_zlabel("H", fontsize=10)
        ax_H.set_title(f"Зазор H — {title}", fontsize=11)

        ax_P = fig.add_subplot(2, 2, idx_P, projection="3d")
        surf_P = ax_P.plot_surface(Phi, Z, P_case, cmap="plasma",
                                    rcount=100, ccount=100)
        fig.colorbar(surf_P, ax=ax_P, shrink=0.5, aspect=10)
        ax_P.set_xlabel("φ, рад", fontsize=10)
        ax_P.set_ylabel("Z", fontsize=10)
        ax_P.set_zlabel("P", fontsize=10)
        ax_P.set_title(f"Давление P — {title}", fontsize=11)

    fig.tight_layout()
    return fig


def plot_F_vs_epsilon(result, dep_name="с углублениями"):
    """F(ε) — нагрузочная способность."""
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["F_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["F_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("F, Н")
    ax.set_title("Нагрузочная способность F(ε)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_mu_vs_epsilon(result, dep_name="с углублениями"):
    """μ(ε) — коэффициент трения."""
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["mu_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["mu_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("μ")
    ax.set_title("Коэффициент трения μ(ε)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_Q_vs_epsilon(result, dep_name="с углублениями"):
    """Q(ε) — расход смазки (л/с)."""
    fig = Figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["Q_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["Q_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("Q, л/с")
    ax.set_title("Расход смазки Q(ε)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def save_results(result, params, folder):
    """Сохраняет графики (PNG) и числовые данные (CSV) в папку."""
    import os
    os.makedirs(folder, exist_ok=True)
    dep_name = params["depression_name"]

    for name, plot_fn in [("pressure_2d_Z0", plot_pressure_2d_section),
                           ("fields_3d", plot_3d_fields),
                           ("F_vs_eps", plot_F_vs_epsilon),
                           ("mu_vs_eps", plot_mu_vs_epsilon),
                           ("Q_vs_eps", plot_Q_vs_epsilon)]:
        fig = plot_fn(result, dep_name)
        fig.savefig(os.path.join(folder, f"{name}.png"), dpi=150)

    eps = result["epsilon_values"]
    header = "epsilon,F_no_dep,F_dep,mu_no_dep,mu_dep,Q_no_dep,Q_dep"
    data = np.column_stack([eps,
                            result["F_nd"], result["F_dep"],
                            result["mu_nd"], result["mu_dep"],
                            result["Q_nd"], result["Q_dep"]])
    np.savetxt(os.path.join(folder, "results.csv"), data,
               delimiter=",", header=header, comments="")
