"""
Постобработка: вычисление F, μ, Q и построение графиков.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # безоконный бэкенд по умолчанию; GUI переключит на TkAgg
import matplotlib.pyplot as plt
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
    Q_nd = U * c * R * _trapz(_trapz(q_nd, phi_1D, axis=1), Z_1D) * 1000  # л/с

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
#  Полный расчёт: 3D-поля + перебор по ε
# -------------------------------------------------------------------

def run_full_calculation(params, epsilon_3d=0.6, num_phi=500, num_Z=500,
                         n_jobs=-1, progress_callback=None):
    """
    Выполняет полный расчёт для выбранного варианта.

    Returns
    -------
    result : dict с ключами:
        'Phi_mesh', 'Z_mesh', 'phi_1D', 'Z_1D',
        'H_nd_3d', 'H_dep_3d', 'P_nd_3d', 'P_dep_3d',  (для epsilon_3d)
        'epsilon_values', 'F_nd', 'F_dep', 'mu_nd', 'mu_dep',
        'Q_nd', 'Q_dep',
        'F_nd_3d', 'mu_nd_3d', 'Q_nd_3d',
        'F_dep_3d', 'mu_dep_3d', 'Q_dep_3d'
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

    # ---- 3D-поля при epsilon_3d ----
    if progress_callback:
        progress_callback("Расчёт 3D полей...")

    H0_3d = base_clearance(epsilon_3d, Phi_mesh)
    H_nd_3d = H0_3d.copy()
    H_dep_3d = create_H_with_depressions(H0_3d, params, Phi_mesh, Z_mesh,
                                          phi_c_flat, Z_c_flat)

    P_nd_3d, _, iter_nd = solve_reynolds_gauss_seidel_numba(
        H_nd_3d, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)
    if progress_callback:
        progress_callback(f"Без углублений: {iter_nd} итераций")

    P_dep_3d, _, iter_dep = solve_reynolds_gauss_seidel_numba(
        H_dep_3d, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)
    if progress_callback:
        progress_callback(f"С углублениями: {iter_dep} итераций")

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

    # ---- Перебор по ε ----
    epsilon_values = np.linspace(0.05, 0.8, 15)

    if progress_callback:
        progress_callback("Перебор по ε (15 точек)...")

    results_eps = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_compute_for_epsilon)(
            eps, params, phi_1D, Z_1D, Phi_mesh, Z_mesh,
            d_phi, d_Z, phi_c_flat, Z_c_flat)
        for eps in epsilon_values
    )

    # Распаковка
    F_nd_list, mu_nd_list, Q_nd_list = [], [], []
    F_dep_list, mu_dep_list, Q_dep_list = [], [], []
    for res in results_eps:
        _, f_nd, m_nd, q_nd, f_dep, m_dep, q_dep = res
        F_nd_list.append(f_nd)
        mu_nd_list.append(m_nd)
        Q_nd_list.append(q_nd)
        F_dep_list.append(f_dep)
        mu_dep_list.append(m_dep)
        Q_dep_list.append(q_dep)

    if progress_callback:
        progress_callback("Расчёт завершён.")

    return {
        "Phi_mesh": Phi_mesh,
        "Z_mesh": Z_mesh,
        "phi_1D": phi_1D,
        "Z_1D": Z_1D,
        # 3D поля
        "H_nd_3d": H_nd_3d,
        "H_dep_3d": H_dep_3d,
        "P_nd_3d": P_nd_3d,
        "P_dep_3d": P_dep_3d,
        # Числовые результаты при eps_3d
        "F_nd_3d": F_nd_3d,
        "mu_nd_3d": mu_nd_3d,
        "Q_nd_3d": Q_nd_3d,
        "F_dep_3d": F_dep_3d,
        "mu_dep_3d": mu_dep_3d,
        "Q_dep_3d": Q_dep_3d,
        # Зависимости от ε
        "epsilon_values": epsilon_values,
        "F_nd": np.array(F_nd_list),
        "F_dep": np.array(F_dep_list),
        "mu_nd": np.array(mu_nd_list),
        "mu_dep": np.array(mu_dep_list),
        "Q_nd": np.array(Q_nd_list),
        "Q_dep": np.array(Q_dep_list),
    }


# -------------------------------------------------------------------
#  Построение графиков (возвращают Figure)
# -------------------------------------------------------------------

def plot_pressure_3d(result, dep_name="с углублениями"):
    """3D-поверхность поля давления: без и с углублениями."""
    fig = Figure(figsize=(14, 6))
    Phi = result["Phi_mesh"]
    Z = result["Z_mesh"]

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(Phi, Z, result["P_nd_3d"], cmap="plasma",
                     rcount=100, ccount=100)
    ax1.set_xlabel("φ, рад")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("P")
    ax1.set_title("Без углублений")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Phi, Z, result["P_dep_3d"], cmap="plasma",
                     rcount=100, ccount=100)
    ax2.set_xlabel("φ, рад")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("P")
    ax2.set_title(dep_name)

    fig.tight_layout()
    return fig


def plot_clearance_3d(result, dep_name="с углублениями"):
    """3D-поверхность зазора H: без и с углублениями."""
    fig = Figure(figsize=(14, 6))
    Phi = result["Phi_mesh"]
    Z = result["Z_mesh"]

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(Phi, Z, result["H_nd_3d"], cmap="viridis",
                     rcount=100, ccount=100)
    ax1.set_xlabel("φ, рад")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("H")
    ax1.set_title("Без углублений")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(Phi, Z, result["H_dep_3d"], cmap="viridis",
                     rcount=100, ccount=100)
    ax2.set_xlabel("φ, рад")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("H")
    ax2.set_title(dep_name)

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

    # Графики
    for name, plot_fn in [("pressure_3d", plot_pressure_3d),
                           ("clearance_3d", plot_clearance_3d),
                           ("F_vs_eps", plot_F_vs_epsilon),
                           ("mu_vs_eps", plot_mu_vs_epsilon),
                           ("Q_vs_eps", plot_Q_vs_epsilon)]:
        fig = plot_fn(result, dep_name)
        fig.savefig(os.path.join(folder, f"{name}.png"), dpi=150)

    # CSV
    eps = result["epsilon_values"]
    header = "epsilon,F_no_dep,F_dep,mu_no_dep,mu_dep,Q_no_dep,Q_dep"
    data = np.column_stack([eps,
                            result["F_nd"], result["F_dep"],
                            result["mu_nd"], result["mu_dep"],
                            result["Q_nd"], result["Q_dep"]])
    np.savetxt(os.path.join(folder, "results.csv"), data,
               delimiter=",", header=header, comments="")
