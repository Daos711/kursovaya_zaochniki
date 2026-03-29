"""
Постобработка: вычисление F, μ, Q и построение графиков.
Расчёт разбит на два этапа для поэтапного отображения в GUI.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as ticker
from matplotlib.figure import Figure

# numpy >= 2.0 переименовал trapz → trapezoid
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from reynolds_solver import solve_reynolds
from .geometry import (base_clearance, create_H_with_depressions,
                       make_grid, compute_depression_centers)


def compute_dP_dphi(P, d_phi):
    """Центральные разности dP/dφ с учётом периодичности."""
    N_Z, N_phi = P.shape
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]
    return dP_dphi

# -------------------------------------------------------------------
#  ГОСТ-оформление графиков
# -------------------------------------------------------------------

def _comma_formatter(x, pos):
    """Десятичный разделитель — запятая."""
    s = f"{x:g}"
    return s.replace(".", ",")

_COMMA_FMT = ticker.FuncFormatter(_comma_formatter)

matplotlib.rcParams.update({
    'font.size': 14,
    'legend.fontsize': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})


def _apply_comma(ax, apply_z=False):
    """Применить запятую к осям."""
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))
    if apply_z and hasattr(ax, 'zaxis'):
        ax.zaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))


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
    P_nd, _, iter_nd = solve_reynolds(
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
    P_dep, _, iter_dep = solve_reynolds(
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
            F_dep, mu_dep, Q_dep,
            iter_nd, iter_dep)


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
    P_nd_3d, _, iter_nd = solve_reynolds(
        H_nd_3d, d_phi, d_Z, R, L, omega=1.5, max_iter=50000)
    if progress_callback:
        progress_callback("log", f"Без углублений: {iter_nd} итераций")
        progress_callback("stage1_text", "Рейнольдс с углублениями...")
        progress_callback("stage1_progress", 50)

    # Решаем с углублениями
    P_dep_3d, _, iter_dep = solve_reynolds(
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
        "iter_nd": iter_nd, "iter_dep": iter_dep,
    }


# -------------------------------------------------------------------
#  ЭТАП 2: Перебор по ε (графики зависимостей)
# -------------------------------------------------------------------

def run_stage2_epsilon_sweep(params, stage1_result, n_jobs=-1,
                             progress_callback=None):
    """
    Перебор по ε с поточечным отчётом прогресса.
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

    F_nd_list, mu_nd_list, Q_nd_list = [], [], []
    F_dep_list, mu_dep_list, Q_dep_list = [], [], []

    for k, eps in enumerate(epsilon_values):
        res = _compute_for_epsilon(
            eps, params, phi_1D, Z_1D, Phi_mesh, Z_mesh,
            d_phi, d_Z, phi_c_flat, Z_c_flat)
        _, f_nd, m_nd, q_nd, f_dep, m_dep, q_dep, _, _ = res
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

def plot_clearance_2d_section(result, dep_name="с углублениями"):
    """2D-график зазора H(φ) при Z = 0."""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    phi_1D = result["phi_1D"]
    Z_1D = result["Z_1D"]
    Z_idx = np.argmin(np.abs(Z_1D - 0.0))

    ax.plot(phi_1D, result["H_nd_3d"][Z_idx, :],
            label="Без углублений", color="blue", linewidth=1.5)
    ax.plot(phi_1D, result["H_dep_3d"][Z_idx, :],
            label=dep_name, color="red", linewidth=1.5)
    ax.set_xlabel("φ, рад")
    ax.set_ylabel("H")
    ax.legend()
    ax.grid(True)
    _apply_comma(ax)
    fig.tight_layout()
    return fig


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
    ax.legend()
    ax.grid(True)
    _apply_comma(ax)
    fig.tight_layout()
    return fig


def _plot_3d_field(result, key_smooth, key_dep, cmap, zlabel):
    """1×2: одно поле (H или P), без углублений и с углублениями."""
    fig = Figure(figsize=(14, 6))
    Phi = result["Phi_mesh"]
    Z = result["Z_mesh"]

    for idx, key in enumerate([key_smooth, key_dep], start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        surf = ax.plot_surface(Phi, Z, result[key], cmap=cmap,
                               rcount=200, ccount=200)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.12)
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))
        ax.set_xlabel("φ, рад", fontsize=9, labelpad=2)
        ax.set_ylabel("Z", fontsize=9, labelpad=2)
        ax.set_zlabel(zlabel, fontsize=9, labelpad=2)
        ax.tick_params(labelsize=7, pad=1)
        _apply_comma(ax, apply_z=True)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03,
                        wspace=0.15)
    return fig


def plot_3d_fields(result, dep_name="с углублениями"):
    """2×2: зазор H и давление P, без и с углублениями."""
    fig = Figure(figsize=(16, 15))
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
                                    rcount=200, ccount=200)
        cbar_H = fig.colorbar(surf_H, ax=ax_H, shrink=0.4, aspect=10, pad=0.12)
        cbar_H.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))
        ax_H.set_xlabel("φ, рад", fontsize=9, labelpad=2)
        ax_H.set_ylabel("Z", fontsize=9, labelpad=2)
        ax_H.set_zlabel("H", fontsize=9, labelpad=2)
        ax_H.tick_params(labelsize=7, pad=1)
        _apply_comma(ax_H, apply_z=True)

        ax_P = fig.add_subplot(2, 2, idx_P, projection="3d")
        surf_P = ax_P.plot_surface(Phi, Z, P_case, cmap="plasma",
                                    rcount=200, ccount=200)
        cbar_P = fig.colorbar(surf_P, ax=ax_P, shrink=0.4, aspect=10, pad=0.12)
        cbar_P.ax.yaxis.set_major_formatter(ticker.FuncFormatter(_comma_formatter))
        ax_P.set_xlabel("φ, рад", fontsize=9, labelpad=2)
        ax_P.set_ylabel("Z", fontsize=9, labelpad=2)
        ax_P.set_zlabel("P", fontsize=9, labelpad=2)
        ax_P.tick_params(labelsize=7, pad=1)
        _apply_comma(ax_P, apply_z=True)

    fig.subplots_adjust(top=0.98, bottom=0.02, hspace=0.12, wspace=0.12)
    return fig


def plot_F_vs_epsilon(result, dep_name="с углублениями"):
    """F(ε) — нагрузочная способность."""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["F_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["F_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("F, Н")
    ax.legend()
    ax.grid(True)
    _apply_comma(ax)
    fig.tight_layout()
    return fig


def plot_mu_vs_epsilon(result, dep_name="с углублениями"):
    """μ(ε) — коэффициент трения."""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["mu_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["mu_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("μ")
    ax.legend()
    ax.grid(True)
    _apply_comma(ax)
    fig.tight_layout()
    return fig


def plot_Q_vs_epsilon(result, dep_name="с углублениями"):
    """Q(ε) — расход смазки (л/с)."""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    eps = result["epsilon_values"]
    ax.plot(eps, result["Q_nd"], "o-", color="blue", label="Без углублений")
    ax.plot(eps, result["Q_dep"], "s-", color="red", label=dep_name)
    ax.set_xlabel("ε")
    ax.set_ylabel("Q, л/с")
    ax.legend()
    ax.grid(True)
    _apply_comma(ax)
    fig.tight_layout()
    return fig


def save_results(result, params, folder):
    """Сохраняет графики (PNG) и числовые данные (CSV) в папку."""
    fig_dir = os.path.join(folder, "figures")
    data_dir = os.path.join(folder, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    dep_name = params["depression_name"]

    # 2D-сечения при ε=0.6
    fig_c2d = plot_clearance_2d_section(result, dep_name)
    fig_c2d.savefig(os.path.join(fig_dir, "clearance_2d.png"), dpi=300)

    fig_p2d = plot_pressure_2d_section(result, dep_name)
    fig_p2d.savefig(os.path.join(fig_dir, "pressure_2d.png"), dpi=300)

    # 3D-поля
    fig_p3d = _plot_3d_field(result, "P_nd_3d", "P_dep_3d", "plasma", "P")
    fig_p3d.savefig(os.path.join(fig_dir, "pressure_3d.png"), dpi=300)

    fig_h3d = _plot_3d_field(result, "H_nd_3d", "H_dep_3d", "viridis", "H")
    fig_h3d.savefig(os.path.join(fig_dir, "clearance_3d.png"), dpi=300)

    # Зависимости от ε
    for name, plot_fn in [("load_vs_epsilon", plot_F_vs_epsilon),
                           ("friction_vs_epsilon", plot_mu_vs_epsilon),
                           ("flow_vs_epsilon", plot_Q_vs_epsilon)]:
        fig = plot_fn(result, dep_name)
        fig.savefig(os.path.join(fig_dir, f"{name}.png"), dpi=300)

    # results.csv
    eps = result["epsilon_values"]
    header = "epsilon,F_no_dep,F_dep,mu_no_dep,mu_dep,Q_no_dep,Q_dep"
    data = np.column_stack([eps,
                            result["F_nd"], result["F_dep"],
                            result["mu_nd"], result["mu_dep"],
                            result["Q_nd"], result["Q_dep"]])
    np.savetxt(os.path.join(data_dir, "results.csv"), data,
               delimiter=",", header=header, comments="")

    # fields.npz — все поля для перерисовки без пересчёта
    np.savez_compressed(os.path.join(data_dir, "fields.npz"),
                        Phi_mesh=result["Phi_mesh"],
                        Z_mesh=result["Z_mesh"],
                        phi_1D=result["phi_1D"],
                        Z_1D=result["Z_1D"],
                        H_nd_3d=result["H_nd_3d"],
                        H_dep_3d=result["H_dep_3d"],
                        P_nd_3d=result["P_nd_3d"],
                        P_dep_3d=result["P_dep_3d"],
                        epsilon_values=result["epsilon_values"],
                        F_nd=result["F_nd"],
                        F_dep=result["F_dep"],
                        mu_nd=result["mu_nd"],
                        mu_dep=result["mu_dep"],
                        Q_nd=result["Q_nd"],
                        Q_dep=result["Q_dep"],
                        F_nd_3d=np.array([result["F_nd_3d"]]),
                        F_dep_3d=np.array([result["F_dep_3d"]]),
                        mu_nd_3d=np.array([result["mu_nd_3d"]]),
                        mu_dep_3d=np.array([result["mu_dep_3d"]]),
                        Q_nd_3d=np.array([result["Q_nd_3d"]]),
                        Q_dep_3d=np.array([result["Q_dep_3d"]]))

    # comparison_eps06.csv
    F_s = result["F_nd_3d"]
    F_t = result["F_dep_3d"]
    mu_s = result["mu_nd_3d"]
    mu_t = result["mu_dep_3d"]
    Q_s = result["Q_nd_3d"]
    Q_t = result["Q_dep_3d"]

    dF = (F_t - F_s) / F_s * 100 if F_s != 0 else 0.0
    dmu = (mu_t - mu_s) / mu_s * 100 if mu_s != 0 else 0.0
    dQ = (Q_t - Q_s) / Q_s * 100 if Q_s != 0 else 0.0

    with open(os.path.join(data_dir, "comparison_eps06.csv"), "w") as f:
        f.write("parameter,smooth,textured,delta_percent\n")
        f.write(f"F_N,{F_s:.1f},{F_t:.1f},{dF:+.2f}\n")
        f.write(f"mu,{mu_s:.6f},{mu_t:.6f},{dmu:+.2f}\n")
        f.write(f"Q_Ls,{Q_s:.4f},{Q_t:.4f},{dQ:+.2f}\n")
