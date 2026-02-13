"""
Формирование зазора подшипника: базовый зазор, размещение углублений,
10 функций профиля микрорельефа.
"""

import numpy as np


def make_grid(num_phi=500, num_Z=500):
    """Создаёт безразмерную сетку (phi, Z) и возвращает 1D- и 2D-массивы."""
    phi_1D = np.linspace(0, 2 * np.pi, num_phi)
    Z_1D = np.linspace(-1, 1, num_Z)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z_1D[1] - Z_1D[0]
    return phi_1D, Z_1D, Phi_mesh, Z_mesh, d_phi, d_Z


def compute_depression_centers(params):
    """
    Вычисляет плоские массивы координат центров углублений.
    Возвращает (phi_c_flat, Z_c_flat).
    """
    A_nd = params["A_nd"]
    B_nd = params["B_nd"]
    N_phi = params["N_phi"]
    N_Z = params["N_Z"]
    phi_start = params["phi_start"]
    phi_end = params["phi_end"]

    # По φ
    delta_phi_gap = (phi_end - phi_start - 2 * N_phi * B_nd) / (N_phi - 1)
    delta_phi_center = 2 * B_nd + delta_phi_gap
    phi_centers = phi_start + B_nd + delta_phi_center * np.arange(N_phi)

    # По Z
    delta_Z_gap = (2 - 2 * N_Z * A_nd) / (N_Z - 1)
    delta_Z_center = 2 * A_nd + delta_Z_gap
    Z_centers = -1 + A_nd + delta_Z_center * np.arange(N_Z)

    phi_c_grid, Z_c_grid = np.meshgrid(phi_centers, Z_centers)
    return phi_c_grid.flatten(), Z_c_grid.flatten()


def base_clearance(epsilon, Phi_mesh):
    """H0(φ) = 1 + ε·cos(φ)"""
    return 1.0 + epsilon * np.cos(Phi_mesh)


# -----------------------------------------------------------------------
#  10 типов микрорельефа
# -----------------------------------------------------------------------

def _depression_ellipsoidal(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 1. Эллипсоидальный."""
    r2 = (delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2
    mask = r2 <= 1.0
    H[mask] += H_p * np.sqrt(1.0 - r2[mask])


def _depression_parabolic(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 2. Параболический."""
    r2 = (delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2
    mask = r2 <= 1.0
    H[mask] += H_p * (1.0 - r2[mask])


def _depression_cylindrical(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 3. Цилиндрический (плоское дно)."""
    r2 = (delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2
    mask = r2 <= 1.0
    H[mask] += H_p


def _depression_elliptic_cylinder_rounded(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 4. Эллиптический цилиндр со скруглением."""
    r = np.sqrt((delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2)
    mask_inner = r <= 0.7
    mask_outer = (r > 0.7) & (r <= 1.0)
    H[mask_inner] += H_p
    H[mask_outer] += H_p * np.cos(np.pi * (r[mask_outer] - 0.7) / 0.6) ** 2


def _depression_spherical_cap(H, H_p, A_nd, B_nd, delta_phi, delta_Z,
                              R_bearing, r0, c, L):
    """Тип 5. Сферическая шапка (a = b = r0).

    Вычисление ведётся в физических координатах (метры), чтобы
    избежать рассогласования масштабов (r0/R vs h_p/c).
    """
    h_p = H_p * c                            # глубина в метрах
    R_s = (r0 ** 2 + h_p ** 2) / (2 * h_p)  # радиус сферы, м

    d_phi_m = delta_phi * R_bearing          # расстояние по φ, м
    d_Z_m = delta_Z * (L / 2)               # расстояние по Z, м
    rho_sq = d_phi_m ** 2 + d_Z_m ** 2

    mask = rho_sq <= r0 ** 2
    H[mask] += (np.sqrt(R_s ** 2 - rho_sq[mask])
                - np.sqrt(R_s ** 2 - r0 ** 2)) / c


def _depression_conical(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 6. Конический."""
    r = np.sqrt((delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2)
    mask = r <= 1.0
    H[mask] += H_p * (1.0 - r[mask])


def _depression_superelliptic(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 7. Суперэллиптический (крутые стенки)."""
    r = np.sqrt((delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2)
    mask = r <= 1.0
    H[mask] += H_p * np.sqrt(1.0 - r[mask] ** 4)


def _depression_flat_bottom_chamfer(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 8. Плоскодонный с фаской (эллиптический контур)."""
    r = np.sqrt((delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2)
    mask_inner = r <= 0.6
    mask_outer = (r > 0.6) & (r <= 1.0)
    H[mask_inner] += H_p
    H[mask_outer] += H_p * (1.0 - r[mask_outer]) / 0.4


def _depression_combined(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 9. Комбинированный (двухуровневый)."""
    r = np.sqrt((delta_phi / B_nd) ** 2 + (delta_Z / A_nd) ** 2)
    mask1 = r <= 0.5
    mask2 = (r > 0.5) & (r <= 0.8)
    mask3 = (r > 0.8) & (r <= 1.0)
    H[mask1] += H_p
    H[mask2] += 0.6 * H_p
    H[mask3] += 0.6 * H_p * np.cos(np.pi * (r[mask3] - 0.8) / 0.4) ** 2


def _depression_trapezoidal(H, H_p, A_nd, B_nd, delta_phi, delta_Z):
    """Тип 10. Трапецеидальный (прямоугольный в плане, max-норма)."""
    r_inf = np.maximum(np.abs(delta_phi / B_nd), np.abs(delta_Z / A_nd))
    mask_inner = r_inf <= 0.6
    mask_outer = (r_inf > 0.6) & (r_inf <= 1.0)
    H[mask_inner] += H_p
    H[mask_outer] += H_p * (1.0 - r_inf[mask_outer]) / 0.4


# -----------------------------------------------------------------------
#  Общий интерфейс
# -----------------------------------------------------------------------

def create_H_with_depressions(H0, params, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat):
    """
    Возвращает H = H0 + ΔH для выбранного типа углубления.

    Parameters
    ----------
    H0 : ndarray  — базовый зазор (копируется внутри)
    params : dict  — словарь из get_variant()
    Phi_mesh, Z_mesh : ndarray  — 2D-сетки координат
    phi_c_flat, Z_c_flat : 1d arrays  — координаты центров углублений
    """
    H = H0.copy()
    dep_type = params["depression_type"]
    H_p = params["H_p"]
    A_nd = params["A_nd"]
    B_nd = params["B_nd"]

    for k in range(len(phi_c_flat)):
        phi_c = phi_c_flat[k]
        Z_c = Z_c_flat[k]

        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c),
                               np.cos(Phi_mesh - phi_c))
        delta_Z = Z_mesh - Z_c

        if dep_type == 1:
            _depression_ellipsoidal(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 2:
            _depression_parabolic(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 3:
            _depression_cylindrical(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 4:
            _depression_elliptic_cylinder_rounded(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 5:
            _depression_spherical_cap(H, H_p, A_nd, B_nd, delta_phi, delta_Z,
                                      params["R"], params["a"],
                                      params["c"], params["L"])
        elif dep_type == 6:
            _depression_conical(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 7:
            _depression_superelliptic(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 8:
            _depression_flat_bottom_chamfer(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 9:
            _depression_combined(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        elif dep_type == 10:
            _depression_trapezoidal(H, H_p, A_nd, B_nd, delta_phi, delta_Z)
        else:
            raise ValueError(f"Неизвестный тип углубления: {dep_type}")

    return H
