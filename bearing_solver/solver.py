"""
Решатель статического уравнения Рейнольдса методом Гаусса-Зейделя
с верхней релаксацией (SOR), ускоренный через Numba.

Безразмерное уравнение:
    d/dφ(H³ · dP/dφ) + (D/L)² · d/dZ(H³ · dP/dZ) = dH/dφ

Граничные условия:
    - По φ: периодические
    - По Z: P = 0 при Z = ±1
    - Кавитация: P ≥ 0
"""

import numpy as np
from numba import njit


@njit
def solve_reynolds_gauss_seidel_numba(H, d_phi, d_Z, R, L,
                                       omega=1.5, tol=1e-5, max_iter=50000):
    """
    Решает безразмерное уравнение Рейнольдса методом SOR.

    Parameters
    ----------
    H : ndarray (N_Z, N_phi) — безразмерный зазор
    d_phi, d_Z : float — шаги сетки
    R, L : float — радиус и длина подшипника (м)
    omega : float — параметр релаксации (1 < omega < 2)
    tol : float — критерий сходимости
    max_iter : int — максимальное число итераций

    Returns
    -------
    P : ndarray — безразмерное поле давления
    delta : float — финальная невязка
    iteration : int — число выполненных итераций
    """
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    # H на полуцелых узлах
    H_i_plus_half = 0.5 * (H[:, :-1] + H[:, 1:])
    H_i_minus_half = np.hstack((H_i_plus_half[:, -1:], H_i_plus_half[:, :-1]))

    H_j_plus_half = 0.5 * (H[:-1, :] + H[1:, :])
    H_j_minus_half = np.vstack((H_j_plus_half[-1:, :], H_j_plus_half[:-1, :]))

    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A = H_i_plus_half ** 3
    B = H_i_minus_half ** 3
    C = alpha_sq * H_j_plus_half ** 3
    D_coef = alpha_sq * H_j_minus_half ** 3

    # Расширяем массивы до (N_Z, N_phi)
    A_full = np.zeros((N_Z, N_phi))
    B_full = np.zeros((N_Z, N_phi))
    C_full = np.zeros((N_Z, N_phi))
    D_full = np.zeros((N_Z, N_phi))

    A_full[:, :-1] = A
    A_full[:, -1] = A[:, 0]
    B_full[:, 1:] = B
    B_full[:, 0] = B[:, -1]

    C_full[:-1, :] = C
    C_full[-1, :] = C[0, :]
    D_full[1:, :] = D_coef
    D_full[0, :] = D_coef[-1, :]

    E = A_full + B_full + C_full + D_full

    F = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = np.zeros((N_Z, N_phi))
    F_full[:, :-1] = F
    F_full[:, -1] = F[:, 0]

    # Итерационный процесс
    delta = 1.0
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0

        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                Ai = A_full[i, j]
                Bi = B_full[i, j]
                Ci = C_full[i, j]
                Di = D_full[i, j]
                Ei = E[i, j]
                Fi = F_full[i, j]

                P_old_ij = P[i, j]

                P_new = (Ai * P[i, (j + 1) % N_phi] +
                         Bi * P[i, (j - 1) % N_phi] +
                         Ci * P[i + 1, j] +
                         Di * P[i - 1, j] - Fi) / Ei
                P_new = max(P_new, 0.0)
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)

                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])

        # Периодические по φ
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]

        # Дирихле по Z
        P[0, :] = 0.0
        P[-1, :] = 0.0

        delta /= (norm_P + 1e-8)
        iteration += 1

    return P, delta, iteration


def compute_dP_dphi(P, d_phi):
    """Центральные разности dP/dφ с учётом периодичности."""
    N_Z, N_phi = P.shape
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]
    return dP_dphi
