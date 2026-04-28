#!/usr/bin/env python3
"""
Расчёт 14 вариантов курсовых работ (GPU-солвер).

Запуск:  python run_all_variants.py
"""

import time
import sys
import os
import numpy as np

from bearing_solver.variants import get_variant
from bearing_solver.postprocess import (
    run_stage1_3d,
    run_stage2_epsilon_sweep,
    save_results,
)

# 14 вариантов для расчёта
VARIANTS = [1, 3, 5, 6, 8, 11, 12, 13, 15, 17, 18, 19, 20, 23]

STUDENTS = {
    1:  "Востриков Я.В.",
    3:  "Грицких Т.А.",
    5:  "Золотухин А.Г.",
    11: "Михайловский А.В.",
    12: "Мысягин А.И.",
    13: "Плотников А.М.",
    15: "Ребров Е.В.",
    17: "Рудчик К.Е.",
    18: "Семёнов Е.А.",
    19: "Строгов М.М.",
    20: "Фисин С.В.",
    23: "Шиян Е.А.",
}


def run_variant(var_num, output_root="results"):
    """Рассчитать один вариант и сохранить результаты."""
    params = get_variant(var_num)
    dep_name = params["depression_name"]
    geom = params["geometry_key"]
    folder = os.path.join(output_root, f"var_{var_num:02d}")

    print(f"\n[var_{var_num:02d}] {dep_name}, геом. {geom} "
          f"({STUDENTS.get(var_num, '')})")

    # --- Этап 1: 3D-поля при ε=0.6 ---
    print(f"[var_{var_num:02d}] 3D поля (ε=0,6)...")
    t0 = time.time()
    stage1 = run_stage1_3d(params, epsilon_3d=0.6, num_phi=500, num_Z=500)
    t1 = time.time()

    iter_nd = stage1.get("iter_nd", "?")
    iter_dep = stage1.get("iter_dep", "?")
    print(f"[var_{var_num:02d}] Рейнольдс (гладкий): {iter_nd} итер., "
          f"{t1 - t0:.1f} с")
    print(f"[var_{var_num:02d}] Рейнольдс (с углубл.): {iter_dep} итер.")

    # --- Этап 2: перебор по ε ---
    print(f"[var_{var_num:02d}] Перебор по ε (15 точек)...")
    t2 = time.time()

    def eps_progress(key, val):
        if key == "stage2_text" and "ε" in str(val):
            print(f"  {val}", end="\r", flush=True)

    result = run_stage2_epsilon_sweep(params, stage1,
                                      progress_callback=eps_progress)
    t3 = time.time()
    print(f"[var_{var_num:02d}] ε=0,05 ... ε=0,80 — готово             ")

    # --- Сохранение ---
    save_results(result, params, folder)
    total = t3 - t0
    print(f"[var_{var_num:02d}] Сохранение → {folder}/")
    print(f"[var_{var_num:02d}] Итого: {total:.1f} с")

    return {
        "var": var_num,
        "dep_name": dep_name,
        "F_nd": result["F_nd_3d"],
        "F_dep": result["F_dep_3d"],
        "time": total,
    }


def main():
    print("=" * 60)
    print("  Расчёт 14 вариантов курсовых работ (GPU-солвер)")
    print("=" * 60)

    output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results")
    os.makedirs(output_root, exist_ok=True)

    summary = []
    t_total_start = time.time()

    for var_num in VARIANTS:
        try:
            info = run_variant(var_num, output_root)
            summary.append(info)
        except Exception as e:
            print(f"[var_{var_num:02d}] ОШИБКА: {e}")
            import traceback
            traceback.print_exc()

    t_total = time.time() - t_total_start

    # --- Сводная таблица ---
    print("\n" + "=" * 75)
    print("  СВОДКА")
    print("=" * 75)
    print(f"{'Вар.':>5} | {'Тип':<30} | {'F_гл':>10} | {'F_угл':>10} "
          f"| {'ΔF%':>7} | {'Время':>7}")
    print("-" * 75)
    for s in summary:
        dF = (s["F_dep"] - s["F_nd"]) / s["F_nd"] * 100 if s["F_nd"] else 0
        print(f"{s['var']:>5} | {s['dep_name']:<30} | "
              f"{s['F_nd']:>10.1f} | {s['F_dep']:>10.1f} | "
              f"{dF:>+6.2f}% | {s['time']:>5.1f}с")
    print("-" * 75)
    print(f"Общее время: {t_total:.1f} с")
    print(f"Результаты: {output_root}/")


if __name__ == "__main__":
    main()
