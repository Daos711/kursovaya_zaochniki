"""
Графический интерфейс (tkinter) для решателя подшипника скольжения.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .variants import get_variant, GEOMETRIES, DEPRESSION_TYPES
from .postprocess import (run_full_calculation,
                          plot_pressure_3d, plot_clearance_3d,
                          plot_F_vs_epsilon, plot_mu_vs_epsilon,
                          plot_Q_vs_epsilon, save_results)


class BearingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Решатель подшипника скольжения с микрорельефом")
        self.geometry("1280x800")
        self.minsize(1024, 700)

        self._result = None
        self._params = None
        self._calculating = False

        self._build_ui()

    # -----------------------------------------------------------------
    #  UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        # --- Левая панель ввода ---
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Параметры расчёта",
                  font=("", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # Вариант 1–30
        ttk.Label(left, text="Вариант (1–30):").pack(anchor=tk.W)
        self._var_num = tk.IntVar(value=1)
        vcmd = (self.register(self._validate_int), "%P")
        self._var_spin = ttk.Spinbox(left, from_=1, to=30, width=8,
                                     textvariable=self._var_num,
                                     validate="key", validatecommand=vcmd)
        self._var_spin.pack(anchor=tk.W, pady=(0, 5))
        self._var_num.trace_add("write", self._on_variant_changed)

        # Информация о варианте
        self._info_text = tk.Text(left, width=38, height=12,
                                  state=tk.DISABLED, wrap=tk.WORD)
        self._info_text.pack(anchor=tk.W, pady=(0, 10))

        # Эксцентриситет для 3D-графиков
        ttk.Label(left, text="ε для 3D-графиков:").pack(anchor=tk.W)
        self._eps_var = tk.DoubleVar(value=0.6)
        ttk.Entry(left, textvariable=self._eps_var, width=10).pack(
            anchor=tk.W, pady=(0, 5))

        # Размер сетки
        ttk.Label(left, text="Размер сетки (N):").pack(anchor=tk.W)
        self._grid_var = tk.IntVar(value=500)
        ttk.Entry(left, textvariable=self._grid_var, width=10).pack(
            anchor=tk.W, pady=(0, 10))

        # Кнопки
        self._btn_calc = ttk.Button(left, text="Рассчитать",
                                    command=self._on_calculate)
        self._btn_calc.pack(fill=tk.X, pady=(0, 5))

        self._btn_save = ttk.Button(left, text="Сохранить результаты",
                                    command=self._on_save, state=tk.DISABLED)
        self._btn_save.pack(fill=tk.X, pady=(0, 10))

        # Лог
        ttk.Label(left, text="Лог:").pack(anchor=tk.W)
        self._log = tk.Text(left, width=38, height=8, state=tk.DISABLED,
                            wrap=tk.WORD)
        self._log.pack(anchor=tk.W, fill=tk.Y, expand=True)

        # --- Правая панель результатов ---
        right = ttk.Frame(self, padding=5)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._tabs = ttk.Notebook(right)
        self._tabs.pack(fill=tk.BOTH, expand=True)

        # Вкладки с графиками
        self._tab_frames = {}
        for tab_name in ["Поле давления", "Зазор",
                         "F(ε)", "μ(ε)", "Q(ε)"]:
            frame = ttk.Frame(self._tabs)
            self._tabs.add(frame, text=tab_name)
            self._tab_frames[tab_name] = frame

        # Вкладка с числовыми результатами
        num_frame = ttk.Frame(self._tabs)
        self._tabs.add(num_frame, text="Результаты")
        self._tab_frames["Результаты"] = num_frame
        self._result_text = tk.Text(num_frame, state=tk.DISABLED, wrap=tk.WORD)
        self._result_text.pack(fill=tk.BOTH, expand=True)

        # Инициализация отображения варианта
        self._on_variant_changed()

    # -----------------------------------------------------------------
    #  Обновление инфо
    # -----------------------------------------------------------------
    def _on_variant_changed(self, *_args):
        try:
            v = self._var_num.get()
            p = get_variant(v)
        except Exception:
            return
        dep = DEPRESSION_TYPES[p["depression_type"]]
        lines = [
            f"Вариант {v}",
            f"Геометрия: {p['geometry_key']}",
            f"  R = {p['R']} м",
            f"  c = {p['c']} м",
            f"  L = {p['L']} м",
            "",
            f"Тип {p['depression_type']}: {p['depression_name']}",
        ]
        if "r0" in dep:
            lines.append(f"  r0 = {dep['r0']} м")
        else:
            lines.append(f"  a = {dep.get('a', '—')} м")
            lines.append(f"  b = {dep.get('b', '—')} м")
        lines.append(f"  h_p = {dep['h_p']*1e6:.0f} мкм")

        self._info_text.config(state=tk.NORMAL)
        self._info_text.delete("1.0", tk.END)
        self._info_text.insert(tk.END, "\n".join(lines))
        self._info_text.config(state=tk.DISABLED)

    # -----------------------------------------------------------------
    #  Расчёт
    # -----------------------------------------------------------------
    def _log_msg(self, msg):
        self._log.config(state=tk.NORMAL)
        self._log.insert(tk.END, msg + "\n")
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _on_calculate(self):
        if self._calculating:
            return
        try:
            v = self._var_num.get()
            eps = self._eps_var.get()
            grid = self._grid_var.get()
            if not (0 < eps < 1):
                raise ValueError("ε должен быть в (0, 1)")
            if not (50 <= grid <= 1000):
                raise ValueError("Размер сетки от 50 до 1000")
        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        self._calculating = True
        self._btn_calc.config(state=tk.DISABLED)
        self._btn_save.config(state=tk.DISABLED)
        self._log.config(state=tk.NORMAL)
        self._log.delete("1.0", tk.END)
        self._log.config(state=tk.DISABLED)

        params = get_variant(v)
        self._params = params

        def progress(msg):
            self.after(0, self._log_msg, msg)

        def worker():
            try:
                progress(f"Вариант {v}: {params['depression_name']}")
                progress(f"Сетка {grid}×{grid}, ε_3D = {eps}")
                result = run_full_calculation(
                    params, epsilon_3d=eps,
                    num_phi=grid, num_Z=grid,
                    progress_callback=progress)
                self._result = result
                self.after(0, self._show_results)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Ошибка расчёта",
                                                           str(exc)))
            finally:
                self.after(0, self._calc_done)

        threading.Thread(target=worker, daemon=True).start()

    def _calc_done(self):
        self._calculating = False
        self._btn_calc.config(state=tk.NORMAL)
        if self._result is not None:
            self._btn_save.config(state=tk.NORMAL)

    # -----------------------------------------------------------------
    #  Отображение результатов
    # -----------------------------------------------------------------
    def _show_results(self):
        r = self._result
        dep_name = self._params["depression_name"]

        # Очищаем старые canvas'ы
        for name, frame in self._tab_frames.items():
            if name == "Результаты":
                continue
            for w in frame.winfo_children():
                w.destroy()

        # Графики
        plots = [
            ("Поле давления", plot_pressure_3d),
            ("Зазор", plot_clearance_3d),
            ("F(ε)", plot_F_vs_epsilon),
            ("μ(ε)", plot_mu_vs_epsilon),
            ("Q(ε)", plot_Q_vs_epsilon),
        ]
        for tab_name, plot_fn in plots:
            fig = plot_fn(r, dep_name)
            frame = self._tab_frames[tab_name]
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Числовые результаты
        eps_3d = self._eps_var.get()
        lines = [
            f"=== Результаты при ε = {eps_3d} ===",
            "",
            "Без углублений:",
            f"  F  = {r['F_nd_3d']:.2f} Н",
            f"  μ  = {r['mu_nd_3d']:.6f}",
            f"  Q  = {r['Q_nd_3d']:.6f} л/с",
            "",
            f"{dep_name}:",
            f"  F  = {r['F_dep_3d']:.2f} Н",
            f"  μ  = {r['mu_dep_3d']:.6f}",
            f"  Q  = {r['Q_dep_3d']:.6f} л/с",
            "",
            "=" * 40,
            "",
            "Зависимость от ε:",
            f"{'ε':>6}  {'F_nd':>10}  {'F_dep':>10}  {'μ_nd':>10}  {'μ_dep':>10}  {'Q_nd':>10}  {'Q_dep':>10}",
        ]
        for i, eps in enumerate(r["epsilon_values"]):
            lines.append(
                f"{eps:6.3f}  {r['F_nd'][i]:10.2f}  {r['F_dep'][i]:10.2f}  "
                f"{r['mu_nd'][i]:10.6f}  {r['mu_dep'][i]:10.6f}  "
                f"{r['Q_nd'][i]:10.6f}  {r['Q_dep'][i]:10.6f}"
            )

        self._result_text.config(state=tk.NORMAL)
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert(tk.END, "\n".join(lines))
        self._result_text.config(state=tk.DISABLED)

    # -----------------------------------------------------------------
    #  Сохранение
    # -----------------------------------------------------------------
    def _on_save(self):
        if self._result is None:
            return
        folder = filedialog.askdirectory(title="Выберите папку для сохранения")
        if not folder:
            return
        try:
            save_results(self._result, self._params, folder)
            self._log_msg(f"Сохранено в {folder}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    # -----------------------------------------------------------------
    #  Валидация
    # -----------------------------------------------------------------
    @staticmethod
    def _validate_int(value):
        if value == "":
            return True
        try:
            int(value)
            return True
        except ValueError:
            return False
