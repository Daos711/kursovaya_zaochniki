"""
GUI на PyQt5 для решателя подшипника скольжения.
Двухэтапный расчёт: 3D-графики появляются сразу после решения Рейнольдса,
графики зависимостей — после перебора по ε.
"""

import sys
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, QPushButton,
    QTabWidget, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
    QSplitter, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QFont

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .variants import get_variant, GEOMETRIES, DEPRESSION_TYPES
from .postprocess import (
    run_stage1_3d, run_stage2_epsilon_sweep,
    plot_pressure_2d_section, plot_3d_fields,
    plot_F_vs_epsilon, plot_mu_vs_epsilon,
    plot_Q_vs_epsilon, save_results,
)


# -------------------------------------------------------------------
#  Сигналы из рабочего потока в GUI
# -------------------------------------------------------------------

class WorkerSignals(QObject):
    log = pyqtSignal(str)
    stage1_progress = pyqtSignal(int)
    stage1_text = pyqtSignal(str)
    stage2_progress = pyqtSignal(int)
    stage2_text = pyqtSignal(str)
    stage1_done = pyqtSignal(dict)
    stage2_done = pyqtSignal(dict)
    error = pyqtSignal(str)


# -------------------------------------------------------------------
#  Главное окно
# -------------------------------------------------------------------

class BearingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Подшипник скольжения — решатель с микрорельефом")
        self.resize(1400, 850)

        self._stage1_result = None
        self._full_result = None
        self._params = None
        self._calculating = False
        self._signals = WorkerSignals()

        self._connect_signals()
        self._build_ui()
        self._on_variant_changed()

    # -----------------------------------------------------------------
    #  Сигналы
    # -----------------------------------------------------------------
    def _connect_signals(self):
        self._signals.log.connect(self._append_log)
        self._signals.stage1_progress.connect(self._bar_stage1.setValue
                                               if hasattr(self, '_bar_stage1')
                                               else lambda v: None)
        self._signals.stage2_progress.connect(self._bar_stage2.setValue
                                               if hasattr(self, '_bar_stage2')
                                               else lambda v: None)
        self._signals.stage1_done.connect(self._on_stage1_done)
        self._signals.stage2_done.connect(self._on_stage2_done)
        self._signals.error.connect(self._on_error)

    def _reconnect_signals(self):
        """Повторное подключение после создания виджетов."""
        try:
            self._signals.stage1_progress.disconnect()
        except TypeError:
            pass
        try:
            self._signals.stage2_progress.disconnect()
        except TypeError:
            pass
        self._signals.stage1_progress.connect(self._bar_stage1.setValue)
        self._signals.stage1_text.connect(self._set_stage1_text)
        self._signals.stage2_progress.connect(self._bar_stage2.setValue)
        self._signals.stage2_text.connect(self._set_stage2_text)

    def _set_stage1_text(self, text):
        self._bar_stage1.setFormat(f"{text}  %p%")

    def _set_stage2_text(self, text):
        self._bar_stage2.setFormat(f"{text}  %p%")

    # -----------------------------------------------------------------
    #  UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ===== Левая панель =====
        left = QWidget()
        left.setFixedWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # --- Параметры ---
        grp_params = QGroupBox("Параметры расчёта")
        gl = QVBoxLayout(grp_params)

        gl.addWidget(QLabel("Вариант (1–30):"))
        self._spin_var = QSpinBox()
        self._spin_var.setRange(1, 30)
        self._spin_var.setValue(1)
        self._spin_var.valueChanged.connect(self._on_variant_changed)
        gl.addWidget(self._spin_var)

        self._lbl_info = QTextEdit()
        self._lbl_info.setReadOnly(True)
        self._lbl_info.setMaximumHeight(170)
        self._lbl_info.setStyleSheet("background: #f5f5f5; border: 1px solid #ccc;")
        gl.addWidget(self._lbl_info)

        gl.addWidget(QLabel("ε для 3D-графиков:"))
        self._spin_eps = QDoubleSpinBox()
        self._spin_eps.setRange(0.01, 0.99)
        self._spin_eps.setSingleStep(0.05)
        self._spin_eps.setValue(0.6)
        self._spin_eps.setDecimals(2)
        gl.addWidget(self._spin_eps)

        gl.addWidget(QLabel("Размер сетки N:"))
        self._spin_grid = QSpinBox()
        self._spin_grid.setRange(50, 1000)
        self._spin_grid.setSingleStep(50)
        self._spin_grid.setValue(500)
        gl.addWidget(self._spin_grid)

        left_layout.addWidget(grp_params)

        # --- Кнопки ---
        self._btn_calc = QPushButton("Рассчитать")
        self._btn_calc.setMinimumHeight(40)
        self._btn_calc.setStyleSheet(
            "QPushButton { background: #2196F3; color: white; font-weight: bold; "
            "border-radius: 4px; } "
            "QPushButton:hover { background: #1976D2; } "
            "QPushButton:disabled { background: #bbb; }"
        )
        self._btn_calc.clicked.connect(self._on_calculate)
        left_layout.addWidget(self._btn_calc)

        self._btn_save = QPushButton("Сохранить результаты")
        self._btn_save.setMinimumHeight(34)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        left_layout.addWidget(self._btn_save)

        # --- Прогресс ---
        grp_progress = QGroupBox("Прогресс")
        pl = QVBoxLayout(grp_progress)

        pl.addWidget(QLabel("Этап 1: Решение Рейнольдса"))
        self._bar_stage1 = QProgressBar()
        self._bar_stage1.setTextVisible(True)
        self._bar_stage1.setFormat("%p%")
        pl.addWidget(self._bar_stage1)

        pl.addWidget(QLabel("Этап 2: Перебор по ε"))
        self._bar_stage2 = QProgressBar()
        self._bar_stage2.setTextVisible(True)
        self._bar_stage2.setFormat("%p%")
        pl.addWidget(self._bar_stage2)

        left_layout.addWidget(grp_progress)

        # --- Лог ---
        grp_log = QGroupBox("Лог")
        ll = QVBoxLayout(grp_log)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet("background: #1e1e1e; color: #d4d4d4;")
        ll.addWidget(self._log)
        left_layout.addWidget(grp_log, stretch=1)

        main_layout.addWidget(left)

        # ===== Правая панель (вкладки) =====
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            "QTabBar::tab { min-width: 120px; padding: 6px 12px; }"
        )

        self._tab_frames = {}
        for name in ["P(φ) при Z=0", "3D поля", "F(ε)", "μ(ε)", "Q(ε)", "Результаты"]:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(2, 2, 2, 2)
            self._tabs.addTab(page, name)
            self._tab_frames[name] = page_layout

        # Текстовое поле на вкладке "Результаты"
        self._result_text = QTextEdit()
        self._result_text.setReadOnly(True)
        self._result_text.setFont(QFont("Consolas", 10))
        self._tab_frames["Результаты"].addWidget(self._result_text)

        main_layout.addWidget(self._tabs, stretch=1)

        # Переподключаем сигналы к реальным виджетам
        self._reconnect_signals()

    # -----------------------------------------------------------------
    #  Инфо о варианте
    # -----------------------------------------------------------------
    def _on_variant_changed(self):
        try:
            v = self._spin_var.value()
            p = get_variant(v)
        except Exception:
            return
        dep = DEPRESSION_TYPES[p["depression_type"]]
        lines = [
            f"<b>Вариант {v}</b>",
            f"Геометрия: <b>{p['geometry_key']}</b>",
            f"  R = {p['R']} м &nbsp; c = {p['c']} м &nbsp; L = {p['L']} м",
            "",
            f"Тип {p['depression_type']}: <b>{p['depression_name']}</b>",
        ]
        if "r0" in dep:
            lines.append(f"  r0 = {dep['r0']} м")
        else:
            lines.append(f"  a = {dep.get('a', '—')} м &nbsp; b = {dep.get('b', '—')} м")
        lines.append(f"  h_p = {dep['h_p']*1e6:.0f} мкм")
        self._lbl_info.setHtml("<br>".join(lines))

    # -----------------------------------------------------------------
    #  Лог
    # -----------------------------------------------------------------
    def _append_log(self, msg):
        self._log.append(msg)

    # -----------------------------------------------------------------
    #  Расчёт
    # -----------------------------------------------------------------
    def _on_calculate(self):
        if self._calculating:
            return

        v = self._spin_var.value()
        eps = self._spin_eps.value()
        grid = self._spin_grid.value()

        self._calculating = True
        self._btn_calc.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._log.clear()
        self._bar_stage1.setValue(0)
        self._bar_stage1.setFormat("Ожидание...  %p%")
        self._bar_stage2.setValue(0)
        self._bar_stage2.setFormat("Ожидание...  %p%")
        self._stage1_result = None
        self._full_result = None

        params = get_variant(v)
        self._params = params

        def progress_cb(event, value):
            if event == "log":
                self._signals.log.emit(str(value))
            elif event == "stage1_progress":
                self._signals.stage1_progress.emit(int(value))
            elif event == "stage1_text":
                self._signals.stage1_text.emit(str(value))
            elif event == "stage2_progress":
                self._signals.stage2_progress.emit(int(value))
            elif event == "stage2_text":
                self._signals.stage2_text.emit(str(value))

        def worker():
            try:
                self._signals.log.emit(
                    f"Вариант {v}: {params['depression_name']}")
                self._signals.log.emit(
                    f"Сетка {grid}×{grid}, ε_3D = {eps}")

                # ЭТАП 1
                s1 = run_stage1_3d(params, epsilon_3d=eps,
                                   num_phi=grid, num_Z=grid,
                                   progress_callback=progress_cb)
                self._signals.stage1_done.emit(s1)

                # ЭТАП 2
                full = run_stage2_epsilon_sweep(
                    params, s1, n_jobs=-1,
                    progress_callback=progress_cb)
                self._signals.stage2_done.emit(full)

            except Exception as exc:
                self._signals.error.emit(str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # -----------------------------------------------------------------
    #  Обработка результатов
    # -----------------------------------------------------------------
    def _embed_figure(self, tab_name, fig):
        """Вставляет matplotlib Figure во вкладку, заменяя старый canvas."""
        layout = self._tab_frames[tab_name]
        # Очищаем
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        canvas.draw()

    def _on_stage1_done(self, s1):
        """Вызывается когда 3D-поля готовы — сразу показываем графики."""
        self._stage1_result = s1
        dep_name = self._params["depression_name"]

        self._embed_figure("P(φ) при Z=0",
                           plot_pressure_2d_section(s1, dep_name))
        self._embed_figure("3D поля", plot_3d_fields(s1, dep_name))
        self._tabs.setCurrentIndex(0)  # Переключаем на 2D-давление

        # Числовые результаты (частичные)
        eps_3d = self._spin_eps.value()
        text = (
            f"=== Результаты при ε = {eps_3d} ===\n\n"
            f"Без углублений:\n"
            f"  F  = {s1['F_nd_3d']:.2f} Н\n"
            f"  μ  = {s1['mu_nd_3d']:.6f}\n"
            f"  Q  = {s1['Q_nd_3d']:.6f} л/с\n\n"
            f"{dep_name}:\n"
            f"  F  = {s1['F_dep_3d']:.2f} Н\n"
            f"  μ  = {s1['mu_dep_3d']:.6f}\n"
            f"  Q  = {s1['Q_dep_3d']:.6f} л/с\n\n"
            f"Перебор по ε — ожидание..."
        )
        self._result_text.setPlainText(text)

        self._append_log("3D-графики построены. Запуск перебора по ε...")

    def _on_stage2_done(self, full):
        """Вызывается когда перебор по ε завершён."""
        self._full_result = full
        dep_name = self._params["depression_name"]

        self._embed_figure("F(ε)", plot_F_vs_epsilon(full, dep_name))
        self._embed_figure("μ(ε)", plot_mu_vs_epsilon(full, dep_name))
        self._embed_figure("Q(ε)", plot_Q_vs_epsilon(full, dep_name))

        # Полные числовые результаты
        eps_3d = self._spin_eps.value()
        lines = [
            f"=== Результаты при ε = {eps_3d} ===",
            "",
            "Без углублений:",
            f"  F  = {full['F_nd_3d']:.2f} Н",
            f"  μ  = {full['mu_nd_3d']:.6f}",
            f"  Q  = {full['Q_nd_3d']:.6f} л/с",
            "",
            f"{dep_name}:",
            f"  F  = {full['F_dep_3d']:.2f} Н",
            f"  μ  = {full['mu_dep_3d']:.6f}",
            f"  Q  = {full['Q_dep_3d']:.6f} л/с",
            "",
            "=" * 50,
            "",
            "Зависимость от ε:",
            f"{'ε':>6}  {'F_nd':>10}  {'F_dep':>10}  "
            f"{'μ_nd':>10}  {'μ_dep':>10}  {'Q_nd':>10}  {'Q_dep':>10}",
        ]
        for i, eps in enumerate(full["epsilon_values"]):
            lines.append(
                f"{eps:6.3f}  {full['F_nd'][i]:10.2f}  {full['F_dep'][i]:10.2f}  "
                f"{full['mu_nd'][i]:10.6f}  {full['mu_dep'][i]:10.6f}  "
                f"{full['Q_nd'][i]:10.6f}  {full['Q_dep'][i]:10.6f}"
            )
        self._result_text.setPlainText("\n".join(lines))

        self._calculating = False
        self._btn_calc.setEnabled(True)
        self._btn_save.setEnabled(True)
        self._append_log("Все графики готовы.")

    def _on_error(self, msg):
        self._calculating = False
        self._btn_calc.setEnabled(True)
        QMessageBox.critical(self, "Ошибка расчёта", msg)

    # -----------------------------------------------------------------
    #  Сохранение
    # -----------------------------------------------------------------
    def _on_save(self):
        if self._full_result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if not folder:
            return
        try:
            save_results(self._full_result, self._params, folder)
            self._append_log(f"Сохранено в {folder}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", str(e))


def run_app():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = BearingApp()
    window.show()
    sys.exit(app.exec_())
