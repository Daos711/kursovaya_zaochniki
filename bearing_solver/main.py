"""
Точка входа — запуск GUI-приложения.
"""

from bearing_solver.gui import BearingApp


def main():
    app = BearingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
