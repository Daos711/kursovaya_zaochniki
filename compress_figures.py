"""Сжатие PNG-файлов в папке figures/ для загрузки в Overleaf."""

from pathlib import Path
from PIL import Image

TARGET_WIDTH = 1200
FIGURES_DIR = Path(__file__).parent / "figures"


def compress(path: Path) -> None:
    img = Image.open(path)
    if img.width > TARGET_WIDTH:
        new_height = int(TARGET_WIDTH * img.height / img.width)
        img = img.resize((TARGET_WIDTH, new_height), Image.LANCZOS)
    img.save(path, optimize=True)
    print(f"  {path.name}: {img.width}x{img.height}")


def main() -> None:
    files = sorted(FIGURES_DIR.glob("*.png"))
    if not files:
        print("Нет PNG-файлов в figures/")
        return
    print(f"Сжимаю {len(files)} файлов...")
    for f in files:
        compress(f)
    print("Готово.")


if __name__ == "__main__":
    main()
