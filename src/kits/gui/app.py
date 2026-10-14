"""GUI entry point: ``kits-gui [project.kicad_pro]``."""

from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication

from .. import __version__
from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, show the main window and run the Qt event loop.

    Args:
        argv: Command-line arguments excluding the program name; ``None`` uses ``sys.argv``.

    Returns:
        The Qt application's exit code.
    """
    parser = argparse.ArgumentParser(prog="kits-gui", description="KITS board viewer")
    parser.add_argument("project", nargs="?", help=".kicad_pro (or .kicad_pcb) file to open")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("KITS")
    app.setOrganizationName("KITS")
    app.setApplicationVersion(__version__)

    window = MainWindow()
    window.show()
    if args.project:
        window.open_path(args.project)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
