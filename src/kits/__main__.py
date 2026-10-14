"""``python -m kits [project.kits]`` launches the GUI."""

import sys

from .gui.app import main

if __name__ == "__main__":
    sys.exit(main())
