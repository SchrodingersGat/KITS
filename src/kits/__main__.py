"""``python -m kits [project.kicad_pro]`` launches the GUI."""

import sys

from .gui.app import main

sys.exit(main())
