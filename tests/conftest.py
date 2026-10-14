import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_qsettings(tmp_path):
    """Keep tests from reading or writing the user's real KITS preferences."""
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        yield
        return
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "qsettings"))
    yield


# Demo projects shipped with KiCad make good real-world ingest fixtures.
_DEMO_ROOTS = [
    Path(os.environ.get("KICAD_DEMOS", "")),
    *sorted(Path("C:/Program Files/KiCad").glob("*/share/kicad/demos"), reverse=True),
    Path("/usr/share/kicad/demos"),
    Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/demos"),
]


def kicad_demo_projects() -> list[Path]:
    for root in _DEMO_ROOTS:
        if str(root) not in ("", ".") and root.is_dir():
            return sorted(root.glob("*/*.kicad_pro"))
    return []


BOARD_V9 = """\
(kicad_pcb (version 20241229) (generator "pcbnew")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (4 "In1.Cu" power "GND plane")
          (25 "Edge.Cuts" user) (5 "F.SilkS" user "F.Silkscreen"))
  (setup (stackup
    (layer "F.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 1" (type "prepreg") (thickness 0.2) (material "FR4") (epsilon_r 4.5))
    (layer "In1.Cu" (type "copper") (thickness 0.035))
    (layer "dielectric 2" (type "core") (thickness 1.2) (material "FR4") (epsilon_r 4.5))
    (layer "B.Cu" (type "copper") (thickness 0.035))))
  (net 0 "")
  (net 1 "GND")
  (net 2 "+5V")
  (footprint "R:R_0603" (layer "F.Cu") (at 10 20 90)
    (property "Reference" "R1" (at 0 0 0) (layer "F.SilkS"))
    (property "Value" "10k" (at 0 0 0) (layer "F.Fab"))
    (fp_line (start -1 -1) (end 1 -1) (stroke (width 0.1) (type solid)) (layer "F.SilkS"))
    (pad "1" smd roundrect (at -1 0 90) (size 0.8 0.9) (layers "F.Cu" "F.Paste" "F.Mask")
      (roundrect_rratio 0.25) (net 1 "GND"))
    (pad "2" thru_hole circle (at 1 0 90) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask")
      (net 2 "+5V")))
  (gr_rect (start 0 0) (end 40 30) (stroke (width 0.05) (type default)) (fill none) (layer "Edge.Cuts"))
  (gr_poly (pts (xy 1 1) (xy 2 1) (xy 2 2)) (stroke (width 0)) (fill solid) (layer "B.Cu") (net 1))
  (gr_text "ignored" (at 5 5 0) (layer "F.Cu"))
  (segment (start 1 2) (end 3 4) (width 0.25) (layer "F.Cu") (net 2))
  (arc (start 0 0) (mid 1 1) (end 2 0) (width 0.3) (layer "B.Cu") (net 1))
  (via (at 5 5) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 1))
  (via blind (at 6 6) (size 0.4) (drill 0.2) (layers "F.Cu" "In1.Cu") (net 2))
  (zone (net 1) (net_name "GND") (layers "F.Cu" "B.Cu") (name "gnd pour")
    (polygon (pts (xy 0 0) (xy 10 0) (xy 10 10) (xy 0 10)))
    (filled_polygon (layer "F.Cu") (pts (xy 1 1) (xy 9 1) (xy 9 9)))
    (filled_polygon (layer "B.Cu") (pts (xy 1 1) (xy 9 1) (xy 9 9))))
  (zone (net 0) (net_name "") (layer "F.Cu") (keepout (tracks not_allowed))
    (polygon (pts (xy 20 20) (xy 21 20) (xy 21 21))))
  (some_future_token (with "nested" (content 1)))
)
"""

# KiCad 10 writes net names directly and drops the net table.
BOARD_V10 = """\
(kicad_pcb (version 20260206) (generator "pcbnew")
  (layers (0 "F.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (footprint "C:C" (layer "B.Cu") (at 5 5)
    (property "Reference" "C1")
    (pad "1" smd rect (at 0 0) (size 1 1) (layers "B.Cu") (net "VCC")))
  (segment (start 0 0) (end 1 0) (width 0.2) (layer "F.Cu") (net "/VPP{slash}MCLR"))
  (zone (net "GND") (layer "B.Cu") (polygon (pts (xy 0 0) (xy 1 0) (xy 1 1))))
)
"""


@pytest.fixture
def board_v9_path(tmp_path) -> Path:
    path = tmp_path / "demo.kicad_pcb"
    path.write_text(BOARD_V9, encoding="utf-8")
    (tmp_path / "demo.kicad_pro").write_text('{"meta": {"version": 3}}', encoding="utf-8")
    return path


@pytest.fixture
def board_v10_path(tmp_path) -> Path:
    path = tmp_path / "v10.kicad_pcb"
    path.write_text(BOARD_V10, encoding="utf-8")
    return path
