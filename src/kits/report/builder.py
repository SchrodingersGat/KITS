"""Build a self-contained HTML report of a project's solve results.

The report is one ``.html`` file with no external resources: summary figures,
tables of every simulation input and its results, and a map of each result
quantity on each copper layer. The maps are value grids (16-bit, zlib
compressed) coloured by the page's script, so hovering shows the value under
the cursor on every layer.
"""

from __future__ import annotations

import base64
import datetime
import html
import json
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import shapely

from .. import __version__
from ..config import HeatLoadKind
from ..ingest.model import Board, Footprint
from ..mesh.fingerprint import board_hash
from ..mesh.geometry import footprint_body
from ..mesh.mesher import layer_heights
from ..postprocess.fields import (
    CURRENT_DENSITY_PERCENTILE,
    DisplayMode,
    format_resistance,
    format_value,
    sample_layer,
    supports_log,
    triangle_values,
    value_range,
)
from ..postprocess.limits import (
    check_limits,
    current_density_basis,
    layer_peak_current_density,
    peak_current_density,
    verdict,
)
from ..project import KitsProject
from ..solve.elements import default_ambient_conductance, default_body_thermal, terminal_thermal_resistances
from ..solve.results import LayerResult, SolveResults

DEFAULT_MAX_PIXELS = 1600  # map width/height along the board's longer side
OUNCE_COPPER_MM = 0.0348  # 1 oz/ft² copper
_CODES = 65534  # quantisation levels; code 0 means "no value here"
_DATA_MARK = "/*KITS:DATA*/null"
Progress = Callable[[str, float], None]


@dataclass(frozen=True)
class Grid:
    """A pixel grid over the board, in board (KiCad) millimetres, rows from the top.

    Attributes:
        x0: Left edge.
        y0: Top edge (smallest KiCad Y).
        step: Pixel size, mm.
        width: Pixels across.
        height: Pixels down.
    """

    x0: float
    y0: float
    step: float
    width: int
    height: int

    @classmethod
    def over(cls, bounds: tuple[float, float, float, float], max_pixels: int) -> Grid:
        """A grid covering some bounds, ``max_pixels`` along the longer side.

        Args:
            bounds: ``(x0, y0, x1, y1)`` in millimetres.
            max_pixels: Pixels along the longer side.

        Returns:
            The grid.
        """
        x0, y0, x1, y1 = bounds
        step = max(x1 - x0, y1 - y0, 1e-6) / max_pixels
        return cls(
            x0, y0, step, max(1, int(np.ceil((x1 - x0) / step))), max(1, int(np.ceil((y1 - y0) / step)))
        )

    def centres(self) -> np.ndarray:
        """Pixel centres, row by row from the top.

        Returns:
            ``(height * width, 2)`` points in millimetres.
        """
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.step
        ys = self.y0 + (np.arange(self.height) + 0.5) * self.step
        gx, gy = np.meshgrid(xs, ys)
        return np.column_stack([gx.ravel(), gy.ravel()])


def encode_raster(values: np.ndarray, log: bool) -> dict:
    """Quantise a value grid to 16 bits and compress it for the page.

    Args:
        values: Values in display units, NaN where there is none.
        log: Quantise ``log10(value)`` (for quantities spanning decades); values
            at or below zero take the lowest code.

    Returns:
        ``{"q": "log"|"lin", "lo", "hi", "data"}``: the value range of the codes
        and base64 zlib-compressed little-endian codes (0 = no value).
    """
    ok = ~np.isnan(values)
    codes = np.zeros(values.shape, "<u2")
    lo = hi = 0.0
    if ok.any():
        v = values[ok]
        if log:
            positive = v[v > 0]
            floor = float(np.log10(positive.min())) if positive.size else 0.0
            v = np.log10(np.maximum(v, 10**floor)) if positive.size else np.zeros_like(v)
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo if hi > lo else 1.0
        codes[ok] = 1 + np.round((v - lo) / span * (_CODES - 1)).astype("<u2")
    data = base64.b64encode(zlib.compress(codes.tobytes(), 6)).decode("ascii")
    return {"q": "log" if log else "lin", "lo": lo, "hi": hi, "data": data}


# -- tables --------------------------------------------------------------------------


def _e(value) -> str:
    """HTML-escape any value as text.

    Args:
        value: The value.

    Returns:
        Escaped text.
    """
    return html.escape(str(value))


def _num(value: float | None, unit: str = "", digits: int = 4, signed: bool = False) -> str:
    """Format a number with a unit, or a dash for none.

    Args:
        value: The number, or None.
        unit: Unit text, e.g. ``"W"``.
        digits: Significant digits.
        signed: Show a ``+`` sign on positive values.

    Returns:
        e.g. ``"1.234 W"`` or ``"—"``.
    """
    if value is None:
        return "—"
    text = f"{value:+.{digits}g}" if signed else f"{value:.{digits}g}"
    if "e+" in text and abs(value) < 1e9:  # 1e+03 K/W reads better as 1,000 K/W
        text = f"{value:+,.0f}" if signed else f"{value:,.0f}"
    return f"{text} {unit}".strip()


def _table(
    title: str,
    note: str,
    headers: list[str],
    rows: list[list[str]],
    numeric_from: int = 1,
    numeric: set[int] | None = None,
) -> str:
    """An HTML table section (cells already escaped).

    Args:
        title: Section heading.
        note: One-line explanation under the heading (escaped here).
        headers: Column headings.
        rows: Cell HTML per row.
        numeric_from: Columns from this index are right-aligned numbers.
        numeric: The right-aligned number columns instead, if given.

    Returns:
        The section's HTML, or an empty string if there are no rows.
    """
    if not rows:
        return ""
    numbers = numeric if numeric is not None else set(range(numeric_from, len(headers)))
    head = "".join(f'<th class="{"num" if i in numbers else ""}">{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{"num" if i in numbers else ""}">{c}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return (
        f'<section class="block"><h3>{_e(title)}</h3><p class="note">{_e(note)}</p>'
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
        "</section>"
    )


def _footprints(board: Board) -> dict[str, Footprint]:
    """Footprints by reference.

    Args:
        board: The board.

    Returns:
        ``{reference: footprint}``.
    """
    return {fp.reference: fp for fp in board.footprints}


def _element_rows(project: KitsProject, board: Board, results: SolveResults) -> list[list[str]]:
    """One row per resistive element: its definition and its results.

    Args:
        project: The project.
        board: Its board.
        results: The results.

    Returns:
        Table rows.
    """
    footprints = _footprints(board)
    rows = []
    for e in project.active_resistive_elements:
        a, b = e.groups
        fp = footprints.get(e.ref)
        if fp is not None:
            default = default_body_thermal(fp, e)
            r_a, r_b = terminal_thermal_resistances(fp, e)
            theta = [
                _num(r, "K/W", 3)
                + (
                    ""
                    if given is not None
                    else f' <span class="dim">(default, {_e(default.body_type)})</span>'
                )
                for r, given in zip((r_a, r_b), e.terminal_thermal_resistances, strict=True)
            ]
            if e.thermal_resistance_ambient_k_w is not None:
                ambient = _num(e.thermal_resistance_ambient_k_w, "K/W", 3)
            else:
                g = default_ambient_conductance(fp, project.solve)
                ambient = (
                    f'convection <span class="dim">({_num(1 / g, "K/W", 3) if g > 0 else "none"})</span>'
                )
        else:
            theta, ambient = ["—", "—"], "—"
        heat = results.element_terminal_heat.get(e.name)
        rows.append(
            [
                f"<b>{_e(e.name)}</b>"
                + (f' <span class="dim">({_e(e.ref)})</span>' if e.name != e.ref else ""),
                f"{_e(a.name)}: {_e(', '.join(a.pads))}<br>{_e(b.name)}: {_e(', '.join(b.pads))}",
                _e(format_resistance(e.resistance_ohm)),
                f"{theta[0]}<br>{theta[1]}",
                ambient,
                _num(results.element_currents.get(e.name), "A", signed=True),
                _num(results.element_power.get(e.name), "W"),
                _e(format_value(results.element_temperatures[e.name], DisplayMode.TEMPERATURE))
                if e.name in results.element_temperatures
                else "—",
                f"{_num(heat[0], 'W', signed=True)}<br>{_num(heat[1], 'W', signed=True)}" if heat else "—",
                _num(results.element_ambient_heat.get(e.name), "W"),
            ]
        )
    return rows


def _losses(results: SolveResults) -> str:
    """Where the Joule power goes: the copper, and each resistive element.

    Args:
        results: The results.

    Returns:
        The section's HTML.
    """
    total = results.total_power_w

    def share(watts: float) -> str:
        """Percentage of the total.

        Args:
            watts: Part of the total.

        Returns:
            e.g. ``"20.7%"`` (a dash if there's no loss at all).
        """
        return f"{100 * watts / total:.1f}%" if total > 0 else "—"

    rows = [
        [
            "Copper (tracks, pours, vias, solder joints)",
            _num(results.copper_losses_w, "W"),
            share(results.copper_losses_w),
        ]
    ]
    for name, watts in sorted(results.element_power.items(), key=lambda item: -item[1]):
        rows.append([f"Resistive element {_e(name)}", _num(watts, "W"), share(watts)])
    if results.element_power:
        rows.append(
            ["All resistive elements", _num(results.element_losses_w, "W"), share(results.element_losses_w)]
        )
    rows.append(["<b>Total</b>", f"<b>{_num(total, 'W')}</b>", "<b>100%</b>" if total > 0 else "—"])
    return _table(
        "Losses",
        "Joule power (I²R) dissipated in the copper and in each resistive element, "
        "and its share of the total.",
        ["Where", "Power", "Share"],
        rows,
    )


def _tables(project: KitsProject, board: Board, results: SolveResults) -> str:
    """Every input and its results, as HTML sections.

    Args:
        project: The project.
        board: Its board.
        results: The results.

    Returns:
        The sections' HTML.
    """
    sections = []
    checks = check_limits(results, project.limits)
    sections.append(
        _table(
            "Pass/fail limits",
            "The project's limits. Current density is the "
            f"{current_density_basis(project.limits.current_density_spike_area_pct)}: sharp inside "
            "corners have mesh-dependent spikes.",
            ["Check", "Limit", "Value", "Where", "Margin", "Result"],
            [
                [
                    _e(c.quantity),
                    _num(c.limit, c.unit),
                    _num(c.value, c.unit),
                    _e(c.where),
                    _num(c.margin, c.unit, 3, signed=True),
                    f'<b class="{"pass" if c.passed else "fail"}">{"PASS" if c.passed else "FAIL"}</b>',
                ]
                for c in checks
            ],
            numeric={1, 2, 4, 5},
        )
    )
    sections.append(_losses(results))
    multiplier = project.current_multiplier
    sources = [
        [
            f"<b>{_e(s.ref)}.{_e(s.pad)}</b>",
            _num(s.current_a, "A", signed=True),
            _num(results.electrode_currents.get(f"{s.ref}.{s.pad}"), "A", signed=True),
            _e(format_value(v, DisplayMode.VOLTAGE))
            if (v := results.electrode_voltages.get(f"{s.ref}.{s.pad}")) is not None
            else "—",
        ]
        for s in project.scaled_current_sources()
    ]
    note = "Current into the board at each pad (negative: a sink); the first sink is the 0 V reference."
    if multiplier != 1:
        note += f" Currents include the project's ×{multiplier:g} multiplier."
    sections.append(_table("Current sources", note, ["Pad", "Set", "Solved", "Voltage"], sources))

    kinds = {
        HeatLoadKind.POWER: "power",
        HeatLoadKind.TEMPERATURE: "fixed temperature",
        HeatLoadKind.RESISTANCE: "Rθ to ambient",
    }
    units = {HeatLoadKind.POWER: "W", HeatLoadKind.TEMPERATURE: "°C", HeatLoadKind.RESISTANCE: "K/W"}
    loads = [
        [
            f"<b>{_e(h.label)}</b>",
            f"{_e(kinds[h.kind])}: {_num(h.value, units[h.kind])}",
            _num(results.load_heat.get(h.label), "W", signed=True),
            _e(format_value(t, DisplayMode.TEMPERATURE))
            if (t := results.load_temperatures.get(h.label)) is not None
            else "—",
        ]
        for h in project.active_heat_loads
    ]
    sections.append(
        _table(
            "Heat loads",
            "Heat into the board at each pad (negative: taken out) and the pad's mean temperature.",
            ["Pad", "Load", "Heat into board", "Pad temperature"],
            loads,
            numeric_from=2,
        )
    )

    sections.append(
        _table(
            "Resistive elements",
            "Current from terminal A to B, I²R power and body temperature. Body → terminal heat is positive "
            "where the body heats that pad (negative where the board heats the body).",
            [
                "Element",
                "Terminals (A, B)",
                "R",
                "Rθ body → A, B",
                "Rθ ambient",
                "Current",
                "Power",
                "Body temperature",
                "Heat to A, B",
                "Heat to ambient",
            ],
            _element_rows(project, board, results),
            numeric_from=5,
        )
    )

    sinks = [
        [
            f"<b>{_e(h.name)}</b>",
            f"{_e(h.side)}, on {_e('components' if h.mount == 'components' else 'the board')}",
            f"{h.width_mm:g} × {h.height_mm:g} mm",
            f"{h.thermal_resistance_board_k_w:g} / {h.thermal_resistance_ambient_k_w:g} K/W",
            _num(results.heatsink_heat.get(h.name), "W"),
            _e(format_value(t, DisplayMode.TEMPERATURE))
            if (t := results.heatsink_temperatures.get(h.name)) is not None
            else "—",
        ]
        for h in project.active_heatsinks
    ]
    sections.append(
        _table(
            "Heatsinks",
            "Heat each heatsink takes from the board (and sheds to ambient), and its temperature.",
            ["Heatsink", "Side", "Size", "Rθ contact / ambient", "Heat", "Temperature"],
            sinks,
            numeric_from=4,
        )
    )
    return "".join(s for s in sections if s)


def _settings(project: KitsProject, board: Board) -> str:
    """Solver, mesh and stackup settings as HTML.

    Args:
        project: The project.
        board: Its board (with any layer thickness edits applied).

    Returns:
        The section's HTML.
    """
    s = project.solve
    solver = [
        ("Ambient", f"{s.ambient_c:g} °C"),
        ("Convection, top / bottom", f"{s.convection_top_w_m2k:g} / {s.convection_bottom_w_m2k:g} W/m²K"),
        ("Copper σ", f"{s.copper_conductivity_s_m / 1e6:g} MS/m"),
        ("Copper k", f"{s.copper_thermal_conductivity_w_mk:g} W/mK"),
        ("Dielectric k", f"{s.dielectric_thermal_conductivity_w_mk:g} W/mK"),
        ("Solder mask k", f"{s.solder_mask_thermal_conductivity_w_mk:g} W/mK"),
        ("Via plating", f"{s.plating_thickness_mm * 1000:g} µm"),
        (
            "Via fill",
            {
                "none": "unfilled",
                "copper": "copper",
                "resin": f"resin, {s.via_fill_thermal_conductivity_w_mk:g} W/mK",
            }[s.via_fill],
        ),
    ]
    m = project.mesh
    tracks = {"all": "all", "none": "none", "nets": f"only {', '.join(m.track_nets) or 'no nets'}"}[m.tracks]
    areas = ", ".join(
        f"{a.name} ({'refine ' + format(a.size_mm, 'g') + ' mm' if a.kind == 'refine' else 'excluded'})"
        for a in m.areas
        if a.enabled
    )
    mesh = [
        ("Element size", f"{m.min_size_mm:g} – {m.max_size_mm:g} mm"),
        ("Corner size", f"{m.corner_size_mm:g} mm" if m.corner_size_mm else "off"),
        ("Hole points", f"{m.min_circle_points}"),
        ("Tracks meshed", tracks),
        ("Mesh areas", areas or "none"),
    ]

    def dl(title: str, items: list[tuple[str, str]]) -> str:
        """A card with a definition list.

        Args:
            title: Card heading.
            items: ``(term, value)`` pairs (escaped here).

        Returns:
            The card's HTML.
        """
        rows = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in items)
        return f'<div class="card"><h3>{_e(title)}</h3><dl>{rows}</dl></div>'

    return dl("Materials & environment", solver) + dl("Mesh", mesh)


def _stackup(project: KitsProject, board: Board) -> str:
    """The board stackup, top to bottom, as an HTML table.

    Args:
        project: The project (for the layer thickness edits made in KITS).
        board: Its board, with those edits applied.

    Returns:
        The section's HTML.
    """
    try:
        edited = set(project.stackup_edits_for(board_hash(board.path)))
    except OSError:
        edited = set()
    copper = board.copper_thicknesses()
    heights = layer_heights(board)
    rows, total = [], 0.0
    entries = [
        (s.name, s.type, s.thickness_mm, s.material, s.epsilon_r, s.loss_tangent) for s in board.stackup
    ]
    listed = {name for name, *_ in entries}
    # Copper layers the stackup doesn't describe still conduct, at a default or edited thickness.
    entries += [(name, "copper", None, None, None, None) for name in copper if name not in listed]
    for name, kind, thickness, material, epsilon, loss in entries:
        notes = []
        if name in copper:
            thickness, from_stackup = copper[name]
            if not from_stackup and name not in edited:
                notes.append("not in the stackup: default thickness")
        if name in edited:
            notes.append("thickness edited in KITS")
        if thickness:
            total += thickness
        weight = f"{thickness / OUNCE_COPPER_MM:.2g} oz" if name in copper and thickness else ""
        rows.append(
            [
                f"<b>{_e(name)}</b>" if name in copper else _e(name),
                _e(kind or "—"),
                _e(material or ""),
                f"{thickness * 1000:.4g} µm" if thickness else "—",
                weight,
                f"{epsilon:g}" if epsilon else "",
                f"{loss:g}" if loss else "",
                f"{heights[name]:.4g} mm" if name in heights else "",
                _e("; ".join(notes)),
            ]
        )
    if total:
        listed_total = f" (board file: {board.thickness_mm:g} mm)" if board.thickness_mm else ""
        rows.append(
            ["<b>Total</b>", "", "", f"<b>{total:.4g} mm</b>", "", "", "", "", _e(listed_total.strip())]
        )
    return _table(
        "Layers",
        "Top to bottom, as solved. Heights are each copper layer's mid-plane above the board's bottom face; "
        f"copper weight assumes {OUNCE_COPPER_MM * 1000:g} µm per oz.",
        ["Layer", "Type", "Material", "Thickness", "Copper", "εr", "tan δ", "Height", "Notes"],
        rows,
        numeric={3, 4, 5, 6, 7},
    )


# -- data for the maps -----------------------------------------------------------------


def _bodies(project: KitsProject, board: Board, results: SolveResults) -> list[dict]:
    """Element bodies to draw on the temperature maps, with their figures.

    Args:
        project: The project.
        board: Its board.
        results: The results.

    Returns:
        ``{name, layer, polygons, temperature, power, current}`` per element that has a body.
    """
    footprints = _footprints(board)
    bodies = []
    for e in project.active_resistive_elements:
        fp = footprints.get(e.ref)
        if fp is None or e.name not in results.element_temperatures:
            continue
        area, _source = footprint_body(fp)
        polygons = [
            [[round(x, 4), round(y, 4)] for x, y in p.exterior.coords]
            for p in shapely.get_parts(area)
            if not p.is_empty and p.geom_type == "Polygon"
        ]
        if polygons:
            bodies.append(
                {
                    "name": e.name,
                    "layer": fp.side,
                    "polygons": polygons,
                    "temperature": results.element_temperatures[e.name],
                    "power": results.element_power.get(e.name),
                    "current": results.element_currents.get(e.name),
                }
            )
    return bodies


def _peaks(layer: LayerResult, bodies: list[dict], spike_area_pct: float) -> dict[str, dict]:
    """Where each quantity peaks on a layer, for the map markers.

    Taken from the solution itself rather than the map pixels, so the value is
    exact. Current density is the per-triangle peak, ignoring corner spikes as
    the project's limits say; voltage and temperature are nodal maxima. On the
    temperature map an element body drawn on the layer is the peak if it is
    hotter than the copper.

    Args:
        layer: The layer's results.
        bodies: Element bodies (see :func:`_bodies`).
        spike_area_pct: Copper area ignored as corner spikes, % (0: none).

    Returns:
        ``{mode name: {x, y, value, body}}`` for each quantity with a value on
        the layer; ``body`` names the element when the peak is its body.
    """
    peaks = {}
    for mode in (DisplayMode.CURRENT_DENSITY, DisplayMode.VOLTAGE, DisplayMode.TEMPERATURE):
        values, shown = triangle_values(layer, mode)
        if not shown.any():
            continue
        if mode is DisplayMode.CURRENT_DENSITY:
            found = layer_peak_current_density(layer, spike_area_pct)
            if found is None:
                continue  # no current on this layer: nothing to mark
            value, t = found
            x, y = layer.nodes[layer.triangles[t]].mean(axis=0)
        else:
            nodal = layer.voltage if mode is DisplayMode.VOLTAGE else layer.temperature
            candidates = np.unique(layer.triangles[shown])
            n = int(candidates[np.argmax(nodal[candidates])])
            (x, y), value = layer.nodes[n], float(nodal[n])
        peak = {"x": round(float(x), 4), "y": round(float(y), 4), "value": value, "body": None}
        if mode is DisplayMode.TEMPERATURE:
            for body in bodies:
                if body["layer"] == layer.layer and body["temperature"] > peak["value"]:
                    polygon = shapely.Polygon(max(body["polygons"], key=len))
                    centre = polygon.representative_point()
                    peak = {"x": centre.x, "y": centre.y, "value": body["temperature"], "body": body["name"]}
        peaks[mode.name] = peak
    return peaks


def report_data(
    project: KitsProject,
    board: Board,
    results: SolveResults,
    staleness: str | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    progress: Progress | None = None,
) -> dict:
    """Everything the report page needs, as JSON-ready data.

    Args:
        project: The project.
        board: Its board (with any layer thickness edits applied).
        results: The results to report.
        staleness: Why the results no longer match the project, if they don't.
        max_pixels: Map resolution along the board's longer side.
        progress: Optional callback ``(message, fraction)``.

    Returns:
        The data (see the page's script for how each part is used).
    """
    report = progress or (lambda message, fraction: None)
    nodes = np.concatenate([r.nodes for r in results.layers])
    (bx0, by0), (bx1, by1) = nodes.min(axis=0), nodes.max(axis=0)
    outline = board.bounding_box() or (bx0, by0, bx1, by1)
    grid = Grid.over((bx0, by0, bx1, by1), max_pixels)
    points = grid.centres()

    modes = []
    for mode in (DisplayMode.CURRENT_DENSITY, DisplayMode.VOLTAGE, DisplayMode.TEMPERATURE):
        log = supports_log(mode)
        modes.append(
            {
                "key": mode.name,
                "label": mode.label,
                "unit": mode.unit,
                "anchors": [list(c) for c in mode.value.anchors],
                "range": value_range(results, mode),
                "log_range": value_range(results, mode, log=True) if log else None,
                "log": log,
                "note": (
                    f"Scale tops out at the {CURRENT_DENSITY_PERCENTILE:g}th percentile: sharp inside "
                    "corners have mesh-dependent spikes."
                    if mode is DisplayMode.CURRENT_DENSITY
                    else "Scale shared by all layers."
                ),
            }
        )

    bodies = _bodies(project, board, results)
    spikes = project.limits.current_density_spike_area_pct
    layers = []
    for i, layer in enumerate(results.layers):
        report(f"Rendering {layer.layer}", i / len(results.layers))
        sampled = sample_layer(layer, points)
        layers.append(
            {
                "name": layer.layer,
                "rasters": {
                    mode.name: encode_raster(values, log=supports_log(mode))
                    for mode, values in sampled.items()
                },
                "peaks": _peaks(layer, bodies, spikes),
            }
        )

    voltages = [v for r in results.layers for v in (float(r.voltage.min()), float(r.voltage.max()))]
    return {
        "title": project.name,
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "version": __version__,
        "project": str(project.path),
        "board": str(board.path),
        "staleness": staleness,
        "verdict": verdict(check_limits(results, project.limits)),
        "summary": {
            "power": results.total_power_w,
            "copper_losses": results.copper_losses_w,
            "element_losses": results.element_losses_w,
            "max_temperature": results.max_temperature_c,
            "peak_density": peak_current_density(results, spikes)[0],
            "density_basis": current_density_basis(spikes),
            "voltage_span": max(voltages) - min(voltages) if voltages else 0.0,
            "layers": len(results.layers),
            "triangles": int(sum(len(r.triangles) for r in results.layers)),
        },
        # Hover coordinates are shown like the GUI's grid: from the board's lower-left corner, Y up.
        "grid": {
            "x0": grid.x0,
            "y0": grid.y0,
            "step": grid.step,
            "width": grid.width,
            "height": grid.height,
            "origin": [outline[0], outline[3]],
        },
        "modes": modes,
        "layers": layers,
        "bodies": bodies,
    }


def render_report(project: KitsProject, board: Board, results: SolveResults, data: dict) -> str:
    """Fill the report template.

    Args:
        project: The project.
        board: Its board.
        results: The results.
        data: :func:`report_data`.

    Returns:
        The page's HTML.
    """
    template = resources.files("kits.report").joinpath("template.html").read_text(encoding="utf-8")
    # "</" can't appear inside a <script> element; JSON allows "<\/" for it.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    replacements = {
        "<!--KITS:TITLE-->": _e(project.name),
        "<!--KITS:TABLES-->": _tables(project, board, results),
        "<!--KITS:SETTINGS-->": _settings(project, board),
        "<!--KITS:STACKUP-->": _stackup(project, board),
        _DATA_MARK: payload,
    }
    for mark, value in replacements.items():
        if mark not in template:
            raise RuntimeError(f"report template is missing {mark}")
        template = template.replace(mark, value)
    return template


def write_report(
    project: KitsProject,
    board: Board,
    results: SolveResults,
    path: str | Path,
    staleness: str | None = None,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    progress: Progress | None = None,
) -> Path:
    """Write the HTML report of some results.

    Args:
        project: The project.
        board: Its board (with any layer thickness edits applied).
        results: The results.
        path: Destination ``.html`` file; parent folders are created.
        staleness: Why the results no longer match the project (shown as a warning), if they don't.
        max_pixels: Map resolution along the board's longer side.
        progress: Optional callback ``(message, fraction)``.

    Returns:
        The written file.

    Raises:
        OSError: If the file can't be written.
    """
    report = progress or (lambda message, fraction: None)
    data = report_data(project, board, results, staleness, max_pixels, lambda m, f: report(m, 0.9 * f))
    report("Writing the report", 0.9)
    page = render_report(project, board, results, data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    report("Done", 1.0)
    return path
