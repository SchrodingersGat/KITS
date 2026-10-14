"""Command-line solver: ``kits-solve project.kits``.

Loads the project and its board, reuses the saved mesh if it is still current
(otherwise meshes the board and saves the mesh), solves DC current and steady
temperature for the project's current sources, and saves the results to
``kits_output/<project>.results.npz``. The GUI runs this in a child process
with ``--progress``; the output protocol matches ``kits-mesh``:

* ``{"progress": 0.42, "message": "..."}`` while running;
* ``{"done": true, "results": "<path>", "mesh": "<path>", "power_w": P, "max_temperature_c": T}``;
* ``{"error": "..."}`` on failure (exit status 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import ConfigError
from ..ingest import BoardLoadError, load_project
from ..mesh.geometry import GeometryError
from ..mesh.mesher import MeshError, mesh_board
from ..mesh.model import load_saved_mesh
from ..project import PROJECT_SUFFIX, load_kits_project
from .results import LayerResult, SolveResults, inputs_hash
from .solver import Electrode, LayeredModel, SolveError

_BALANCE_TOLERANCE_A = 1e-9


def _emit(obj: dict) -> None:
    """Write one JSON progress record to stdout and flush it.

    Args:
        obj: The record.
    """
    print(json.dumps(obj), flush=True)


def solve_project(project_path: str | Path, progress=None) -> tuple[SolveResults, Path, Path]:
    """Mesh (if needed) and solve a project, saving mesh and results.

    Args:
        project_path: The ``.kits`` file.
        progress: Optional callback ``(message, fraction)``.

    Returns:
        ``(results, results_path, saved_mesh_path)``.

    Raises:
        ConfigError: If the project is invalid, has no board, or its sources don't balance.
        BoardLoadError: If the board can't be loaded.
        GeometryError: If the board can't be meshed (e.g. no outline).
        MeshError: If gmsh fails.
        SolveError: If a source pad can't be found or the problem is ill-posed.
        OSError: If output files can't be written.
    """
    report = progress or (lambda message, fraction: None)
    if Path(project_path).suffix != PROJECT_SUFFIX:
        raise ConfigError(f"expected a KITS project file ({PROJECT_SUFFIX}), not {Path(project_path).name}")
    kits = load_kits_project(project_path)
    if kits.board is None:
        raise ConfigError(f"{kits.path.name} doesn't link a KiCad board")
    sources = kits.scaled_current_sources()
    if not sources:
        raise ConfigError("the project has no current sources; add some before solving")
    if abs(sum(s.current_a for s in sources)) > _BALANCE_TOLERANCE_A:
        raise ConfigError("current sources and sinks must sum to zero")
    try:
        kits.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create output folder {kits.output_dir}: {exc.strerror or exc}") from exc

    report("Loading board", 0.0)
    board = load_project(kits.board).board
    mesh, _ = load_saved_mesh(kits.saved_mesh_path, board.path, kits.mesh)
    if mesh is None:
        mesh = mesh_board(board, kits.mesh, kits.msh_path, lambda m, f: report(m, 0.6 * f))
        mesh.save_npz(kits.saved_mesh_path)
        start = 0.6
    else:
        report("Using the saved mesh", 0.05)
        start = 0.05

    settings = kits.solve
    span = 1.0 - start
    report("Assembling the model", start)
    model = LayeredModel(board, mesh, settings)
    report("Solving current flow", start + 0.1 * span)
    dc = model.solve_dc([Electrode(s.ref, s.pad, current=s.current_a) for s in sources])
    report("Solving temperature", start + 0.45 * span)
    thermal = model.solve_thermal(dc)

    report("Saving results", start + 0.9 * span)
    results = SolveResults(
        layers=[
            LayerResult(
                layer=lm.layer,
                nodes=lm.nodes,
                triangles=lm.triangles,
                copper=lm.copper,
                voltage=dc.voltage[lm.layer],
                temperature=thermal.temperature[lm.layer],
                current_density=dc.current_density[lm.layer],
            )
            for lm in mesh.layers
        ],
        total_power_w=dc.total_power,
        electrode_voltages={f"{ref}.{pad}": v for (ref, pad), v in dc.electrode_voltages.items()},
        electrode_currents={f"{ref}.{pad}": i for (ref, pad), i in dc.electrode_currents.items()},
        mesh_fingerprint=mesh.fingerprint,
        inputs=inputs_hash(sources, settings),
    )
    results.save_npz(kits.results_path)
    report("Done", 1.0)
    return results, kits.results_path, kits.saved_mesh_path


def main(argv: list[str] | None = None) -> int:
    """Solve a KITS project from the command line.

    Args:
        argv: Arguments excluding the program name; ``None`` uses ``sys.argv``.

    Returns:
        Process exit status: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(prog="kits-solve", description="Solve current and temperature for KITS")
    parser.add_argument("project", help="KITS project file (.kits)")
    parser.add_argument("--progress", action="store_true", help="report progress as JSON lines on stdout")
    args = parser.parse_args(argv)

    def progress(message: str, fraction: float) -> None:
        """Report progress in the selected output format.

        Args:
            message: Description of the current step.
            fraction: Overall completion, 0 to 1.
        """
        if args.progress:
            _emit({"progress": round(fraction, 4), "message": message})
        else:
            print(f"[{fraction:4.0%}] {message}", flush=True)

    try:
        results, results_path, mesh_path = solve_project(args.project, progress)
    except (BoardLoadError, ConfigError, GeometryError, MeshError, SolveError, OSError) as exc:
        if args.progress:
            _emit({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = {"power_w": results.total_power_w, "max_temperature_c": results.max_temperature_c}
    if args.progress:
        _emit({"done": True, "results": str(results_path), "mesh": str(mesh_path), **summary})
    else:
        print(f"Wrote {results_path}")
        print(f"Joule power {results.total_power_w:.4g} W, max temperature {results.max_temperature_c:.2f} C")
        for label, voltage in results.electrode_voltages.items():
            print(f"  {label}: {voltage * 1e3:+.3f} mV, {results.electrode_currents[label]:+.4g} A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
