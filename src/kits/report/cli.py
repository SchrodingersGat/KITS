"""Command-line report: ``kits-report project.kits [-o report.html] [--solve]``.

Writes the HTML report of the project's saved results. Results that no longer
match the project are reported with a warning (and a banner in the report);
``--solve`` solves first whenever they're missing or out of date. The GUI runs
this in a child process with ``--progress``; the output protocol matches
``kits-solve``:

* ``{"progress": 0.42, "message": "..."}`` while running;
* ``{"done": true, "report": "<path>", "stale": "<reason>" | null}``;
* ``{"error": "..."}`` on failure (exit status 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import ConfigError
from ..ingest import BoardLoadError, load_project
from ..mesh.fingerprint import MeshFingerprint, board_hash, settings_hash
from ..mesh.geometry import GeometryError
from ..mesh.mesher import MeshError
from ..project import PROJECT_SUFFIX, KitsProject, load_kits_project
from ..solve.cli import solve_project
from ..solve.results import SolveResults, inputs_hash, load_saved_results
from ..solve.solver import SolveError
from .builder import DEFAULT_MAX_PIXELS, write_report


def default_report_path(project: KitsProject) -> Path:
    """Where a project's report goes unless told otherwise.

    Args:
        project: The project.

    Returns:
        ``kits_output/<project>.report.html`` next to the project file.
    """
    return project.output_dir / f"{project.name}.report.html"


def _emit(obj: dict) -> None:
    """Write one JSON progress record to stdout and flush it.

    Args:
        obj: The record.
    """
    print(json.dumps(obj), flush=True)


def generate_report(
    project_path: str | Path,
    output: str | Path | None = None,
    solve: bool = False,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    progress=None,
) -> tuple[Path, str | None]:
    """Write a project's HTML report from its saved results (solving first if asked).

    Args:
        project_path: The ``.kits`` file.
        output: Report file; :func:`default_report_path` if omitted.
        solve: Solve first if the saved results are missing or out of date.
        max_pixels: Map resolution along the board's longer side.
        progress: Optional callback ``(message, fraction)``.

    Returns:
        ``(report_path, staleness)``: the written file, and why the reported
        results don't match the project (None if they do).

    Raises:
        ConfigError: If the project is invalid, has no board, or has no results (and ``solve`` is off).
        BoardLoadError: If the board can't be loaded.
        GeometryError: If solving needs a mesh and the board can't be meshed.
        MeshError: If gmsh fails.
        SolveError: If solving fails.
        OSError: If a file can't be read or written.
    """
    report = progress or (lambda message, fraction: None)
    if Path(project_path).suffix != PROJECT_SUFFIX:
        raise ConfigError(f"expected a KITS project file ({PROJECT_SUFFIX}), not {Path(project_path).name}")
    project = load_kits_project(project_path)
    if project.board is None:
        raise ConfigError(f"{project.path.name} doesn't link a KiCad board")

    report("Loading board", 0.0)
    board = load_project(project.board).board
    digest = board_hash(board.path)
    stackup = project.stackup_edits_for(digest)
    board = board.with_thickness_overrides(stackup)
    fingerprint = MeshFingerprint(settings=settings_hash(project.mesh), board=digest)
    inputs = inputs_hash(
        project.scaled_current_sources(),
        project.solve,
        project.active_heat_loads,
        project.active_resistive_elements,
        stackup,
        project.active_heatsinks,
    )
    results, staleness = load_saved_results(project.results_path, fingerprint, inputs)
    start = 0.05
    if results is None and solve:
        report("Solving", 0.05)
        results, _, _ = solve_project(project.path, lambda m, f: report(m, 0.05 + 0.6 * f))
        staleness, start = None, 0.65
    elif results is None and staleness is None:
        raise ConfigError(
            f"{project.path.name} has no saved results: solve it first (or pass --solve to solve now)"
        )
    elif results is None:
        # Out of date (or unreadable): report what's there, with a warning.
        try:
            results = SolveResults.load_npz(project.results_path)
        except (OSError, ValueError, KeyError) as exc:
            raise ConfigError(f"{staleness}; pass --solve to solve again") from exc

    path = Path(output) if output else default_report_path(project)
    written = write_report(
        project, board, results, path, staleness, max_pixels, lambda m, f: report(m, start + (1 - start) * f)
    )
    return written, staleness


def main(argv: list[str] | None = None) -> int:
    """Write a project's HTML report from the command line.

    Args:
        argv: Arguments excluding the program name; ``None`` uses ``sys.argv``.

    Returns:
        Process exit status: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(prog="kits-report", description="Write an HTML report of a KITS solve")
    parser.add_argument("project", help="KITS project file (.kits)")
    parser.add_argument("-o", "--output", help="report file (default: kits_output/<project>.report.html)")
    parser.add_argument(
        "--solve", action="store_true", help="solve first if the saved results are missing or out of date"
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help=f"map resolution along the board's longer side (default {DEFAULT_MAX_PIXELS})",
    )
    parser.add_argument("--progress", action="store_true", help="report progress as JSON lines on stdout")
    args = parser.parse_args(argv)
    if not 50 <= args.max_pixels <= 4000:
        parser.error("--max-pixels must be between 50 and 4000")

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
        path, staleness = generate_report(args.project, args.output, args.solve, args.max_pixels, progress)
    except (BoardLoadError, ConfigError, GeometryError, MeshError, SolveError, OSError) as exc:
        if args.progress:
            _emit({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.progress:
        _emit({"done": True, "report": str(path), "stale": staleness})
    else:
        if staleness:
            print(f"warning: {staleness}; the report says so (use --solve to solve again)", file=sys.stderr)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
