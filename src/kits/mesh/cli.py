"""Command-line mesher: ``kits-mesh project.kicad_pro``.

Reads the board and its ``.kits.yaml`` mesh settings, meshes every copper layer
and writes ``kits_output/<project>.msh`` next to the project. The GUI runs this
same entry point in a child process with ``--progress`` so a gmsh crash can't
take the GUI down and meshing can be cancelled.

With ``--progress``, stdout carries one JSON object per line:

* ``{"progress": 0.42, "message": "Meshing F.Cu (1/2)"}`` while running;
* ``{"done": true, "msh": "...", "nodes": N, "triangles": M}`` on success;
* ``{"error": "..."}`` on failure (exit status 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import ConfigError, config_path_for, load_config
from ..ingest import BoardLoadError, load_project
from .geometry import GeometryError
from .mesher import MeshError, mesh_board

OUTPUT_DIR = "kits_output"


def default_msh_path(project_path: str | Path) -> Path:
    """Return where the mesh for a project is written by default.

    Args:
        project_path: The ``.kicad_pro`` or ``.kicad_pcb`` path.

    Returns:
        ``<project dir>/kits_output/<stem>.msh``.
    """
    project_path = Path(project_path)
    return project_path.parent / OUTPUT_DIR / f"{project_path.stem}.msh"


def _emit(obj: dict) -> None:
    """Write one JSON progress record to stdout and flush it.

    Args:
        obj: The record.
    """
    print(json.dumps(obj), flush=True)


def main(argv: list[str] | None = None) -> int:
    """Mesh a KiCad project from the command line.

    Args:
        argv: Arguments excluding the program name; ``None`` uses ``sys.argv``.

    Returns:
        Process exit status: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(prog="kits-mesh", description="Mesh a KiCad board for KITS")
    parser.add_argument("project", help=".kicad_pro (or .kicad_pcb) file")
    parser.add_argument("-o", "--output", help="output .msh path (default: kits_output/<project>.msh)")
    parser.add_argument("--npz", help="also save the mesh arrays to this .npz file")
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
        project = load_project(args.project)
        config = load_config(config_path_for(project.path))
        msh_path = Path(args.output) if args.output else default_msh_path(project.path)
        # Fail fast on an unwritable output location (e.g. a project under
        # Program Files) rather than after minutes of meshing.
        try:
            msh_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MeshError(f"cannot create output folder {msh_path.parent}: {exc.strerror or exc}") from exc
        mesh = mesh_board(project.board, config.mesh, msh_path, progress)
        if args.npz:
            mesh.save_npz(args.npz)
    except (BoardLoadError, ConfigError, GeometryError, MeshError, OSError) as exc:
        if args.progress:
            _emit({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.progress:
        _emit({"done": True, "msh": str(msh_path), "nodes": mesh.num_nodes, "triangles": mesh.num_triangles})
    else:
        print(f"Wrote {msh_path}: {mesh.num_nodes} nodes, {mesh.num_triangles} triangles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
