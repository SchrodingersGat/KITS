"""Solved fields per layer, saved with fingerprints of everything they depend on."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import CurrentSource
from ..mesh.fingerprint import MeshFingerprint
from .solver import SolveSettings

# Bump when the physics model changes in a way that alters results.
SOLVER_VERSION = 2  # 2: stackup sub-layers summed; top/bottom convection


def inputs_hash(sources: list[CurrentSource], settings: SolveSettings) -> str:
    """Fingerprint the solve inputs that aren't part of the mesh.

    Args:
        sources: Current sources (order doesn't matter).
        settings: Material and boundary settings.

    Returns:
        A hex SHA-256 digest.
    """
    payload = {
        "solver_version": SOLVER_VERSION,
        "sources": sorted((s.ref, s.pad, s.current_a) for s in sources),
        "settings": dataclasses.asdict(settings),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class LayerResult:
    """Solved fields on one layer's mesh.

    Attributes:
        layer: Copper layer name.
        nodes: ``(N, 2)`` node coordinates in board millimetres.
        triangles: ``(M, 3)`` node indices.
        copper: ``(M,)`` True for copper triangles.
        voltage: ``(N,)`` potential in volts.
        temperature: ``(N,)`` temperature in °C.
        current_density: ``(M,)`` current density magnitude in A/m² (zero off copper).
    """

    layer: str
    nodes: np.ndarray
    triangles: np.ndarray
    copper: np.ndarray
    voltage: np.ndarray
    temperature: np.ndarray
    current_density: np.ndarray


@dataclass
class SolveResults:
    """Results of one DC + thermal solve.

    Attributes:
        layers: Per-layer fields, top to bottom.
        total_power_w: Total Joule power.
        electrode_voltages: Mean potential of each source pad, keyed ``"REF.PAD"``.
        electrode_currents: Current into the board at each source pad, keyed ``"REF.PAD"``.
        mesh_fingerprint: Fingerprint of the board and mesh settings the mesh was built from.
        inputs: :func:`inputs_hash` of the sources and solver settings.
    """

    layers: list[LayerResult]
    total_power_w: float
    electrode_voltages: dict[str, float]
    electrode_currents: dict[str, float]
    mesh_fingerprint: MeshFingerprint = field(default_factory=MeshFingerprint)
    inputs: str = ""

    def layer(self, name: str) -> LayerResult | None:
        """Return one layer's results.

        Args:
            name: Copper layer name.

        Returns:
            The layer's results, or None.
        """
        return next((r for r in self.layers if r.layer == name), None)

    @property
    def max_temperature_c(self) -> float:
        """Highest temperature on any layer."""
        return max(float(r.temperature.max()) for r in self.layers)

    def save_npz(self, path: str | Path) -> None:
        """Save to a ``.npz`` file (no pickled objects).

        Args:
            path: Destination file.
        """
        meta = {
            "layers": [r.layer for r in self.layers],
            "total_power_w": self.total_power_w,
            "electrode_voltages": self.electrode_voltages,
            "electrode_currents": self.electrode_currents,
            "mesh_settings_hash": self.mesh_fingerprint.settings,
            "mesh_board_hash": self.mesh_fingerprint.board,
            "inputs": self.inputs,
        }
        arrays: dict[str, np.ndarray] = {"meta": np.array(json.dumps(meta))}
        for i, r in enumerate(self.layers):
            for name in ("nodes", "triangles", "copper", "voltage", "temperature", "current_density"):
                arrays[f"{name}_{i}"] = getattr(r, name)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load_npz(cls, path: str | Path) -> SolveResults:
        """Load results written by :meth:`save_npz`.

        Args:
            path: Source file.

        Returns:
            The results.
        """
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            layers = [
                LayerResult(
                    layer=name,
                    nodes=data[f"nodes_{i}"],
                    triangles=data[f"triangles_{i}"],
                    copper=data[f"copper_{i}"],
                    voltage=data[f"voltage_{i}"],
                    temperature=data[f"temperature_{i}"],
                    current_density=data[f"current_density_{i}"],
                )
                for i, name in enumerate(meta["layers"])
            ]
        return cls(
            layers=layers,
            total_power_w=float(meta["total_power_w"]),
            electrode_voltages={k: float(v) for k, v in meta["electrode_voltages"].items()},
            electrode_currents={k: float(v) for k, v in meta["electrode_currents"].items()},
            mesh_fingerprint=MeshFingerprint(meta["mesh_settings_hash"], meta["mesh_board_hash"]),
            inputs=str(meta["inputs"]),
        )

    def staleness(self, mesh_fingerprint: MeshFingerprint, inputs: str) -> str | None:
        """Explain why these results don't match the current project, if they don't.

        Args:
            mesh_fingerprint: Fingerprint of the current board and mesh settings.
            inputs: :func:`inputs_hash` of the current sources and solver settings.

        Returns:
            None if up to date, otherwise a human-readable reason.
        """
        reason = self.mesh_fingerprint.staleness(mesh_fingerprint)
        if reason is not None:
            return reason.replace("saved mesh", "results")
        if self.inputs != inputs:
            return "The current sources or solver settings changed since the results were computed"
        return None


def load_saved_results(
    path: str | Path, mesh_fingerprint: MeshFingerprint, inputs: str
) -> tuple[SolveResults | None, str | None]:
    """Load saved results if they match the current project.

    Args:
        path: The saved results (``.results.npz``).
        mesh_fingerprint: Fingerprint of the current board and mesh settings.
        inputs: :func:`inputs_hash` of the current sources and solver settings.

    Returns:
        ``(results, None)`` if current; ``(None, reason)`` if out of date or
        unreadable; ``(None, None)`` if there are no saved results.
    """
    path = Path(path)
    if not path.is_file():
        return None, None
    try:
        results = SolveResults.load_npz(path)
    except (OSError, ValueError, KeyError) as exc:
        return None, f"The saved results couldn't be read ({exc})"
    reason = results.staleness(mesh_fingerprint, inputs)
    return (None, reason) if reason else (results, None)
