"""Fingerprints of a mesh's inputs, so a saved mesh is only reused while still valid.

A mesh depends on two inputs:

* the mesh settings (plus :data:`MESHER_VERSION`, bumped whenever a change to
  the geometry/meshing code would produce a different mesh), and
* the board file's exact contents.

Each is hashed separately (SHA-256), so a stale mesh can be reported with the
reason: "the mesh settings changed" vs "the board changed".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ..config import MeshSettings

# Bump when geometry extraction or meshing changes in a way that alters results.
# 2: holes cut; 3: pad outlines embedded; 4: stackup sub-layers summed (layer heights);
# 5: via pad rings embedded.
MESHER_VERSION = 5

_CHUNK = 1 << 20


def settings_hash(settings: MeshSettings) -> str:
    """Fingerprint the mesh settings and mesher version.

    Args:
        settings: The mesh settings.

    Returns:
        A hex SHA-256 digest.
    """
    payload = json.dumps({"mesher_version": MESHER_VERSION, **settings.effective().to_dict()}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def board_hash(board_file: str | Path) -> str:
    """Fingerprint a board file's contents.

    Args:
        board_file: The ``.kicad_pcb`` file.

    Returns:
        A hex SHA-256 digest.

    Raises:
        OSError: If the file can't be read.
    """
    digest = hashlib.sha256()
    with open(board_file, "rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MeshFingerprint:
    """The input fingerprints a mesh was built from.

    Attributes:
        settings: :func:`settings_hash` of the mesh settings.
        board: :func:`board_hash` of the board file.
    """

    settings: str = ""
    board: str = ""

    @classmethod
    def of(cls, board_file: str | Path, settings: MeshSettings) -> MeshFingerprint:
        """Fingerprint the current inputs.

        Args:
            board_file: The ``.kicad_pcb`` file.
            settings: The mesh settings.

        Returns:
            The fingerprint.

        Raises:
            OSError: If the board file can't be read.
        """
        return cls(settings=settings_hash(settings), board=board_hash(board_file))

    def staleness(self, current: MeshFingerprint) -> str | None:
        """Explain why a mesh with this fingerprint doesn't match the current inputs.

        Args:
            current: The fingerprint of the current board and settings.

        Returns:
            None if the mesh is up to date, otherwise a human-readable reason.
        """
        changed = []
        if self.board != current.board:
            changed.append("the board")
        if self.settings != current.settings:
            changed.append("the mesh settings")
        if not changed:
            return None
        return f"{' and '.join(changed).capitalize()} changed since the saved mesh was built"
