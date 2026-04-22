"""KLayout DRC runner that returns a structured :class:`DRCReport`.

The legacy ``PhotonicsAI/Photon/drc/drc.py`` only ``print``s its output and
writes a ``.lydrb`` file. This adapter:

1. Locates the ``klayout`` binary, returning a "skipped" report when it
   cannot be found instead of raising. The Reflexion loop must not spin
   on missing system dependencies.
2. Invokes the new ``drc_script_xml.drc`` so the report is XML.
3. Delegates parsing to :func:`PhotonicsAI.graph.adapters.drc_parser.parse_lyrdb_xml`.
4. Optionally accepts ``wmin`` / ``gmin`` overrides so unit tests can
   force violations.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PhotonicsAI.config import PATH
from PhotonicsAI.graph.adapters.drc_parser import parse_lyrdb_xml
from PhotonicsAI.graph.adapters.eda_report import DRCReport


_KLAYOUT_CANDIDATES = (
    "klayout",
    "/usr/bin/klayout",
    "/usr/local/bin/klayout",
    "/opt/klayout/bin/klayout",
)

DEFAULT_DRC_SCRIPT = (
    PATH.photon / "drc" / "drc_script_xml.drc"
)


def find_klayout() -> Optional[str]:
    """Return the absolute path to the ``klayout`` binary, or ``None``."""
    which = shutil.which("klayout")
    if which:
        return which
    for candidate in _KLAYOUT_CANDIDATES:
        try:
            result = subprocess.run(
                [candidate, "-v"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def run_drc_structured(
    gds_path: str | Path,
    *,
    drc_script: str | Path | None = None,
    report_path: str | Path | None = None,
    wmin: float | None = None,
    gmin: float | None = None,
    timeout: int = 60,
    top_k: int = 3,
) -> DRCReport:
    """Run KLayout DRC on ``gds_path`` and return a structured report.

    Args:
        gds_path: input GDS to check.
        drc_script: override for the DRC script path (defaults to the new
            XML-emitting script in the legacy ``Photon/drc`` folder).
        report_path: where to write the ``.lyrdb`` file. Defaults to a
            sibling of ``gds_path``.
        wmin / gmin: override Si width / spacing rule. Used by tests.
        timeout: KLayout subprocess timeout in seconds.
        top_k: max examples to keep per rule.
    """
    gds_path = Path(gds_path)
    drc_script = Path(drc_script) if drc_script else DEFAULT_DRC_SCRIPT
    if report_path is None:
        report_path = gds_path.with_suffix(".lyrdb")
    report_path = Path(report_path)

    klayout = find_klayout()
    if klayout is None:
        return DRCReport(
            ok=True,  # treated as a soft pass downstream, with a warning
            skipped_reason="klayout_not_found",
            raw_report_path=str(report_path),
        )
    if not gds_path.exists():
        return DRCReport(
            ok=False,
            skipped_reason=f"gds_not_found: {gds_path}",
            raw_report_path=str(report_path),
        )
    if not drc_script.exists():
        return DRCReport(
            ok=False,
            skipped_reason=f"drc_script_not_found: {drc_script}",
            raw_report_path=str(report_path),
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        report_path.unlink()

    cmd = [
        klayout,
        "-b",
        "-r",
        str(drc_script),
        "-rd",
        f"input_gds={gds_path}",
        "-rd",
        f"report={report_path}",
    ]
    if wmin is not None:
        cmd += ["-rd", f"wmin={wmin}"]
    if gmin is not None:
        cmd += ["-rd", f"gmin={gmin}"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return DRCReport(
            ok=False,
            skipped_reason=f"klayout_timeout_after_{timeout}s",
            raw_report_path=str(report_path),
        )
    except OSError as exc:
        return DRCReport(
            ok=False,
            skipped_reason=f"klayout_exec_error: {exc}",
            raw_report_path=str(report_path),
        )

    if result.returncode != 0:
        return DRCReport(
            ok=False,
            skipped_reason=(
                f"klayout_returncode={result.returncode}: "
                f"{(result.stderr or result.stdout).strip()[:200]}"
            ),
            raw_report_path=str(report_path),
        )

    return parse_lyrdb_xml(report_path, top_k=top_k)
