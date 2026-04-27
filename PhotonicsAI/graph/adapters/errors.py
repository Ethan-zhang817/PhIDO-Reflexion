"""Structured exceptions raised by the EDA adapters.

The legacy ``DemoPDK.yaml_netlist_to_gds`` swallows tool failures with
``print`` calls, which is fine for human inspection but useless for the
Reflector agent. The adapters in this package re-throw the same
conditions as typed exceptions so the executor node can serialize them
into ``EdaReport``.
"""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for any structured adapter error."""


class SaxModelMissingError(AdapterError):
    """Raised when a netlist references SAX models that are not installed.

    Attributes:
        missing: cell names referenced by the netlist but absent from the
            ``all_models`` registry.
        available: cell names registered in ``DemoPDK.all_models`` (used
            by the Reflector to suggest a replacement device).
    """

    def __init__(
        self,
        missing: list[str],
        available: list[str],
        diagnostics: list[str] | None = None,
    ) -> None:
        self.missing = sorted(set(missing))
        self.available = sorted(set(available))
        self.diagnostics = diagnostics or []
        diag_text = (
            " Diagnostics: " + "; ".join(self.diagnostics[:5])
            if self.diagnostics
            else ""
        )
        super().__init__(
            f"SAX model(s) missing: {self.missing}. "
            f"({len(self.available)} models available.){diag_text}"
        )


class NetlistError(AdapterError):
    """Raised when GDSFactory cannot consume the YAML netlist."""

    def __init__(self, message: str, *, stage: str = "from_yaml") -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class GdsBuildError(AdapterError):
    """Raised when ``gf.Component.write_gds`` fails (layer numbers, routing, ...)."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        self.hint = hint
        super().__init__(message if hint is None else f"{message} (hint: {hint})")
