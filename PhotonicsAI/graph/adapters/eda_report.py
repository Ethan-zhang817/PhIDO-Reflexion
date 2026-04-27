"""Typed report objects shared across the EDA adapters.

The shape mirrors REFACTOR_PLAN.md section 4.5:

    {
      "ok": False,
      "stages": {
        "gdsfactory": {"ok": True,  "errors": []},
        "sax":        {"ok": False, "missing_models": [...], "errors": [...]},
        "drc": {
          "ok": False,
          "n_violations": 42,
          "by_rule": {"Si_width": 0, "Si_space": 42},
          "examples": [{"rule": "Si_space", "coord": [...], "actual": ..., "min": ...}],
          "skipped_reason": null
        }
      },
      "summary_text": "DRC: 42 violations (all Si_space)..."
    }
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class GdsReport(BaseModel):
    """GDSFactory build stage result."""

    ok: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    """Non-fatal notes (e.g. "auto-derived top-level ports")."""
    routing_failed: bool = False
    layer_overflow: bool = False
    gds_path: Optional[str] = None
    gds_png_path: Optional[str] = None
    """PNG snapshot of the layout (from ``gf.Component.plot``)."""


class SaxReport(BaseModel):
    """SAX simulation stage result."""

    ok: bool = True
    missing_models: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    required_models: list[str] = Field(default_factory=list)


class DRCExample(BaseModel):
    """A single representative DRC violation, used as few-shot context for the Reflector."""

    rule: str
    category: str = ""
    cell: str = ""
    coord: list[float] = Field(default_factory=list)
    actual: Optional[float] = None
    min: Optional[float] = None
    description: str = ""


class DRCReport(BaseModel):
    """Aggregated DRC stage result.

    ``ok`` is True when there are zero violations. ``skipped_reason`` is set
    (and ``ok`` becomes None semantically) when KLayout itself is missing –
    the evaluator treats that as a soft pass with a warning rather than
    spinning the Reflexion loop forever.
    """

    ok: bool = True
    n_violations: int = 0
    by_rule: dict[str, int] = Field(default_factory=dict)
    examples: list[DRCExample] = Field(default_factory=list)
    skipped_reason: Optional[str] = None
    raw_report_path: Optional[str] = None


class EdaReport(BaseModel):
    """Top-level structured report passed to the Evaluator and Reflector."""

    ok: bool = True
    stages: dict[str, GdsReport | SaxReport | DRCReport] = Field(default_factory=dict)
    summary_text: str = ""

    def get_gds(self) -> GdsReport:
        return self.stages.get("gdsfactory", GdsReport())  # type: ignore[return-value]

    def get_sax(self) -> SaxReport:
        return self.stages.get("sax", SaxReport())  # type: ignore[return-value]

    def get_drc(self) -> DRCReport:
        return self.stages.get("drc", DRCReport())  # type: ignore[return-value]


def _truncate(text: str, *, limit: int = 400) -> str:
    """Return ``text`` collapsed to ``limit`` chars with a "..." marker.

    Collapses newlines/whitespace to keep the summary one-error-per-line
    legible; real error listings (like pydantic's "valid values") are
    almost always uninteresting past the first sentence.
    """
    if not text:
        return ""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def summarize_report(report: EdaReport, *, top_k: int = 3) -> str:
    """Generate a short human/LLM readable summary for the Reflector prompt.

    Keeps the output well under 200 tokens even when the DRC report is huge,
    which is critical because we ship this string into every Reflector call.
    """
    parts: list[str] = []

    gds = report.get_gds()
    sax = report.get_sax()
    drc = report.get_drc()

    if gds.ok:
        parts.append("GDSFactory: ok")
    else:
        flags = []
        if gds.routing_failed:
            flags.append("routing failed")
        if gds.layer_overflow:
            flags.append("layer overflow")
        flags_text = (" [" + ", ".join(flags) + "]") if flags else ""
        first_error = gds.errors[0] if gds.errors else "unknown error"
        # gdsfactory pydantic validation errors embed the whole PDK cell
        # catalog (several kB). Truncate aggressively so the Reflector
        # gets the actionable head of the error, not an 8-kB wall.
        parts.append(
            f"GDSFactory: FAIL{flags_text}: {_truncate(first_error, limit=400)}"
        )

    if sax.ok and not sax.missing_models:
        parts.append(
            "SAX: ok"
            + (f" ({len(sax.required_models)} models)" if sax.required_models else "")
        )
    else:
        miss_preview = ", ".join(sax.missing_models[:5])
        if len(sax.missing_models) > 5:
            miss_preview += ", ..."
        parts.append(
            f"SAX: FAIL — missing models: [{miss_preview}]"
            f" (of {len(sax.required_models)} required)"
        )
        if sax.errors:
            parts.append(f"  sax error: {sax.errors[0]}")
        if sax.warnings:
            parts.append(f"  sax warning: {sax.warnings[0]}")

    if drc.skipped_reason:
        parts.append(f"DRC: skipped ({drc.skipped_reason})")
    elif drc.ok:
        parts.append("DRC: ok (0 violations)")
    else:
        rule_breakdown = ", ".join(
            f"{rule}={count}" for rule, count in drc.by_rule.items() if count > 0
        )
        parts.append(
            f"DRC: {drc.n_violations} violations [{rule_breakdown}]"
        )
        for ex in drc.examples[:top_k]:
            coord_text = (
                f"@({ex.coord[0]:.2f},{ex.coord[1]:.2f})"
                if len(ex.coord) >= 2
                else ""
            )
            actual_text = (
                f" actual={ex.actual:.3f}" if ex.actual is not None else ""
            )
            min_text = f" min={ex.min:.3f}" if ex.min is not None else ""
            parts.append(
                f"  - {ex.rule}{coord_text}{actual_text}{min_text}"
            )

    return "\n".join(parts)


def build_eda_report(
    gds: GdsReport, sax: SaxReport, drc: DRCReport, *, top_k: int = 3
) -> EdaReport:
    """Aggregate three stage reports into a single ``EdaReport``."""
    drc_pass = drc.ok or drc.skipped_reason is not None
    overall_ok = gds.ok and sax.ok and drc_pass and not sax.missing_models
    report = EdaReport(
        ok=overall_ok,
        stages={"gdsfactory": gds, "sax": sax, "drc": drc},
    )
    report.summary_text = summarize_report(report, top_k=top_k)
    return report
