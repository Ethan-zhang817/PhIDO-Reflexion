"""Structured EDA adapters that turn print/binary output into typed reports."""

from PhotonicsAI.graph.adapters.eda_report import (
    DRCExample,
    DRCReport,
    EdaReport,
    GdsReport,
    SaxReport,
    summarize_report,
)
from PhotonicsAI.graph.adapters.errors import (
    GdsBuildError,
    NetlistError,
    SaxModelMissingError,
)

__all__ = [
    "DRCExample",
    "DRCReport",
    "EdaReport",
    "GdsBuildError",
    "GdsReport",
    "NetlistError",
    "SaxModelMissingError",
    "SaxReport",
    "summarize_report",
]
