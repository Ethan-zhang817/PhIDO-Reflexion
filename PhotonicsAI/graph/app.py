"""Standalone Streamlit page for the PhIDO Reflexion PoC.

Run with::

    conda activate phido-reflexion
    cd /home/hesai/workspace/PhIDO-Reflexion
    export PYTHONPATH=.
    streamlit run PhotonicsAI/graph/app.py

The page is intentionally minimal: a prompt box, a model dropdown for
each agent, a Run button, then a streaming view of the LangGraph
updates. Token usage and timings come from ``state["token_usage"] /
state["timings"]`` (lazy — populated when implemented).
"""

from __future__ import annotations

import uuid

import streamlit as st
import yaml

from PhotonicsAI.graph.graph import get_compiled_graph
from PhotonicsAI.graph.state import (
    DEFAULT_DESIGNER_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REFLECTOR_MODEL,
    initial_state,
)


_DESIGNER_MODELS = [
    "o1",
    "gpt-4o",
    "claude-3-7-sonnet-20250219",
    "deepseek-reasoner",
    "gemini-2.5-pro",
]

_REFLECTOR_MODELS = [
    "claude-3-7-sonnet-20250219",
    "gpt-4o",
    "o1",
    "deepseek-reasoner",
]


@st.cache_resource(show_spinner=False)
def _compiled_graph():
    return get_compiled_graph()


def _ensure_thread_id() -> str:
    if "phido_graph_thread_id" not in st.session_state:
        st.session_state["phido_graph_thread_id"] = uuid.uuid4().hex
    return st.session_state["phido_graph_thread_id"]


def _format_state_panel(state_value):
    """Render an EdaReport / circuit_dsl in a way that is easy to scan."""
    if state_value is None:
        return "(no value)"
    if hasattr(state_value, "model_dump"):
        return yaml.dump(
            state_value.model_dump(), sort_keys=False, default_flow_style=False
        )
    if isinstance(state_value, dict):
        return yaml.dump(state_value, sort_keys=False, default_flow_style=False)
    return str(state_value)


def _run_graph(graph, init_state, *, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    progress = st.empty()
    for update in graph.stream(init_state, config=config, stream_mode="updates"):
        for node_name, node_state in update.items():
            with progress.container():
                st.markdown(f"**Node:** `{node_name}`")
                if "circuit_dsl" in node_state:
                    with st.expander("circuit_dsl", expanded=False):
                        st.code(_format_state_panel(node_state["circuit_dsl"]), language="yaml")
                if "eda_report" in node_state:
                    with st.expander("eda_report", expanded=True):
                        st.code(
                            _format_state_panel(node_state["eda_report"]),
                            language="yaml",
                        )
                if "reflections" in node_state and node_state["reflections"]:
                    with st.expander("reflection", expanded=True):
                        st.write(node_state["reflections"][-1])
    return graph.get_state(config)


def main() -> None:
    st.set_page_config(page_title="PhIDO Reflexion PoC", layout="wide")
    st.title("PhIDO Reflexion PoC")
    st.caption(
        "Experimental Designer ↔ Reflector loop — coexists with the legacy "
        "PhotonicsAI.Photon workflow. Safe to delete the `PhotonicsAI/graph` "
        "directory to roll back."
    )

    with st.sidebar:
        st.subheader("Configuration")
        designer_model = st.selectbox(
            "Designer model", _DESIGNER_MODELS, index=_DESIGNER_MODELS.index(DEFAULT_DESIGNER_MODEL)
        )
        reflector_model = st.selectbox(
            "Reflector model",
            _REFLECTOR_MODELS,
            index=_REFLECTOR_MODELS.index(DEFAULT_REFLECTOR_MODEL),
        )
        max_retries = st.number_input(
            "Max retries", min_value=0, max_value=10, value=DEFAULT_MAX_RETRIES
        )
        st.markdown("---")
        if st.button("Reset thread"):
            st.session_state.pop("phido_graph_thread_id", None)
            st.rerun()
        st.caption(
            f"Thread ID: `{st.session_state.get('phido_graph_thread_id', '(not yet allocated)')}`"
        )

    user_prompt = st.text_area(
        "Design intent",
        value="Design a 1x4 wavelength division demultiplexer at 1550 nm using cascaded MZIs.",
        height=120,
    )

    seed_dsl_text = st.text_area(
        "Optional seed circuit DSL (YAML, leave empty for from-scratch)",
        value="",
        height=160,
    )

    if st.button("Run Reflexion loop", type="primary"):
        thread_id = _ensure_thread_id()
        seed_dsl = None
        if seed_dsl_text.strip():
            try:
                seed_dsl = yaml.safe_load(seed_dsl_text)
                if not isinstance(seed_dsl, dict):
                    st.error("Seed DSL must be a YAML mapping.")
                    return
            except yaml.YAMLError as exc:
                st.error(f"Seed DSL is not valid YAML: {exc}")
                return

        graph = _compiled_graph()
        init_state = initial_state(
            user_prompt=user_prompt,
            seed_circuit_dsl=seed_dsl,
            designer_model=designer_model,
            reflector_model=reflector_model,
            max_retries=int(max_retries),
            thread_id=thread_id,
        )

        with st.status("Running Reflexion loop...", expanded=True):
            final = _run_graph(graph, init_state, thread_id=thread_id)
        st.success("Run complete.")
        st.subheader("Final state")
        if final and final.values:
            st.code(
                _format_state_panel(final.values.get("eda_report")),
                language="yaml",
            )
            st.code(
                _format_state_panel(final.values.get("circuit_dsl")),
                language="yaml",
            )


if __name__ == "__main__":
    main()
