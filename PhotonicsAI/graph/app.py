"""Streamlit UI for the PhIDO Reflexion PoC — fully-automatic flow.

Run with::

    conda activate phido-reflexion
    cd /home/hesai/workspace/PhIDO-Reflexion
    export PYTHONPATH=.
    streamlit run PhotonicsAI/graph/app.py

UX goal: keep the LangGraph backend but restore the *look* of the legacy
``PhotonicsAI/Photon/webapp.py`` automatic workflow — per-phase
progress cards, inline outputs, and the final GDSII layout + SAX sweep
shown directly on the page. Only the automatic flow is exposed; the
step-by-step UI has been removed.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

import streamlit as st
import yaml

from PhotonicsAI.graph.adapters.eda_report import EdaReport
from PhotonicsAI.graph.graph import get_compiled_graph
from PhotonicsAI.graph.model_choices import DESIGNER_MODEL_CHOICES, REFLECTOR_MODEL_CHOICES
from PhotonicsAI.graph.nodes.evaluator import evaluator_verdict
from PhotonicsAI.graph.state import (
    DEFAULT_DESIGNER_MODEL,
    DEFAULT_LEGACY_STAGE_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REFLECTOR_MODEL,
    initial_state,
)

def _select_index(choices: list[str], preferred: str) -> int:
    try:
        return choices.index(preferred)
    except ValueError:
        return 0


_EXAMPLE_PROMPTS = [
    "A 2x2 MZI",
    "A wavelength division demultiplexer",
    "A four channel WDM",
    "A low loss 1x2 power splitter connected to two GHz modulators each with a delta length of 100 um.",
    "Cascaded 2x2 MZIs to create a switch tree network with 8 outputs",
    "A 1x2 splitter connected to two amplitude modulators with 100 dB extinction ratio",
]

# Ordered legacy-style stage keys + human-readable labels used to drive the
# progress bar. Reflexion nodes are rendered inline after a failed
# Layout & Simulation attempt rather than as top-level stages.
_PHASE_ORDER = [
    ("entity_extraction", "Entity Extraction"),
    ("component_selection", "Component Selection"),
    ("schematic_generation", "Schematic Generation"),
    ("eda_executor", "Layout & Simulation"),
]


# =============================================================================
# STREAMLIT SETUP & HELPERS
# =============================================================================


@st.cache_resource(show_spinner=False)
def _compiled_graph():
    return get_compiled_graph()


def _ensure_thread_id() -> str:
    if "phido_graph_thread_id" not in st.session_state:
        st.session_state["phido_graph_thread_id"] = uuid.uuid4().hex
    return st.session_state["phido_graph_thread_id"]


def _yaml_block(value: Any) -> str:
    if value is None:
        return "(empty)"
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, (dict, list)):
        return yaml.safe_dump(value, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return str(value)


def _coerce_eda_report(obj: Any) -> Optional[EdaReport]:
    """Return an ``EdaReport`` whether ``obj`` is one already or a dict.

    LangGraph checkpoint round-trips may return a dict for custom types
    (the ``Deserializing unregistered type`` warning); we normalize here
    so the UI only has to handle a single shape.
    """
    if obj is None:
        return None
    if isinstance(obj, EdaReport):
        return obj
    if isinstance(obj, dict):
        try:
            return EdaReport.model_validate(obj)
        except Exception:  # noqa: BLE001
            return None
    return None


def _set_example(prompt: str) -> None:
    st.session_state["phido_user_prompt"] = prompt


# =============================================================================
# PHASE PANELS
# =============================================================================


def _render_designer_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
    *,
    attempt_index: int,
) -> None:
    """Render the circuit DSL revision produced after a failed attempt."""
    with container:
        st.markdown(f"##### Reflexion DSL Revision · for Attempt {attempt_index + 1}")
        dsl = state_delta.get("circuit_dsl")
        if dsl is None:
            st.warning("designer 未输出 circuit_dsl")
            return
        with st.expander("Revised Circuit DSL (YAML)", expanded=False):
            st.code(_yaml_block(dsl), language="yaml")


def _render_entity_extraction_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
) -> None:
    """Render Stage 1: Entity Extraction."""
    with container:
        st.markdown("##### Stage 1 · Entity Extraction")
        ee_result = state_delta.get("ee_result")
        if not ee_result:
            st.error("Entity Extraction 未生成结果。")
            return
        st.success("Entity Extraction completed.")
        with st.expander("Entity Extraction Output", expanded=True):
            st.code(_yaml_block(ee_result), language="yaml")


def _render_component_selection_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
) -> None:
    """Render Stage 2: Component Selection."""
    with container:
        st.markdown("##### Stage 2 · Component Selection")
        if state_delta.get("selected_components"):
            st.success("Component Selection completed.")
            st.markdown("**Selected components:**")
            for comp in state_delta["selected_components"]:
                st.write(f"- `{comp}`")
        else:
            st.error("Component Selection 未选出 PDK 组件。")
        if state_delta.get("selected_pretemplate"):
            with st.expander("Selected Pretemplate", expanded=False):
                st.code(_yaml_block(state_delta["selected_pretemplate"]), language="yaml")


def _render_schematic_generation_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
) -> None:
    """Render Stage 3: Schematic Generation."""
    with container:
        st.markdown("##### Stage 3 · Schematic Generation")
        dsl = state_delta.get("schematic_dsl") or state_delta.get("circuit_dsl")
        if not dsl:
            st.error("Schematic Generation 未生成 circuit DSL。")
            return
        st.success("Schematic Generation completed.")
        with st.expander("Circuit DSL", expanded=True):
            st.code(_yaml_block(dsl), language="yaml")
        if state_delta.get("legacy_debug"):
            with st.expander("Generated Schematic / DOT artifacts", expanded=False):
                st.code(_yaml_block(state_delta["legacy_debug"]), language="yaml")


def _render_eda_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
    *,
    attempt_index: int,
) -> None:
    """Render GDS layout image, SAX plot, DRC banner + structured report."""
    report = _coerce_eda_report(state_delta.get("eda_report"))

    with container:
        st.markdown(f"##### Stage 4 · Layout & Simulation · Attempt {attempt_index}")

        # --- Stage chips (GDS / SAX / DRC) ------------------------------
        if report is not None:
            gds = report.get_gds()
            sax = report.get_sax()
            drc = report.get_drc()
            chips = st.columns(3)
            with chips[0]:
                _stage_chip("GDS", gds.ok, gds.errors[:1])
            with chips[1]:
                ok = sax.ok and not sax.missing_models
                extra = sax.missing_models[:2] if sax.missing_models else sax.errors[:1]
                _stage_chip("SAX", ok, extra)
            with chips[2]:
                if drc.skipped_reason:
                    _stage_chip("DRC", None, [drc.skipped_reason])
                else:
                    _stage_chip(
                        "DRC",
                        drc.ok,
                        [f"{drc.n_violations} violations"] if not drc.ok else [],
                    )

        # --- GDS layout image ------------------------------------------
        gds_png = state_delta.get("gds_png_path")
        if gds_png and Path(gds_png).exists():
            if report is not None and report.get_gds().routing_failed:
                st.warning(
                    "当前 PNG 来自 `ignore_links=True` fallback：器件摆放可参考，"
                    "但 optical routes 未成功连线，不能视为已完成版图。"
                )
            st.caption("GDSII 版图")
            st.image(str(gds_png), use_container_width=True)
        elif report is not None and not report.get_gds().ok:
            st.error(
                "版图生成失败：" + (report.get_gds().errors[0] if report.get_gds().errors else "unknown")
            )

        if report is not None and report.get_gds().warnings:
            with st.expander("Normalizer 警告（Designer 输出被自动修补）", expanded=False):
                for w in report.get_gds().warnings:
                    st.caption(f"• {w}")

        # --- SAX sweep plot --------------------------------------------
        sax_png = state_delta.get("sax_png_path")
        if sax_png and Path(sax_png).exists():
            st.caption("SAX S 参数扫描（1500–1600 nm）")
            st.image(str(sax_png), use_container_width=True)
        elif report is not None and report.get_sax().missing_models:
            st.warning(
                "SAX 缺少模型："
                + ", ".join(report.get_sax().missing_models[:5])
            )

        # --- DRC banner -------------------------------------------------
        if report is not None:
            drc = report.get_drc()
            if drc.skipped_reason:
                st.info(f"DRC 跳过：{drc.skipped_reason}")
            elif drc.ok:
                st.success("✅ DRC 全通过（0 violations）")
            else:
                st.error(f"❌ DRC 检出 {drc.n_violations} 个违规")
                if drc.examples:
                    with st.expander("DRC 违规示例", expanded=False):
                        for ex in drc.examples[:5]:
                            st.write(f"- {ex.rule}  {ex.coord}")

        # --- Structured report YAML ------------------------------------
        if report is not None:
            with st.expander("结构化 EDA 报告 (YAML)", expanded=False):
                st.code(_yaml_block(report), language="yaml")

        # --- GDS path / download ---------------------------------------
        gds_path = state_delta.get("gds_path")
        if gds_path and Path(gds_path).exists():
            with open(gds_path, "rb") as fh:
                st.download_button(
                    "下载 GDSII 文件",
                    fh.read(),
                    file_name=Path(gds_path).name,
                    mime="application/octet-stream",
                )


def _render_evaluator_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
    *,
    attempt_index: int,
) -> None:
    report = _coerce_eda_report(state_delta.get("eda_report"))
    require_drc = state_delta.get("require_drc_pass", True)
    with container:
        st.markdown(f"##### Attempt {attempt_index} · Evaluation")
        if report is None:
            st.info("evaluator 执行完毕（无结构化报告）")
            return
        drc = report.get_drc()
        route = evaluator_verdict(report, require_drc_pass=require_drc)
        if route == "pass":
            if (not require_drc) and drc.n_violations > 0 and not drc.skipped_reason:
                st.success(
                    "GDS+SAX 已满足；按侧边栏「DRC 必过」为关，本轮不因此进入反思。DRC 见上方 EDA 区块。"
                )
            else:
                st.success("本轮通过，无需反思。")
        else:
            st.warning("本轮未通过，将交由反思器生成修正建议。")
            bits: list[str] = []
            if not report.get_gds().ok:
                bits.append("GDS")
            s = report.get_sax()
            if not s.ok or s.missing_models:
                bits.append("SAX")
            if require_drc and drc.n_violations and not drc.skipped_reason:
                bits.append(f"DRC（{drc.n_violations}）")
            if bits:
                st.caption("未通过项：" + "、".join(bits))
        if report.summary_text:
            st.code(report.summary_text, language="text")


def _render_reflector_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
    *,
    attempt_index: int,
) -> None:
    reflections = state_delta.get("reflections") or []
    with container:
        st.markdown(f"##### Attempt {attempt_index} Failed · Reflexion")
        if not reflections:
            st.info("无反思输出。")
            return
        # Show the reflection that was just appended (accumulated state
        # may also contain earlier rounds; the relevant one for this
        # panel is the last entry).
        latest = reflections[-1]
        st.write(latest)
        if len(reflections) > 1:
            with st.expander(
                f"历史反思（{len(reflections) - 1} 条）", expanded=False
            ):
                for idx, text in enumerate(reflections[:-1], start=1):
                    st.markdown(f"**#{idx}**")
                    st.write(text)


def _render_escalation_panel(
    container: st.delta_generator.DeltaGenerator,
    state_delta: dict,
) -> None:
    with container:
        st.markdown("##### 🚨 Human Escalation")
        st.warning(
            "已达到 max_retries。人工介入节点被触发，流程已结束。"
            "请查看最后一轮 EDA 报告与反思，再手工调整 seed DSL 重新提交。"
        )
        st.code(_yaml_block(state_delta), language="yaml")


def _stage_chip(
    label: str,
    ok: Optional[bool],
    extras: list[str],
) -> None:
    """Render a single stage status chip inside an EDA panel."""
    if ok is True:
        icon = "✅"
        color = "normal"
    elif ok is False:
        icon = "❌"
        color = "inverse"
    else:
        icon = "⚪"
        color = "off"
    st.metric(
        label=f"{icon} {label}",
        value="OK" if ok else ("SKIP" if ok is None else "FAIL"),
        delta=" ".join(extras) if extras else None,
        delta_color=color,
    )


# =============================================================================
# RUN LOOP
# =============================================================================


def _run_graph_streaming(
    graph,
    init_state,
    *,
    thread_id: str,
) -> None:
    """Stream node updates and lay them out as phase cards + progress bar.

    We also mirror the final state into ``st.session_state`` so that a
    rerun triggered by a Streamlit widget won't wipe the results.
    """
    config = {"configurable": {"thread_id": thread_id}}
    progress_bar = st.progress(0, text="准备运行 LangGraph Reflexion 流程…")
    status_box = st.status("执行中…", expanded=True, state="running")

    accumulated_state: dict[str, Any] = dict(init_state)
    attempt_index = 0
    _phase_weights = {name: i for i, (name, _) in enumerate(_PHASE_ORDER)}
    n_phases = len(_PHASE_ORDER)

    try:
        for update in graph.stream(init_state, config=config, stream_mode="updates"):
            for node_name, node_delta in update.items():
                # Merge delta so downstream panels always see a complete
                # view (e.g. evaluator panel needs eda_report written by
                # eda_executor). ``reflections`` uses LangGraph's ``add``
                # reducer at the checkpointer layer, so the streamed
                # delta only contains the newly-appended item; we mirror
                # that accumulation manually here so the UI sees the
                # full history across rounds.
                new_reflections = node_delta.get("reflections") or []
                prior_reflections = list(accumulated_state.get("reflections") or [])
                accumulated_state.update(node_delta)
                if new_reflections:
                    accumulated_state["reflections"] = (
                        prior_reflections + list(new_reflections)
                    )

                if node_name == "eda_executor":
                    attempt_index += 1

                idx = _phase_weights.get(node_name, n_phases - 1)
                pct = int(100 * (idx + 1) / n_phases)
                label = next((lbl for n, lbl in _PHASE_ORDER if n == node_name), node_name)
                if node_name == "eda_executor":
                    label = f"{label} · Attempt {attempt_index}"
                elif node_name == "reflector":
                    label = f"Reflexion after Attempt {attempt_index}"
                elif node_name == "designer":
                    label = f"Revise DSL for Attempt {attempt_index + 1}"
                progress_bar.progress(pct, text=label)

                # Dispatch to the appropriate panel renderer.
                panel = status_box.container()
                if node_name == "entity_extraction":
                    _render_entity_extraction_panel(panel, node_delta)
                elif node_name == "component_selection":
                    _render_component_selection_panel(panel, node_delta)
                elif node_name == "schematic_generation":
                    _render_schematic_generation_panel(panel, node_delta)
                elif node_name == "designer":
                    _render_designer_panel(panel, node_delta, attempt_index=attempt_index)
                elif node_name == "eda_executor":
                    _render_eda_panel(panel, accumulated_state, attempt_index=attempt_index)
                elif node_name == "evaluator":
                    _render_evaluator_panel(panel, accumulated_state, attempt_index=attempt_index)
                elif node_name == "reflector":
                    _render_reflector_panel(panel, accumulated_state, attempt_index=attempt_index)
                elif node_name == "human_escalation":
                    _render_escalation_panel(panel, node_delta)
                else:
                    with panel:
                        st.markdown(f"##### ⏺️ `{node_name}`")
                        st.code(_yaml_block(node_delta), language="yaml")

    except Exception as exc:  # noqa: BLE001
        progress_bar.progress(100, text="运行失败")
        status_box.update(label="运行失败", state="error", expanded=True)
        status_box.exception(exc)
        if _coerce_eda_report(accumulated_state.get("eda_report")) is not None:
            crash_panel = status_box.container()
            _render_eda_panel(
                crash_panel, accumulated_state, attempt_index=max(attempt_index, 1)
            )
        st.session_state["phido_last_run"] = accumulated_state
        return

    progress_bar.progress(100, text="完成")
    status_box.update(label="运行完成", state="complete", expanded=True)

    # Pull the authoritative final state from the checkpointer so the
    # result survives any retries triggered by reflexion.
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            accumulated_state.update(snapshot.values)
    except Exception:  # noqa: BLE001
        pass

    st.session_state["phido_last_run"] = accumulated_state


def _render_final_summary(state: dict[str, Any]) -> None:
    """Big, webapp-style final card: GDS image + SAX plot + DRC banner."""
    st.markdown("---")
    st.markdown("### 最终结果")

    report = _coerce_eda_report(state.get("eda_report"))
    gds_png = state.get("gds_png_path")
    sax_png = state.get("sax_png_path")
    gds_path = state.get("gds_path")

    cols = st.columns([3, 2])
    with cols[0]:
        st.markdown("**GDSII 版图**")
        if gds_png and Path(gds_png).exists():
            if report is not None and report.get_gds().routing_failed:
                st.warning("该版图 PNG 是未连线 fallback 预览，routing 仍失败。")
            st.image(str(gds_png), use_container_width=True)
        else:
            st.info("本轮未生成版图 PNG。")
    with cols[1]:
        st.markdown("**SAX S 参数**")
        if sax_png and Path(sax_png).exists():
            st.image(str(sax_png), use_container_width=True)
        else:
            st.info("本轮未生成 SAX 仿真图。")

    if report is not None:
        drc = report.get_drc()
        if (not state.get("require_drc_pass", True)) and drc.n_violations and not drc.skipped_reason:
            st.info(
                "当前运行关闭了「DRC 零违规才算通过」：流程可在有 DRC 报告时仍结束，下方为参考。"
            )
        if drc.ok and not drc.skipped_reason:
            st.success("✅ DRC 全通过")
        elif drc.skipped_reason:
            st.info(f"DRC 跳过：{drc.skipped_reason}")
        else:
            st.error(f"❌ DRC 检出 {drc.n_violations} 处违规")

        with st.expander("完整 EDA 报告 (YAML)", expanded=False):
            st.code(_yaml_block(report), language="yaml")

    if state.get("reflections"):
        with st.expander("历次反思（Reflections）", expanded=False):
            for i, r in enumerate(state["reflections"], 1):
                st.markdown(f"**#{i}**")
                st.write(r)

    if gds_path and Path(gds_path).exists():
        with open(gds_path, "rb") as fh:
            st.download_button(
                "下载 GDSII",
                fh.read(),
                file_name=Path(gds_path).name,
                mime="application/octet-stream",
                key="final_gds_download",
            )


# =============================================================================
# PAGE LAYOUT
# =============================================================================


def main() -> None:
    st.set_page_config(page_title="PhIDO ⚡ Reflexion", layout="wide")
    st.markdown("#### PhIDO ⚡")
    st.markdown("PHotonic Intelligent Design & Optimization · Reflexion 全自动流程")

    with st.sidebar:
        st.subheader("Configuration")
        st.caption(
            "OpenAI 与兼容代理（在 `.env` 设置 `OPENAI_BASE_URL` / `OPENAI_API_KEY`；"
            "例如聚合服务 `https://api.gptsapi.net/v1` ）请在下拉里选用上游实际提供的模型名。"
        )
        designer_model = st.selectbox(
            "Designer model",
            DESIGNER_MODEL_CHOICES,
            index=_select_index(DESIGNER_MODEL_CHOICES, DEFAULT_DESIGNER_MODEL),
        )
        legacy_stage_model = st.selectbox(
            "Legacy stage model",
            DESIGNER_MODEL_CHOICES,
            index=_select_index(DESIGNER_MODEL_CHOICES, DEFAULT_LEGACY_STAGE_MODEL),
            help="用于原版多阶段链路（EE / 元件选型 / 原理图与 DOT），与 Reflexion 的 Designer/Reflector 独立。",
        )
        reflector_model = st.selectbox(
            "Reflector model",
            REFLECTOR_MODEL_CHOICES,
            index=_select_index(REFLECTOR_MODEL_CHOICES, DEFAULT_REFLECTOR_MODEL),
        )
        max_retries = st.number_input(
            "Max retries", min_value=0, max_value=10, value=DEFAULT_MAX_RETRIES
        )
        require_drc_pass = st.checkbox(
            "DRC 零违规才算通过",
            value=True,
            help="关闭时：GDS+SAX 无致命错误则评估为通过并结束，不进入反思；DRC 仍会跑并展示。",
        )
        st.markdown("---")
        if st.button("重置会话（新建 thread）"):
            st.session_state.pop("phido_graph_thread_id", None)
            st.session_state.pop("phido_last_run", None)
            st.rerun()
        st.caption(
            f"Thread ID: `{st.session_state.get('phido_graph_thread_id', '(尚未分配)')}`"
        )

    # --- Prompt & optional seed DSL ---------------------------------------
    user_prompt = st.text_input(
        "⇨ 请用自然语言描述想要设计的光子电路：",
        key="phido_user_prompt",
        placeholder="例如：A 1x4 wavelength division demultiplexer using cascaded MZIs",
    )

    with st.expander("示例（点击即可填入）", expanded=not bool(user_prompt)):
        cols = st.columns(2)
        for i, ex in enumerate(_EXAMPLE_PROMPTS):
            with cols[i % 2]:
                st.button(
                    ex,
                    key=f"ex_btn_{i}",
                    use_container_width=True,
                    on_click=_set_example,
                    args=(ex,),
                )

    with st.expander("高级：提供 seed 电路 DSL（可选）", expanded=False):
        seed_dsl_text = st.text_area(
            "YAML（留空表示从零开始）",
            value="",
            height=160,
            key="phido_seed_dsl",
        )

    run_clicked = st.button("🚀 开始自动设计", type="primary", disabled=not user_prompt.strip())

    if run_clicked:
        thread_id = _ensure_thread_id()
        seed_dsl: Optional[dict] = None
        if seed_dsl_text.strip():
            try:
                parsed = yaml.safe_load(seed_dsl_text)
            except yaml.YAMLError as exc:
                st.error(f"Seed DSL 不是合法 YAML：{exc}")
                return
            if not isinstance(parsed, dict):
                st.error("Seed DSL 必须是一个 YAML mapping。")
                return
            seed_dsl = parsed

        graph = _compiled_graph()
        init = initial_state(
            user_prompt=user_prompt,
            seed_circuit_dsl=seed_dsl,
            designer_model=designer_model,
            legacy_stage_model=legacy_stage_model,
            reflector_model=reflector_model,
            max_retries=int(max_retries),
            require_drc_pass=bool(require_drc_pass),
            thread_id=thread_id,
        )

        _run_graph_streaming(graph, init, thread_id=thread_id)

    last_run = st.session_state.get("phido_last_run")
    if last_run:
        _render_final_summary(last_run)


if __name__ == "__main__":
    main()
