# PhotonicsAI/graph — LangGraph + Reflexion PoC

This subpackage is a **side-by-side experiment** that introduces a
LangGraph-based Reflexion loop for the Layout → SAX → DRC stage of the
PhIDO workflow. The legacy Streamlit app under `PhotonicsAI/Photon/`
remains the production entry point and is **not modified**. To roll
back the experiment, simply delete this directory.

---

## What this module covers (and what it doesn't)

This implements **里程碑 A 的最小子集 + 里程碑 B 的 PoC** from
`REFACTOR_PLAN.md`:

- ✅ Structured DRC report (`adapters/drc_parser.py` parses KLayout
  `.lyrdb` XML into a `DRCReport`)
- ✅ Structured SAX adapter that raises `SaxModelMissingError` instead
  of printing
- ✅ Planarity adapter returning crossing edge pairs (not just a bool)
- ✅ Designer / EDA Executor / Evaluator / Reflector LangGraph nodes
  with a hard retry cap and human escalation
- ✅ SqliteSaver checkpointer (falls back to MemorySaver on failure)
- ✅ Optional LangSmith tracing (auto-enabled when `LANGSMITH_API_KEY`
  is present)
- ❌ Entity / Component / DSL / Schematic Reflexion loops (out of
  scope — left for a later milestone)
- ❌ Replacing the legacy `webapp.run_step_by_step_*` functions
- ❌ Rewriting `DemoPDK.yaml_netlist_to_gds` (we wrap it instead)

---

## Architecture

```
START → designer → eda_executor → evaluator
                                  ├─ pass ──→ END
                                  └─ fail ──→ reflector
                                              ├─ retry (<N) ──→ designer
                                              └─ escalate ────→ human_escalation
                                                                ├─ override / hint ──→ designer
                                                                └─ abort         ────→ END
```

**Key design rules** (from REFACTOR_PLAN §10):

| Decision | Choice |
|---|---|
| Reflector / Designer chat history | Separate (Reflector cannot see Designer's drafts in conversation form) |
| Reflection granularity | One critique per round, not per violation |
| Reflector model | `claude-3-7-sonnet-20250219` by default; configurable in the UI |
| LLM provider integration | Path B — wrap `llm_api.call_llm` in a `RunnableLambda`; no `ChatOpenAI` rewrite yet |
| Code organization | Side-by-side new package; the legacy Photon module is untouched |

---

## File layout

```
PhotonicsAI/graph/
  state.py             # PhIDOState TypedDict + initial_state()
  graph.py             # build_graph() / get_compiled_graph()
  llm_runnables.py     # invoke_llm() wrapping legacy llm_api.call_llm
  tracing.py           # optional LangSmith bootstrap
  app.py               # standalone Streamlit page
  adapters/
    eda_report.py      # EdaReport / GdsReport / SaxReport / DRCReport pydantic models
    errors.py          # SaxModelMissingError, NetlistError, GdsBuildError
    drc_parser.py      # KLayout .lyrdb XML -> DRCReport
    drc_adapter.py     # run_drc_structured() (subprocess + parse)
    sax_adapter.py     # run_sax_structured()
    gds_adapter.py     # build_gds_structured()
    planarity_adapter.py
  nodes/
    designer.py
    eda_executor.py
    evaluator.py
    reflector.py
    human_escalation.py
    _pdk_context.py    # cached cell list / one-liner docstrings
  prompts/
    designer.txt
    reflector.txt
```

The KLayout DRC script that emits XML lives next to the legacy one at
`PhotonicsAI/Photon/drc/drc_script_xml.drc`; the original
`drc_script.drc` is unchanged.

---

## Running

```bash
conda activate phido-reflexion
cd /home/hesai/workspace/PhIDO-Reflexion
export PYTHONPATH=.
streamlit run PhotonicsAI/graph/app.py
```

The page exposes:

- a free-form design intent box and an optional seed DSL,
- model selectors for the Designer and Reflector (defaults: `o1` /
  `claude-3-7-sonnet-20250219`),
- a max-retries dial,
- a streaming view of LangGraph node updates.

Checkpoints land at `build/phido_checkpoints.db`. To start a fresh
thread click "Reset thread" in the sidebar (this changes the
`thread_id` Streamlit holds in `session_state`).

---

## Testing

```bash
pytest tests/graph -v
```

Tests do **not** require KLayout / gdsfactory / sax / any LLM
provider — every adapter is exercised against fixtures and every node
exercises a mocked `invoke_llm` / `eda_executor`. The
`tests/graph/fixtures/drc_strict.drc` script can also be passed
explicitly to `run_drc_structured(drc_script=...)` for manual
verification once KLayout is installed.

---

## Deviations from `REFACTOR_PLAN.md`

| Plan reference | What we shipped | Why |
|---|---|---|
| §4.1 "聚合 5000 条 Si_width" | Top-k examples per rule (default k=3) | Simpler; covers the vast majority of real cases |
| §4.2 SaxModelMissingError | Exposed via both raise and `SaxReport.missing_models` | Lets the executor decide whether to swallow the error |
| §4.3 GDSFactory adapter | Single ``build_gds_structured`` covering both routing and layer-overflow | Plan only listed routing; layer overflow is the dominant failure in practice |
| §4.4 planarity | Returns `CrossingEdgePair` namedtuple list | More structured than the plan's tuple sketch |
| §8.5 LangSmith | Enabled only when `LANGSMITH_API_KEY` is set | Plan suggested "yes"; we keep it opt-in to avoid a hard dep |
| §9 milestone B "skip Schematic / Component reflexion" | Same — single `designer → eda_executor` cycle, no separate Schematic node | Stays within PoC scope |

---

## Rollback

This experiment introduces no behavioural changes to the legacy app.
To remove it:

```bash
rm -rf PhotonicsAI/graph tests/graph
git checkout -- requirements.txt pyproject.toml \
                PhotonicsAI/Photon/drc/drc_script_xml.drc
```
