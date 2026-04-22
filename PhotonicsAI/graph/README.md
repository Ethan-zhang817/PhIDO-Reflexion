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

## API keys

All keys live in `.env` at the repo root (loaded by
`PhotonicsAI/config.py`). Copy `.env.example` and fill what you need:

| Variable | Purpose | Required? |
|---|---|---|
| `OPENAI_API_KEY` | GPT / o-series / pydantic parse; also reused for Claude when a proxy is set | Always |
| `OPENAI_BASE_URL` | OpenAI-compatible proxy URL (e.g. `https://api.gptsapi.net/v1`). Empty means call api.openai.com directly | Optional |
| `ANTHROPIC_API_KEY` | Required only when calling api.anthropic.com directly. Ignored if `OPENAI_BASE_URL` already proxies Claude | Optional |
| `GOOGLEGENAI_API_KEY` | Native `google.generativeai` SDK; not affected by `OPENAI_BASE_URL` | Optional |
| `DEEPSEEK_API_KEY` | Calls api.deepseek.com directly; not affected by `OPENAI_BASE_URL` | Optional |
| `NVIDIA_API_KEY` | NVIDIA NIM endpoint | Optional |
| `LANGSMITH_API_KEY` | When set, `tracing.maybe_enable_langsmith()` enables LangChain tracing v2 | Optional |
| `LANGCHAIN_PROJECT` | LangSmith project name (default `phido-reflexion`) | Optional |

### Third-party OpenAI-compatible gateways (e.g. GPTsAPI)

Setting `OPENAI_BASE_URL` reroutes **every** OpenAI / Claude call:

- `call_openai`, `call_openai_reasoning`, `callgpt_pydantic` use the
  proxy via the standard OpenAI SDK `base_url` parameter.
- `call_anthropic` detects the proxy and switches from
  `anthropic.Anthropic(...).messages.create` to OpenAI-compat
  `chat.completions.create` (model id stays `claude-3-7-sonnet-20250219`).
  **Caveat**: extended-thinking blocks are not exposed by these
  gateways, so the thinking budget is dropped on the proxy path. Leave
  `OPENAI_BASE_URL` empty to keep the native Anthropic path with
  thinking.

Gemini and DeepSeek keep their native SDKs and are **not** rerouted by
`OPENAI_BASE_URL` — point those keys at their respective services.

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

---
---

# 中文说明

## 模块定位

`PhotonicsAI/graph/` 是一个**与旧代码并存的实验性子包**，在 PhIDO 的
Layout → SAX → DRC 环节引入基于 LangGraph 的 Reflexion 闭环。
`PhotonicsAI/Photon/` 下的旧 Streamlit 应用仍然是生产入口，**一行未改**。
要回滚本次实验，删除本目录即可。

---

## 本模块实现了什么、没实现什么

对应 `REFACTOR_PLAN.md` 中的 **里程碑 A 最小子集 + 里程碑 B 的 PoC**：

- ✅ 结构化 DRC 报告（`adapters/drc_parser.py` 把 KLayout `.lyrdb` XML
  解析成 `DRCReport`）
- ✅ 结构化 SAX 适配器：抛出 `SaxModelMissingError` 代替原来的
  `print`
- ✅ 平面性适配器返回具体的交叉边对，而不仅仅是一个布尔值
- ✅ Designer / EDA Executor / Evaluator / Reflector 四个 LangGraph
  节点，带最大重试次数与人工升级
- ✅ `SqliteSaver` 检查点（失败时自动退化为 `MemorySaver`）
- ✅ 可选 LangSmith Tracing（仅在检测到 `LANGSMITH_API_KEY` 时启用）
- ❌ 实体 / 元件 / DSL / 原理图层级的 Reflexion 闭环（留给后续里程碑）
- ❌ 替换旧 `webapp.run_step_by_step_*`
- ❌ 重写 `DemoPDK.yaml_netlist_to_gds`（我们只做了一层封装）

---

## 架构

```
START → designer → eda_executor → evaluator
                                  ├─ pass ──→ END
                                  └─ fail ──→ reflector
                                              ├─ retry (<N) ──→ designer
                                              └─ escalate ────→ human_escalation
                                                                ├─ override / hint ──→ designer
                                                                └─ abort         ────→ END
```

**关键设计决策**（来自 REFACTOR_PLAN §10）：

| 决策点 | 选择 |
|---|---|
| Reflector / Designer 对话历史 | 各自独立，Reflector 看不到 Designer 的草稿对话 |
| 反思粒度 | 每轮一条整体建议，而不是每条违规一条建议 |
| Reflector 默认模型 | `claude-3-7-sonnet-20250219`，UI 可切换 |
| LLM Provider 接入方式 | 路 B：用 `RunnableLambda` 包装 `llm_api.call_llm`，暂不改写成 `ChatOpenAI` |
| 代码组织 | 新包并存，旧 Photon 模块保持不动 |

---

## 目录结构

```
PhotonicsAI/graph/
  state.py             # PhIDOState TypedDict + initial_state()
  graph.py             # build_graph() / get_compiled_graph()
  llm_runnables.py     # invoke_llm()，包装旧的 llm_api.call_llm
  tracing.py           # 可选的 LangSmith 启动入口
  app.py               # 独立的 Streamlit 演示页
  adapters/
    eda_report.py      # EdaReport / GdsReport / SaxReport / DRCReport pydantic 模型
    errors.py          # SaxModelMissingError、NetlistError、GdsBuildError
    drc_parser.py      # KLayout .lyrdb XML -> DRCReport
    drc_adapter.py     # run_drc_structured()（subprocess + 解析）
    sax_adapter.py     # run_sax_structured()
    gds_adapter.py     # build_gds_structured()
    planarity_adapter.py
  nodes/
    designer.py
    eda_executor.py
    evaluator.py
    reflector.py
    human_escalation.py
    _pdk_context.py    # 缓存的元件名清单 + 一行 docstring
  prompts/
    designer.txt
    reflector.txt
```

产出 XML 的 KLayout DRC 脚本放在旧脚本旁边：
`PhotonicsAI/Photon/drc/drc_script_xml.drc`；原始的 `drc_script.drc`
**未做任何改动**。

---

## API Key 配置

所有 key 都放在仓库根目录的 `.env` 中，由 `PhotonicsAI/config.py` 自动
加载。复制一份 `.env.example` 然后按需填写：

| 变量 | 用途 | 是否必填 |
|---|---|---|
| `OPENAI_API_KEY` | GPT / o 系列 / pydantic parse；设了代理后也会被复用给 Claude | 必填 |
| `OPENAI_BASE_URL` | OpenAI 兼容代理地址（例如 `https://api.gptsapi.net/v1`）；留空则直连 api.openai.com | 选填 |
| `ANTHROPIC_API_KEY` | 只有直连 api.anthropic.com 时才需要；当 `OPENAI_BASE_URL` 已经代理 Claude 时可留空 | 选填 |
| `GOOGLEGENAI_API_KEY` | 走原生 `google.generativeai` SDK，不受 `OPENAI_BASE_URL` 影响 | 选填 |
| `DEEPSEEK_API_KEY` | 直连 api.deepseek.com，不受 `OPENAI_BASE_URL` 影响 | 选填 |
| `NVIDIA_API_KEY` | NVIDIA NIM 接口 | 选填 |
| `LANGSMITH_API_KEY` | 设置后 `tracing.maybe_enable_langsmith()` 会启用 LangChain v2 tracing | 选填 |
| `LANGCHAIN_PROJECT` | LangSmith 项目名（默认 `phido-reflexion`） | 选填 |

### 第三方 OpenAI 兼容网关（例如 GPTsAPI）

设置 `OPENAI_BASE_URL` 会把**所有 OpenAI / Claude 调用**都路由到代理：

- `call_openai`、`call_openai_reasoning`、`callgpt_pydantic` 直接通过
  OpenAI SDK 的 `base_url` 参数走代理；
- `call_anthropic` 检测到代理时会自动从
  `anthropic.Anthropic(...).messages.create` 切到 OpenAI 兼容的
  `chat.completions.create`（model id 仍然写
  `claude-3-7-sonnet-20250219`）。**注意**：第三方网关一般不暴露 Claude
  的 extended-thinking 块，所以代理路径会丢掉 thinking budget；如果你
  希望保留 thinking，把 `OPENAI_BASE_URL` 留空走原生 Anthropic SDK。

Gemini 和 DeepSeek **不走** `OPENAI_BASE_URL`，仍然使用各自的官方 SDK
和 key，需要单独配置。

---

## 运行方式

```bash
conda activate phido-reflexion
cd /home/hesai/workspace/PhIDO-Reflexion
export PYTHONPATH=.
streamlit run PhotonicsAI/graph/app.py
```

页面提供：

- 一个自由文本的设计意图输入框和可选的种子 DSL；
- Designer / Reflector 的模型下拉（默认 `o1` / `claude-3-7-sonnet-20250219`）；
- 最大重试次数调节；
- LangGraph 节点更新的流式视图。

Checkpoint 存在 `build/phido_checkpoints.db`。想开新会话点侧栏的
"Reset thread" 按钮即可（会刷新 `st.session_state` 里的
`thread_id`）。

---

## 测试

```bash
pytest tests/graph -v
```

测试**不依赖** KLayout / gdsfactory / sax / 任何 LLM 提供商：每个
adapter 都针对 fixtures 跑，每个节点都对 `invoke_llm` / `eda_executor`
做了 mock。如果本机装了 KLayout，可以手动把
`tests/graph/fixtures/drc_strict.drc` 传给
`run_drc_structured(drc_script=...)` 做端到端验证。

---

## 与 `REFACTOR_PLAN.md` 的偏差

| 方案条目 | 实际落地 | 原因 |
|---|---|---|
| §4.1 「聚合 5000 条 Si_width」 | 每条规则最多保留 k 个示例（默认 k=3） | 更简单，真实场景已经足够 |
| §4.2 SaxModelMissingError | 既 raise 也暴露在 `SaxReport.missing_models` 里 | 让上层 executor 自行决定吞不吞 |
| §4.3 GDSFactory 适配器 | 单个 `build_gds_structured` 覆盖路由错误和图层溢出 | 方案只列了路由，而图层溢出才是实际高发故障 |
| §4.4 平面性 | 返回 `CrossingEdgePair` namedtuple 列表 | 比方案里的裸 tuple 更结构化 |
| §8.5 LangSmith | 仅当检测到 `LANGSMITH_API_KEY` 时才启用 | 方案写的是"接入"，我们做成可选，避免硬依赖 |
| §9 里程碑 B「跳过 Schematic / Component reflexion」 | 只做单个 `designer → eda_executor` 循环，不拆 Schematic 节点 | 保持在 PoC 范围内 |

---

## 回滚

本实验对旧 app 不引入任何行为变更。移除方式：

```bash
rm -rf PhotonicsAI/graph tests/graph
git checkout -- requirements.txt pyproject.toml \
                PhotonicsAI/Photon/drc/drc_script_xml.drc
```
