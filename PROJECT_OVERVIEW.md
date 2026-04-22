# PhIDO-Reflexion 项目详解

> 本文档基于仓库实际代码，对 **PhIDO-Reflexion** 项目做一次系统性的讲解，重点说明代码结构与各模块职责。

---

## 一、项目定位

**PhIDO** = **Photonic Intelligent Design and Optimization**（光子智能设计与优化），是一个用 **大语言模型（LLM）** 自动化设计 **光子集成电路（PIC, Photonic Integrated Circuits）** 的 Streamlit Web 应用。

典型流程：

> 用户用自然语言描述一个光路 → LLM 分步推理 → 生成电路 DSL → 生成 GDS 版图 + SAX 仿真 → KLayout DRC 验证

仓库里还包含作者的 arXiv 论文链接（<https://arxiv.org/abs/2508.14123>），这是一篇学术项目的配套代码。

---

## 二、顶层目录结构

```text
PhIDO-Reflexion/
├── PhotonicsAI/                       # 主 Python 包（所有代码都在这里）
│   ├── __init__.py
│   ├── config.py                      # 路径 & 环境变量配置
│   ├── Photon/                        # 应用核心
│   └── KnowledgeBase/                 # 元器件库 + FDTD 仿真数据
├── GETTING_STARTED.md                 # 详细使用教程
├── GETTING_STARTED_EXAMPLE_OUTPUTS/   # 4 个难度级别的示例输出
│   ├── Level 1 Prompt/ … Level 4 Prompt/
├── README.md                          # 项目总览 + 安装说明
├── CHANGELOG.md                       # 目前是空的
├── LICENSE                            # MIT
├── Makefile                           # install / dev / run / test / build 等目标
├── pyproject.toml                     # 包元信息 + ruff/mypy/pytest 配置
├── requirements.txt                   # pip 依赖
├── uv.lock                            # uv 的锁文件
├── Testbench.xlsx                     # 102 条测试 prompt
└── .pre-commit-config.yaml
```

### `Makefile` 核心目标

| 目标 | 作用 |
|---|---|
| `make install` | 用 `uv` 创建 venv 并同步依赖 |
| `make dev` | 以可编辑模式安装 dev + docs 额外依赖 |
| `make run` | `streamlit run PhotonicsAI/Photon/webapp.py` |
| `make test` | `pytest -s` |
| `make build` | 重建 `dist/` 并调用 `python -m build` |
| `make docs` | `jupyter-book build docs` |

---

## 三、`PhotonicsAI/config.py`：全局配置

非常轻量，只做两件事：

1. 用 `pydantic_settings` + `python-dotenv` 读取 `.env`（主要读 `openai_api_key`，其它 API key 在 `llm_api.py` 里按需读）。
2. 定义 `Path` 类集中所有相对路径：

```python
class Path:
    module = module_path
    repo = repo_path
    cells = module / "cells"
    photon = module / "Photon"
    prompts = photon / "prompts.yaml"
    templates = photon / "templates.yaml"
    logs = module / "log"
    pdk = module / "KnowledgeBase" / "DesignLibrary"
    build = repo / "build"
```

这样代码里可以直接写 `PATH.prompts`、`PATH.pdk` 等，不必到处拼路径。

---

## 四、`PhotonicsAI/Photon/`：应用核心

```text
Photon/
├── webapp.py              # Streamlit UI + 工作流编排（2753 行，最大文件）
├── llm_api.py             # 多 LLM 提供商统一封装（1460 行）
├── utils.py               # 电路 DSL ↔ DOT ↔ GDSFactory 转换等工具
├── DemoPDK.py             # 用 gdsfactory 聚合 KnowledgeBase 里的器件，构造 PDK
├── prompts.yaml           # 所有 LLM 提示词模板
├── templates.yaml         # 预定义电路模板（CIRCUIT_* / TEMPLATE_*）
├── CIRCUIT_wdd0.yaml      # 示例电路：1×4 WDM
├── icon.png
└── drc/
    ├── drc.py             # 调 KLayout 做 DRC
    ├── drc_script.drc     # KLayout 的 DRC 规则脚本
    └── report.lydrb       # DRC 报告
```

### 1. `webapp.py` —— 入口 & 工作流编排

文件开头定义每一步使用的 LLM（默认都是 `o1`）：

```python
entity_extraction_model = "o1"
component_selection_model = "o1"
component_specification_model = "o1"
schematic_model = "o1"
layout_model = "o1"
```

启动时会：

- 加载 `prompts.yaml`
- 扫描 `KnowledgeBase/DesignLibrary` 收集所有器件的 docstring
- 加载 `templates.yaml`

#### 两种工作流

1. **Automatic Workflow**（自动引导模式）
2. **Step-by-Step Workflow**（分步独立执行）

两种模式都走同一套 **5 阶段管线**，对应 `webapp.py` 中的一组 `run_step_by_step_*` 函数：

| 阶段 | 函数 | 作用 |
|---|---|---|
| ① Entity Extraction | `run_step_by_step_entity_extraction` | 自然语言 → 结构化 `pretemplate`（组件清单 + 连接意图） |
| ② Component Specification | `run_step_by_step_component_specification` | 为每个"抽象元器件"在 PDK 中检索具体实现 |
| ③ Circuit DSL Creation | `run_step_by_step_circuit_dsl_creation` | 生成电路 DSL（nodes + edges + properties） |
| ④ Schematic Generation | `run_step_by_step_schematic_generation` | 生成 DOT 图，校验无交叉 |
| ⑤ Layout & Simulation | `run_step_by_step_layout_simulation` | 生成 GDS + SAX 仿真 + 可选 DRC |

同时还实现了：

- token 用量跟踪
- session log 文件
- 组件检索结果的单选 radio 交互
- 模板 vs 定制两种路径
- 三段 `# ===` 分区（配置 / 数据加载 / 工作流函数 / UI 渲染）

### 2. `llm_api.py` —— LLM 统一封装层

支持 5 家厂商，函数签名基本一致，方便 `webapp.py` 按 model 名字切换：

| 厂商 | 主要函数 | 支持模型 |
|---|---|---|
| OpenAI | `call_openai`, `call_openai_reasoning` | GPT-4o、o1、o3-mini |
| Anthropic | `call_anthropic` | Claude 3.7 Sonnet、Claude 4 Opus（带 thinking 块调试） |
| Google | `call_google` | Gemini 2.5 Pro / 1.5 Pro / 1.5 Flash / 2.0 Flash |
| NVIDIA NIM | `call_nvidia` | Nemotron 系列 |
| DeepSeek | `call_deepseek` | DeepSeek-Reasoner（含 `count_deepseek_tokens`、`truncate_prompt`） |

顶层调度器：

```python
def call_llm(prompt, sys_prompt, llm_api_selection="nvidia/nemotron-4-340b-instruct"):
    ...
```

根据 `llm_api_selection` 字符串路由到具体厂商函数。

还有一组"业务层"的 LLM 封装（复用 `prompts.yaml`）：

- `intent_classification` / `verify_input_clarity` / `entity_extraction` / `papers_entity_extraction`
- `preschematic`、`parse_user_specs`、`apply_settings`
- `dot_add_edges` / `dot_add_edges_errorfunc` / `dot_add_edges_templates` / `dot_verify`
- `netlist_cleanup`
- `llm_retrieve` / `llm_search`（BM25 / 语义检索的组件查找）
- Pydantic 结构化输出：`callgpt_pydantic`、`calldeepseek_pydantic`、`callgoogle_pydantic`

Token 统计用 Streamlit 的 `session_state` 做单用户隔离（`get_session_token_usage` / `add_token_usage`）。

> ⚠️ README 特别提醒：即使你用别家模型，**也必须配 `OPENAI_API_KEY`**，因为 Pydantic 结构化输出仍然靠 GPT。

### 3. `utils.py` —— 电路 DSL / 图 / 版图之间的桥梁

提供以下关键转换（在 `webapp.py` 工作流中反复调用）：

| 函数 | 作用 |
|---|---|
| `search_directory_for_docstrings(directory=PATH.pdk)` | 扫描 `DesignLibrary/*.py`，提取 module-level docstring（里面嵌了 YAML 元数据：端口、规格、参数等），是"组件发现"的基础 |
| `circuit_to_dot(circuit_dsl)` | 电路 DSL → Graphviz DOT 图（带 record-shape 节点和端口） |
| `edges_dot_to_yaml(session)` | LLM 生成的 DOT 边写回 DSL |
| `dsl_to_gf(circuit_dsl)` | DSL → GDSFactory netlist（`instances` / `routes.optical.links` / `placements` / `ports`） |
| `get_graphviz_placements(dot_string)` | 用 pygraphviz 做 dot 布局，得到节点坐标（注意 Graphviz 返回中心点，GDSFactory 用角点，代码里有坐标修正） |
| `add_final_ports(session)` | 扫描 DOT 找"未连接端口"作为电路对外端口 |
| `dot_planarity(dot_string)` / `dot_crossing_edges(session)` | 粗略平面性检测（防止 LLM 生成互相交叉的连线） |
| `model_from_npz(...)` | 从 `.npz` 文件中读取 S 参数，构造 JAX-jit 的 SAX 模型——FDTD 数据进仿真的入口 |

### 4. `DemoPDK.py` —— 动态构造 PDK

**关键思想**：把 `KnowledgeBase/DesignLibrary/` 里的每个 `.py` 当成一个器件 cell，用 `importlib` 动态导入，再喂给 `gdsfactory` 组成 PDK：

```python
module_names = list_python_files(PATH.pdk)
cells = import_modules(module_names)
all_models = import_models(module_names)

DemoPDK = gf.Pdk(
    name="DemoPDK",
    layers=LAYER,
    cross_sections=cross_sections,
    cells=cells,
    layer_views=layer_views,
)
DemoPDK.activate()
```

同时导出工作流用的核心函数：

- `yaml_netlist_to_gds(session)`：把电路 DSL 转成的 GF netlist → GDS + SAX 电路仿真 + `plot_gds.png`
- `footprint_netlist`, `get_params`, `get_ports_info`：从 PDK 和 docstring 补全节点的 `dx/dy`、参数默认值和端口信息
- `circuit_optimizer`：用 **Bayesian Optimization** 调整自由参数（比如 `delta_length`、`voltage`），目标函数是 SAX 输出 S 参数与设计目标（`cosine` 传递函数）的 MSE。**这是 "Optimization" 的落脚点**。

### 5. `prompts.yaml` —— 提示词库

用 YAML 键名组织不同任务的系统提示词，至少包含：

| 键名 | 作用 |
|---|---|
| `edges_yaml_to_dot` | 让 LLM 给现有节点添加不交叉的边（含 one-shot 示例） |
| `dot_verify` | 清理 DOT 图，保证两节点只有一条边 |
| `yaml_syntax_cleaner` | 把 LLM 产出的乱糟糟 YAML 清洗成合法 netlist |
| `dot_simple` | 从 pretemplate 直接生成 DOT |
| `absorb_settings` | 把用户描述里的"length=40um"等规格吸收进 netlist 的 `settings` |

文件里还存了大量**被注释掉的历史版本 prompt**（以 `#` 开头），起到版本留档作用。

### 6. `templates.yaml` —— 预定义电路

两类键名：

- **`TEMPLATE_*`**：**抽象模板**，节点是"1×2 MZI"这种描述，`edges` 是文字指示；用于 LLM 理解
  - 例：`TEMPLATE_wdd0`、`TEMPLATE_mzi_2x2_heater_tin_cband`、`TEMPLATE_transceiver0`、`TEMPLATE_IQmodulator`
- **`CIRCUIT_*`**：**具体电路 DSL**，节点绑定到真实 PDK 器件（如 `component: mzi_1x2_pindiode_cband`），带参数、端口、placement、specs、optimizer 配置

例：`CIRCUIT_wdd0`（与 `Photon/CIRCUIT_wdd0.yaml` 基本等价）是 3 个 `mzi_1x2_pindiode_cband` 级联，形成 1×4 波分解复用器。

### 7. `drc/` —— Design Rule Check

- `drc.py`（仅 66 行）：
  1. 查找 `klayout` 可执行文件路径
  2. 以 batch 模式 `-b -r drc_script.drc` 运行脚本
  3. 把 GDS 传进去，把报告写到 `report.lydrb`
- `drc_script.drc`：KLayout 的 DRC 规则脚本（最小线宽、间距等，来自工艺工程师在 PDK 里的定义）

---

## 五、`PhotonicsAI/KnowledgeBase/`：元器件库

```text
KnowledgeBase/
├── __init__.py
├── DesignLibrary/    # 每个器件一个 .py（约 35 个）
│   ├── bend_euler.py
│   ├── straight.py
│   ├── _mmi1x2.py  _mmi2x2.py
│   ├── _gc.py                      # grating coupler（前缀 _ 表低层原语）
│   ├── mrr_1x1.py / mrr_2x2.py / mrr_1x1_heater_tin.py
│   ├── mrm_1x1_pndiode.py
│   ├── mzi1.py / mzi_arm.py
│   ├── mzi_1x1_heater_doped_si_cband.py
│   ├── mzi_1x1_pindiode_cband.py / mzi_1x2_pindiode_cband.py
│   ├── mzi_2x2_heater_tin_cband.py / mzi_2x2_pindiode_cband.py / mzi_2x2_pn_diode.py
│   ├── heater_tin_cband.py / heater_doped_si_cband.py
│   ├── pindiode_cband.py / pndiode.py / photodetector.py
│   ├── polarization_splitter_rotator.py
│   ├── mode_converter.py / crossing.py
│   ├── edge_coupler.py / laser.py / tw_mzm.py
│   ├── wdm_mzi1x4.py
│   └── ...
└── FDTD/             # 预先跑好的 FDTD S 参数
    ├── bend_euler/ bezier_curve/ mmi1x2/ straight/
    └── cband/ oband/            # 按波段组织
```

### 单个器件文件的模式

以 `mzi_2x2_heater_tin_cband.py` 为例：

1. **模块级 docstring**，特殊格式：前面一段自然语言 + `---` + 一段 YAML
   - 字段包括：`ports`、`Insertion loss`、`Drive voltage/power`、`Args`、`specs`
   - 这个 YAML 就是 LLM 做组件检索时看到的"说明书"
2. **一个带 `@gf.cell` 装饰的同名函数**（`mzi_2x2_heater_tin_cband(delta_length, length)`）
   - 用 `gdsfactory.components.mzi2x2_2x2` 把 `_mmi2x2`、`heater_tin_cband` 等组合成完整器件
   - 导出 `o1..o4` 端口
3. **一个 `get_model(model="fdtd")` 函数**：把所有子组件的 SAX 模型合并成一个 dict，供上层电路仿真使用
4. **一个 `if __name__ == "__main__":` 块**：方便独立调试（画版图 / 扫频 / 打印 footprint）

> `DemoPDK.py` 里的 `import_modules` / `import_models` 就是靠这个统一接口（同名函数 + `get_model`）来批量接入所有器件。

---

## 六、一次"自动工作流"的端到端路径

把上面各模块串起来看：假设用户输入 `"Design a 2x2 Mach-Zehnder interferometer with heaters"`，流程如下：

### ① Entity Extraction

- 调用 `llm_api.entity_extraction` + `prompts.yaml`
- LLM 产出 `pretemplate`：`components_list: [2x2 MZI with heater]`、`title`、`circuit_instructions`

### ② Component Specification

- `webapp.run_step_by_step_component_specification` → `llm_retrieve` / `llm_search`
- 用 BM25 / sentence-transformers 在 `utils.search_directory_for_docstrings` 的结果中检索
- 候选：`mzi_2x2_heater_tin_cband`、`mzi_2x2_pn_diode` 等
- UI 里让用户用 radio 选（或自动选最佳）

### ③ Circuit DSL Creation

- `DemoPDK.footprint_netlist` / `get_params` / `get_ports_info` 补全每个节点的 `dx/dy`、参数、端口
- 如果匹配到某个 `TEMPLATE_*`，直接用 `templates.yaml` 里的边
- 否则 LLM（`dot_add_edges`）生成边

### ④ Schematic Generation

- `utils.circuit_to_dot` 生成 DOT
- `dot_planarity` 校验无交叉
- `get_graphviz_placements` 取坐标
- `utils.add_placements_to_dsl` / `add_final_ports` 填回 DSL

### ⑤ Layout & Simulation

- `utils.dsl_to_gf` → `DemoPDK.yaml_netlist_to_gds`：
  - `gdsfactory` 写 GDS 文件
  - `sax.circuit(..., all_models)` 组合每个器件的 S 参数模型做电路级仿真
  - `matplotlib` 画 `plot_gds.png` 和 `plot_sax.png`
- 可选：`Photon/drc/drc.py::run_drc(gds_file, testcase)` 调 KLayout 输出 DRC 报告

### ⑥ 可选优化

- `DemoPDK.circuit_optimizer(session)` 用 Bayesian Optimization 反向调 `delta_length` 等自由参数，逼近目标传递函数

### 可观测性

- Token 消耗：`llm_api.get_token_usage` 监控
- 运行时间：`webapp.py` 的 `session.p100_start_time` 系列变量追踪

---

## 七、可扩展性

README 明确列出的扩展点：

| 需求 | 做法 |
|---|---|
| **加新器件** | 在 `KnowledgeBase/DesignLibrary/` 加一个 `.py`（符合 docstring + `get_model` 的格式），重启即可被 `list_python_files` 自动发现 |
| **加新 DRC 规则** | 改 `Photon/drc/drc_script.drc` |
| **接入新 LLM** | 在 `llm_api.py` 加 `call_xxx`，再在 `call_llm` 路由表里加 model 名字；然后在 `webapp.py` 顶部改对应的 `*_model` 常量 |
| **加新模板** | 写进 `templates.yaml`（`TEMPLATE_*` 给抽象意图，`CIRCUIT_*` 给具体落地电路） |

---

## 八、关于 "Reflexion" 后缀的说明

README / CHANGELOG 并没有显式描述 "Reflexion" 机制的实现细节（CHANGELOG 目前是空的），但从代码层面看，具备 Reflexion 式"自我检查-修正"雏形的地方有：

- `prompts.yaml::dot_verify`：让 LLM 重新检查并修复自己生成的 DOT 图
- `utils.dot_planarity` + `llm_api.dot_add_edges` / `dot_verify`：对生成的连线做平面性验证，失败则让 LLM 再改
- `llm_api.verify_input_clarity`：对用户输入做澄清确认

仓库里没有独立的 `reflexion.py` 文件，因此 "Reflexion" 应当是**内嵌在工作流的验证-重写循环**中，而非一个独立模块。如需深挖，可重点阅读：

- `llm_api.py` 中 `dot_verify` / `netlist_cleanup` 的调用位置
- `webapp.py` 中每一阶段的 `try/except` 重试逻辑

---

## 九、快速上手命令速查

```bash
# 1. 系统依赖（Ubuntu/Debian）
sudo apt-get install graphviz libgraphviz-dev pkg-config klayout
sudo apt-get install -y build-essential python3-dev swig

# 2. Python 环境
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install kfactory==0.21.1   # 覆盖 gdsfactory 带来的冲突版本

# 3. 在项目根目录建 .env 放 API keys（至少 OPENAI_API_KEY）

# 4. 设置 PYTHONPATH
export PYTHONPATH='.'

# 5. 在 PhotonicsAI/ 下建 log 目录
mkdir -p PhotonicsAI/log

# 6. 启动
make run
# 或
streamlit run PhotonicsAI/Photon/webapp.py
```

---

## 十、文件/函数速查表

### 关键入口

| 文件 | 角色 |
|---|---|
| `PhotonicsAI/Photon/webapp.py` | Streamlit 前端 + 5 阶段工作流编排 |
| `PhotonicsAI/Photon/llm_api.py` | 多厂商 LLM 统一调用 + 业务级 prompt 调度 |
| `PhotonicsAI/Photon/utils.py` | DSL/DOT/GDSFactory 转换、SAX 模型加载 |
| `PhotonicsAI/Photon/DemoPDK.py` | 动态构建 PDK、GDS 生成、Bayesian 优化 |
| `PhotonicsAI/Photon/drc/drc.py` | 调用 KLayout 做 DRC |
| `PhotonicsAI/config.py` | `PATH` 单例与 `.env` 读取 |

### 关键数据文件

| 文件 | 角色 |
|---|---|
| `PhotonicsAI/Photon/prompts.yaml` | 全部 LLM 提示词（dot_verify、edges_yaml_to_dot、absorb_settings …） |
| `PhotonicsAI/Photon/templates.yaml` | 预定义抽象模板 `TEMPLATE_*` 与具体电路 `CIRCUIT_*` |
| `PhotonicsAI/KnowledgeBase/DesignLibrary/*.py` | 所有器件 cell 与 SAX 模型 |
| `PhotonicsAI/KnowledgeBase/FDTD/` | 预跑好的 FDTD S 参数 `.npz` |
| `Testbench.xlsx` | 102 条测试 prompt |

---

## 参考

- 论文：<https://arxiv.org/abs/2508.14123>
- 项目 README：`./README.md`
- 详细教程：`./GETTING_STARTED.md`
- 示例输出：`./GETTING_STARTED_EXAMPLE_OUTPUTS/`
