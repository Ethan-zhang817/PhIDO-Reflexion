"""Graph-only OpenAI calls for legacy EE / component-search (configurable model).

The classic PhIDO ``llm_api`` entry points use a fixed default model. The
LangGraph path uses ``legacy_stage_model``; implementations call
:func:`PhotonicsAI.graph.llm.invoke_structured_openai` (standard ``Runnable``-backed
structured output) so ``webapp`` can stay unchanged.
"""

from __future__ import annotations

import json
from typing import Any

from PhotonicsAI.Photon import llm_api
from PhotonicsAI.graph.llm import StructuredOpenAIRequest, invoke_structured_openai
from PhotonicsAI.graph.state import DEFAULT_LEGACY_STAGE_MODEL

_InputEntities = llm_api.InputEntities
_MatchedComponents = llm_api.MatchedComponents


def entity_extraction_with_model(
    input_prompt: str, *, model: str = DEFAULT_LEGACY_STAGE_MODEL
) -> dict[str, Any]:
    sp_normal = """You are an assistant to a photonic engineer.
Your task is to extract specific information from the input text and present it in the following structured format:

title: A concise title describing the function of the photonic circuit based on the input text.

components_list: Extract a list of components mentioned in the input text.
This list must contain at least one component. For each component:
- Include all provided specifications and descriptions, if any.
- Include the number of optical input and output ports in the format [input]x[output] (e.g., 1x2),
  but only if explicitly stated. Do not assume numbers that are not provided.
- Do not list specifications or descriptive modifiers as separate components. For example if a phase shifter or heater is integrated into a photonic modulator (MZI) only list the modulator.
- If multiple copies of the same component are described, list each one explicitly.
- If it is implied that a component is used more than once, list each instance separately.

circuit_instructions: Extract any instructions from the input text about how the components
  should be connected or used in the circuit. If none are provided, set this to an empty string.

brief_summary: Provide a summary of the input text in less than 150 words.

Note: All extracted information must be exclusively derived from the input text.
"""

    sp_paper = """You are an assistant for a photonic engineer.
Your task is to extract specific information from the input text and attached figure and present it in the following structured format:

components_list: extract a list of on-chip photonic components, following these guidelines:

(1) For each component, include all provided specifications and descriptions, if any.

(2) For each component, include the number of optical input and output ports in the format [input]x[output] (e.g., 1x2). Make an educated guess from the function if not explicitly stated. For example if the component is in an add-drop configuration it should be 2x2.

(3) Do not list specifications or descriptive modifiers as separate components. For example if a phase shifter or heater is integrated into a photonic modulator only list the modulator.

(4) If multiple copies of the same component are described, explictly state the number of copies of the component. i.e. 4 Germanium photodetectors

(5) Do not group components into "Arrays". Separate arrays into individual components.

(6) Exclude electronic components (e.g., oscilloscope, transimpedance amplifier, DAC, RF source)
and off-chip components (e.g., fiber, free-space lenses/lasers, EDFA).

(7) If the text does not contain any on-chip photonic components, set this field to an empty list.

(8) compose this list in YAML. have each instance labelled as C1, C2, .... Add any
provided specification to each component. make sure multiple copies of components are created as new instances.

circuit_instructions: Extract any instructions from the input text about how the components should be connected or used in the circuit. If none are provided, set this to an empty string.

brief_summary: Provide a summary of the input text in less than 150 words."""

    sys_prompt = sp_paper if len(input_prompt) > 1000 else sp_normal
    r = invoke_structured_openai(
        StructuredOpenAIRequest(
            system_prompt=sys_prompt,
            user_prompt=input_prompt,
            response_model=_InputEntities,
            model=model,
            run_name="graph:legacy:entity_extraction",
        )
    )
    return r.model_dump()


def llm_search_with_model(
    query: str, contexts: list, *, model: str = DEFAULT_LEGACY_STAGE_MODEL
) -> Any:
    desc_ = dict(enumerate(contexts))
    desc_json = json.dumps(desc_, indent=2)

    sys_prompt = f"""You are a photonic chip layout developer.
    You have access to {len(contexts)} photonic devices/components, provided in the JSON below.
    Your task is to find the best-matched component(s) based on the described functionality and port configuration.

    Key matching criteria:
    1. Often a component is described with many specifications and modifiers.
       Identify the main component and functionality and search for a match to that.
       (e.g. a coupler with 10 nm bandwidth and with s-bend; the coupler is the main component and not the s-bend).
    2. Functionality is the highest priority.
    3. Match optical port configuration (e.g., [input]x[output] such as 1x2) when possible.
    4. If no exact match exists, prioritize functionality, then select the closest port configuration.
    5. If multiple close matches are found, rank them by the number of ports first, then by functionality closeness.
    6. If the query is ambiguous (missing function or port count details), make reasonable assumptions and provide a note in 'match_comment'.
    7. If no match is found, output the nearest match. Never output an empty list.

    For each matched item, return a qualitative score:
    - exact: Exactly matches both functionality and port configuration.
    - partial: A partial match with some differences in functionality or port configuration.
    - poor: Weak match or significantly different.

    Output the following:
    - match_list: List of matched item IDs.
    - match_scores: Corresponding qualitative scores.

    JSON of available components:

    {desc_json}
    """

    r = invoke_structured_openai(
        StructuredOpenAIRequest(
            system_prompt=sys_prompt,
            user_prompt=query,
            response_model=_MatchedComponents,
            model=model,
            run_name="graph:legacy:llm_search",
        )
    )
    if len(r.match_list) != len(r.match_scores):
        print(
            "Error: match_list and match_scores have different lengths.... trying again"
        )
        print(r.match_list, r.match_scores)
        r = invoke_structured_openai(
            StructuredOpenAIRequest(
                system_prompt=sys_prompt,
                user_prompt=query,
                response_model=_MatchedComponents,
                model=model,
                run_name="graph:legacy:llm_search:retry",
            )
        )
    return r
