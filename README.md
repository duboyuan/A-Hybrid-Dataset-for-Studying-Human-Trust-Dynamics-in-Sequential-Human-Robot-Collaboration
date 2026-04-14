## Trust Dynamics across LLM-Simulated, VR Based, and Real World Settings

This repository contains the code accompanying the paper “A Hybrid Dataset for Studying Human Trust Dynamics in Sequential Human-Robot Collaboration”. It provides concise analysis scripts and simple trust transfer baselines across three settings: LLM Simulated, VR Based, and Real World.

### Repository Structure

- `data/`
  - `LLM_Simulated_data.json`
  - `VR_Based_data.json`
  - `Real_World_data.json`
- `validation/`
  - `main.py`: Core analyses and plotting utilities (trust distributions, trust change analyses, dependence tests, etc.)
  - `trust_predict.py`: Lightweight trust transfer model wrapper
  - `trust_transfer_model.py`: Model definition and training/testing utilities
- `utils/`
  - `constants.py`: API keys and model-related constants for optional LLM scripts
  - `utils.py`: Shared helpers (e.g. `auto_parse_text` for tagged LLM outputs)
- `llm/` (optional — LLM-based human action, trust update, and reflection)
  - `llm_base/`: `Agent` wrapper, `model()` factory (provider routing)
  - `llm_human_action/`: decision prompts + `LlmHumanAction`
  - `llm_trust/`: trust-update prompts + `LlmTrust`
  - `llm_reflection/`: reflection prompts + `LlmReflection`
  - `decision_trust_reflection_demo.py`: end-to-end demo (decision → trust → two reflection checks, logs to a `.txt`)

### Data Description (high-level)

Each JSON file is a list of participants. A participant record contains:
- `human_id`: string identifier
- `modality`: `LLM Simulated`, `VR Based`, or `Real World`
- `trust`: list[float], normalized to [0, 1]
- `state`: list[int]
- `robot_observation`: list[float|int]
- `robot_decision_making`: list[int]
- `human_decision_making`: list[int]
- `task_result`: list[int]

### Installation

1. Create environment and install dependencies
   Requires Python 3.10–3.11.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Minimal dependencies (pinned in `requirements.txt`):
- numpy, pandas, scipy, matplotlib, seaborn, tabulate, torch (CPU ok)

**LLM scripts** (`llm/`) additionally need LangChain / LangGraph and the provider packages you use, for example:

```bash
pip install langchain-openai langchain-core langgraph
# optional local models:
# pip install langchain-ollama
```

Match the `--model` / `llm_model_name` string to a branch implemented in `llm/llm_base/llm.py` (e.g. `gpt-4o`, `gpt-4o-mini`).

### Quick Start

Run the core analysis and figures:
```bash
python -m validation.main
```

This will:
- Print descriptive statistics of trust per modality
- Generate distribution plots and trust change analyses
- Produce the updated figure for average |Δtrust| by reward sign 
- Simple trust transfer experiments
- Analyze trust change vs task factors (robot-state agreement, task success, adoption) via `analyze_trust_change_and_factors_v2`

### LLM pipeline demo (human decision → trust update → reflection)

1. **Configure API keys** in `utils/constants.py`. Which variable is used depends on the model alias in `llm/llm_base/llm.py` (e.g. OpenAI models use `OPENAI_API_KEY`).
2. From the **repository root** (the folder that contains both `llm/` and `utils/`), run:

```bash
python llm/decision_trust_reflection_demo.py --model gpt-4o-mini -o outputs/llm_pipeline_run.txt
```

The script adds the repo root to `sys.path`, so this command works even though the file lives under `llm/`.

**Expected LLM output format** (parsed by `utils.auto_parse_text`):

- Human action: `<Decision>Directly enter</Decision>` or `<Decision>Call for support</Decision>`
- Trust update: `<trust>0.85</trust>`
- Reflection: `<Judgment>True</Judgment>` or `<Judgment>False</Judgment>`

The run log written to `-o` includes the full prompts and raw model outputs for debugging.

### Dataset Access and Licensing

- The JSON files in `data/` are provided for reproducibility. If your downstream use requires raw/derivative datasets not committed here, please contact the authors.

### FAQ

- Q: How do I replace the provided JSON with my own?
  - A: Place your files under `data/` with the same schema; update `validation/main.py` to point to your paths if needed.
- Q: Can I disable specific plots?
  - A: Comment out the corresponding calls at the bottom of `validation/main.py` under `if __name__ == '__main__':`.
- Q: `ModuleNotFoundError: No module named 'llm'` when running the demo?
  - A: Run `python llm/decision_trust_reflection_demo.py` from the **repository root**, not from inside `llm/`. The demo prepends the parent of `llm/` to `sys.path`.
- Q: Parsed tags look truncated (e.g. decision text missing leading letters)?
  - A: Use the current `utils.auto_parse_text` implementation (tag bodies are taken from the regex capture group, not `str.strip` on tag names).

