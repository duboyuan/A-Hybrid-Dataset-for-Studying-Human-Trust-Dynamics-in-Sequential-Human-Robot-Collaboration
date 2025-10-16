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

### Dataset Access and Licensing

- The JSON files in `data/` are provided for reproducibility. If your downstream use requires raw/derivative datasets not committed here, please contact the authors.

### FAQ

- Q: How do I replace the provided JSON with my own?
  - A: Place your files under `data/` with the same schema; update `validation/main.py` to point to your paths if needed.
- Q: Can I disable specific plots?
  - A: Comment out the corresponding calls at the bottom of `validation/main.py` under `if __name__ == '__main__':`.

