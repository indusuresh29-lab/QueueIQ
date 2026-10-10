# QueueIQ — Queueing Decision Intelligence

QueueIQ is an MBA Operations & Supply Chain Analytics capstone application for analysing and optimizing waiting-line systems.

## Academic scope

The application is designed around the Group 6 Queuing Theory requirements:

- Estimate arrival rate (λ) and service rate (μ) from data
- Diagnose arrival behaviour, including a chi-square Poisson goodness-of-fit test
- M/M/1 analysis
- M/M/c / Erlang-C analysis
- M/D/1 analysis
- Finite-capacity queue
- Finite-population queue
- Little's Law validation
- Discrete-event simulation
- Peak/off-peak and parameter sensitivity
- Server-count what-if analysis
- Waiting-cost + server-cost optimisation
- Priority queues
- Governance and decision recommendations

## Architecture

```text
Dataset / Manual Input
        ↓
Diagnostics & Parameter Estimation
        ↓
Queue Model
        ↓
Analytical Results
        ↓
Simulation & Validation
        ↓
Sensitivity / What-If
        ↓
Cost Optimisation
        ↓
Decision Recommendation
        ↓
Governance
```

The application deliberately does NOT contain a natural-language problem-statement parser.

## Run locally — macOS

```bash
cd QueueIQ
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## First test

The default manual example is:

- λ = 12 customers/hour
- μ = 15 customers/hour
- servers = 1
- Exponential service
- FIFO

This gives ρ = 0.80 for M/M/1.

## Project folders

- `app.py` — Streamlit interface
- `core/queue_models.py` — analytical queueing mathematics
- `core/priority_queue.py` — priority queue calculations
- `core/simulation.py` — discrete-event simulation
- `core/diagnostics.py` — CSV diagnostics and rate estimation
- `core/sensitivity.py` — sensitivity and what-if analysis
- `core/optimization.py` — staffing/cost optimisation
- `core/validation.py` — theory vs simulation and Little's Law checks
- `data/sample/` — sample dataset
- `tests/` — model tests
- `utils/helpers.py` — shared utilities

## Important academic rule

The application is a decision-support system. Analytical assumptions, stability, distributional fit, data quality, and operational feasibility must be reviewed before implementing staffing decisions.
