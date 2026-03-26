# Galetech BOO Optimizer and Bankability Assistant

## Overview
Galetech is a techno-economic optimization tool for designing and evaluating on-site energy systems under a Build-Own-Operate (BOO) model. The application co-optimizes wind, solar PV, battery energy storage (BESS), and electric boiler capacity/dispatch to identify commercially viable project configurations.

The tool is implemented as an interactive Streamlit application and is designed for project screening, investment decision support, and auditable reporting.

## Core Objectives
The model evaluates candidate portfolios against financial and decarbonization outcomes, including:

- CAPEX, OPEX, payback, IRR, and NPV
- Electricity offset and thermal decarbonization
- Gas displacement and CO2 reduction
- Dispatch feasibility under site and grid constraints

Users can select optimization priority by:

- Shortest Payback
- Highest IRR
- Highest NPV

## Optimization Architecture
The optimization is executed in two stages:

1. Coarse capacity sweep to locate promising regions in the design space.
2. Fine local search around the stage-1 best candidate.

For each candidate, a daily dispatch LP is solved across representative day profiles with annual weighting.

## Current Commercial and Carbon Logic
The model reflects the latest commercial logic:

- Gas purchase and direct carbon cost are borne by the customer, not Galetech.
- Carbon-credit value from avoided emissions is monetized.
- 50% of carbon-credit value is allocated to Galetech as revenue (`carbon_credit_share = 0.5`).
- CO2 reduction includes both:
	- Natural gas displacement (thermal side)
	- Grid import displacement (electric side)

This structure provides explicit economic incentive for higher decarbonization performance while preserving transparent revenue attribution.

## Main Features
- Joint sizing of wind, solar, BESS, and electric boiler.
- Constraint-aware dispatch optimization (grid limits, SOC dynamics, curtailment, heat balance).
- Pre-optimization preview charts generated on demand.
- Persistent report behavior: once a report is generated, it remains visible across UI reruns (for example when generating preview charts) until a new optimization run is triggered.
- Built-in Monte Carlo sensitivity workflow.
- Exportable auditable hourly dispatch pack (CSV).

## Inputs
The app supports uploaded hourly CSV/XLSX data with one row per hour.

Preferred columns:

- `elec_load`
- `gas_load`
- `wind_speed`
- `irradiance`

If any column is missing, the app falls back to default representative profiles for the missing fields.

## Outputs
The application produces:

- Executive recommendation for best configuration
- Financial dashboard (payback, IRR, NPV)
- Customer and developer decarbonization economics
- Annual cash-flow table (including cumulative profit)
- Benchmarking and heat maps
- Monte Carlo risk summary
- Downloadable hourly dispatch traces

## Project Files
- `Galetech.py`: main Streamlit app and optimization workflow
- `GaletechPP.py`, `GaletechIRR.py`: related model variants/experiments
- `test_gurobi.py`: solver environment test
- Typical-day CSV files: sample/input profile data

## Running the App
Install dependencies in your Python environment, then run:

```bash
streamlit run Galetech.py
```

## Solver Note
The dispatch LP uses CVXPY with GUROBI in the current implementation. Ensure GUROBI is correctly installed and licensed in your environment.
