# Galetech BOO Optimizer and Bankability Assistant

## Overview
Galetech is a Streamlit-based techno-economic optimizer for on-site energy systems under a Build-Own-Operate (BOO) model.

It co-optimizes:

- Wind turbine selection and count
- Solar PV capacity (MW)
- Battery energy storage (MWh)
- Electric boiler capacity (kW)

The app is designed for project screening, investment decision support, and auditable dispatch reporting.

## What The Model Optimizes
Each candidate configuration is evaluated on annualized financial and decarbonization outcomes, including:

- Payback, IRR, NPV
- CAPEX and OPEX
- Electricity supplied to customer load
- Thermal decarbonization (gas displaced by E-boiler heat)
- CO2 displacement and carbon-credit value

Users can choose optimization objective:

- Shortest Payback
- Highest IRR
- Highest NPV

## Optimization Architecture
The optimization runs in two stages:

1. Coarse sweep across wind/solar/BESS/E-boiler design space
2. Fine search around the best stage-1 region

For each capacity candidate, a weighted representative-day dispatch MILP is solved with CVXPY + GUROBI.

## Dispatch and Commercial Logic (Current)
The current implementation reflects a customer-first BOO dispatch logic:

- Customer electric demand is split into project supply and customer grid backup.
- Only project-supplied electricity is billed at BOO PPA price.
- Customer backup grid purchase is not project revenue.
- Project-side grid import is modeled separately and only allowed for BESS charging.
- Hard service-priority constraint: if customer needs grid backup in an hour, the project cannot simultaneously charge BESS, export to grid, or curtail.
- Gas purchase and direct carbon compliance costs are treated as customer-side economics.
- Carbon-credit value from avoided emissions is fully allocated to the energy system (`carbon_credit_share = 1.0` in UI defaults).

## Weather and Typical-Day Inputs
The app supports uploaded hourly CSV/XLSX data (24h blocks), and builds representative day profiles using user-provided day weights.

Expected columns:

- `elec_load`
- `gas_load`
- `wind_speed`
- `irradiance` (absolute W/m^2)

If weather-related fields are missing, the app uses weather defaults. If no location is set, Dublin is used by default. If weather fetch fails, synthetic seasonal profiles are used with Dublin as nominal fallback.

## Key UI Defaults (Current)
- Site area: 15 acres
- Customer electricity price: 130 EUR/MWh
- BOO PPA electricity price: 100 EUR/MWh
- Grid export price: 100 EUR/MWh
- Customer grid-backup penalty: 0 EUR/MWh (optional soft penalty)

## Outputs and Report Tabs
After optimization, the app provides:

- Executive summary with headline recommendation
- Financial KPI cards (Payback, IRR, NPV)
- Customer green electricity share KPI
- Decarbonization and customer-saving metrics
- Cost breakdown table and annual cash-flow table
- Technology benchmarking table
- Monte Carlo risk analysis (P10/P50/P90 + histogram guidance)
- Downloadable hourly dispatch audit CSV

Note: legacy heat maps were removed from the current report workflow.

## Project Files
- `Galetech.py`: main Streamlit app, optimization engine, UI, reporting
- `typical day data.csv`: sample profile data
- `requirements.txt`: Python dependencies for local development/runtime setup

## Run
Create/activate a Python environment, install dependencies, then run:

```bash
pip install -r requirements.txt
streamlit run Galetech.py
```

Optional (recommended) virtual environment workflow:

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run Galetech.py
```

## Solver Note
The dispatch problem uses CVXPY with the GUROBI solver (mixed-integer model). Ensure GUROBI is installed and licensed in your environment.
