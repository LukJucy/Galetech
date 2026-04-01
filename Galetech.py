import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy_financial as npf
import io
import json
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

# ==========================================
# 1. Core optimization engine
# ==========================================
class GaletechAssetOptimizer:
    def __init__(self, params):
        self.params = params
        self.bess_eff = 0.95
        self.bess_cycles = 8000
        self.gas_carbon_factor = 0.202   # kg CO2 per kWh_th of natural gas
        self.project_life_years = 20
        self.discount_rate = 0.10
        self.acre_to_m2 = 4046.8564224
        self._warned_solver_exceptions = set()

        self.turbine_models = {
            "None":        {"mw": 0,    "curve": [0]*13,                                                          "equip_cost": 0,       "civil_cost": 0},
            "EWT 500kW":   {"mw": 0.5,  "curve": [0, 0, 3, 15, 35, 50, 120, 220, 340, 480, 500, 500, 500],       "equip_cost": 800000,  "civil_cost": 1000000},
            "EWT 1MW":     {"mw": 1.0,  "curve": [0, 35, 45, 54, 63, 100, 200, 300, 480, 610, 700, 850, 1000],   "equip_cost": 1300000, "civil_cost": 1200000},
            "E82 2.3MW":   {"mw": 2.3,  "curve": [0, 0, 3, 25, 82, 174, 321, 532, 815, 1180, 1580, 1900, 2300],  "equip_cost": 2500000, "civil_cost": 1500000},
            "V90 3MW":     {"mw": 3.0,  "curve": [0, 0, 0, 4, 77, 190, 353, 581, 886, 1273, 1710, 2200, 3000],   "equip_cost": 3100000, "civil_cost": 1800000},
            "E115 4.2MW":  {"mw": 4.2,  "curve": [0, 0, 9, 57, 163, 351, 628, 1008, 1501, 2092, 2733, 3300, 4200],"equip_cost": 4200000, "civil_cost": 2200000},
            "E138 4.26MW": {"mw": 4.26, "curve": [0, 0, 2, 69, 250, 540, 952, 1506, 2173, 2865, 3474, 3900, 4260],"equip_cost": 4500000, "civil_cost": 2500000},
        }

    # ------------------------------------------------------------------
    # Wind power output from turbine power curve
    # ------------------------------------------------------------------
    def get_wind_power(self, wind_speed, model_name, count, shock_factor=1.0):
        if count == 0 or model_name == "None":
            return np.zeros_like(wind_speed, dtype=float)
        curve = self.turbine_models[model_name]["curve"]
        w_speed_capped = np.clip(wind_speed, 0, 12)
        return np.interp(w_speed_capped, np.arange(13), curve) * count * shock_factor

    # ------------------------------------------------------------------
    # CAPEX calculation
    # ------------------------------------------------------------------
    def get_capex(self, t_model, t_count, s_mw, b_mwh, capex_shock=1.0, eboiler_kw=None):
        equip_wind  = self.turbine_models[t_model]["equip_cost"] * t_count
        equip_solar = s_mw  * self.params['cost_solar_mw']
        equip_bess  = b_mwh * self.params['cost_bess_mwh']
        eboiler_capex = 0
        eboiler_kw_effective = self.params.get('eboiler_max_kw', 0) if eboiler_kw is None else eboiler_kw
        if self.params.get('enable_e_boiler', True):
            eboiler_capex = self.params.get('eboiler_capex_per_kw', 120) * eboiler_kw_effective
        equip_total = equip_wind + equip_solar + equip_bess + eboiler_capex

        civils = 0
        if t_count > 0: civils += self.turbine_models[t_model]["civil_cost"] * t_count
        if s_mw  > 0:   civils += self.params['civil_solar']
        if b_mwh > 0:   civils += self.params['civil_bess']
        if self.params.get('enable_e_boiler', True) and eboiler_kw_effective > 0:
            civils += self.params.get('eboiler_civil_cost', 80000)

        w_mw = self.turbine_models[t_model]["mw"] * t_count
        sources_count = sum([w_mw > 0, s_mw > 0, b_mwh > 0,
                     self.params.get('enable_e_boiler', True) and eboiler_kw_effective > 0])
        elec = sources_count * self.params['elec_per_source']

        subtotal = equip_total + civils + elec
        pm_cost  = subtotal * self.params['pm_rate']
        return (subtotal + pm_cost) * capex_shock, equip_total, civils, elec, pm_cost

    # ------------------------------------------------------------------
    # OPEX calculation
    # ------------------------------------------------------------------
    def get_opex(self, total_capex, t_count=0, s_mw=0):
        insurance   = total_capex * self.params['insurance_rate']
        maintenance = total_capex * self.params['maintenance_rate']
        lease_base  = self.params['land_lease']
        site_area_acre = float(self.params.get('site_area_acre', 0.0) or 0.0)
        if site_area_acre > 0:
            # Lease scales with utilized site share under the same land rule used by constraints.
            max_units = site_area_acre / 5.0
            used_units = max(0.0, t_count / 2.0 + s_mw)
            lease_share = min(1.0, used_units / max_units) if max_units > 0 else 1.0
            lease = lease_base * lease_share
        else:
            lease = lease_base
        eboiler_fixed_opex = self.params.get('eboiler_fixed_opex', 15000) if self.params.get('enable_e_boiler', True) else 0
        return insurance + maintenance + lease + eboiler_fixed_opex, insurance, maintenance, lease + eboiler_fixed_opex

    def _safe_irr(self, capex, net_profit):
        """Return IRR with robust fallback for always-negative cashflows.

        If cashflows do not change sign (e.g. all negative), npf.irr returns NaN.
        In that case we report -100% to reflect clearly non-viable economics,
        instead of masking the value as 0%.
        """
        irr = npf.irr([-capex] + [net_profit] * self.project_life_years)
        if irr is None or not np.isfinite(irr):
            return -1.0
        return float(irr)

    # ------------------------------------------------------------------
    # Daily dispatch optimisation (MILP with service-priority binary variables)
    #
    # FIX 1: Removed z_grid / z_bess Big-M mutual-exclusion constraints.
    #   The economic price spread (buy price > sell price) naturally
    #   prevents the solver from simultaneously buying and selling power.
    #   The original Big-M formulation with continuous relaxation caused
    #   ECOS to report numerical infeasibility on well-posed problems.
    #
    # FIX 2: SOC initial-state dynamics now cover t=0 correctly.
    #   Previously p_ch[0]/p_dis[0] appeared in the energy balance but
    #   were not linked to any SOC equation, giving the solver "free"
    #   charge/discharge at hour 0 with no state cost.
    # ------------------------------------------------------------------
    def evaluate_combination(self, t_model, t_count, s_mw, b_mwh,
                             rep_days, return_traces=False,
                             wind_shock=1.0, ppa_shock=1.0,
                             eboiler_kw=None):

        annual_revenue          = 0.0
        annual_co2_saved        = 0.0
        annual_btm_supply_kwh   = 0.0
        annual_heat_from_elec   = 0.0
        annual_gas_used_kwh     = 0.0
        annual_curtailed_kwh    = 0.0
        annual_curtail_cost     = 0.0

        # Battery wear cost (€ per kWh discharged)
        bess_wear_cost = (
            self.params['cost_bess_mwh'] / (self.bess_cycles * 1000)
            if b_mwh > 0 else 0.0
        )

        carbon_credit_share = self.params.get('carbon_credit_share', 1.0)
        grid_carbon_factor  = self.params.get('grid_carbon_factor', 0.35)  # kg CO2 per kWh_e
        eboiler_enabled  = self.params.get('enable_e_boiler', True)
        eboiler_eff      = self.params.get('eboiler_eff', 0.95)
        eboiler_var_cost = self.params.get('eboiler_var_cost', 0.008)   # €/kWh_th
        p_heat_sell      = self.params.get('p_heat_sell', 0.04)         # €/kWh_th
        p_grid_buy_price = self.params.get('p_grid_buy', self.params.get('p_galetech', 0.10) * 1.2)
        p_cust_grid_penalty = self.params.get('p_cust_grid_penalty', 0.0)  # €/kWh, service-quality soft penalty
        eboiler_max_kw   = self.params.get('eboiler_max_kw', 1e6) if eboiler_kw is None else eboiler_kw

        export_limit_kw   = self.params.get('export_limit_kw', 5000)
        grid_buy_limit_kw = self.params.get('grid_buy_limit_kw', 5000)
        bess_max_power_kw = (b_mwh * 1000) * 0.5 if b_mwh > 0 else 0.0
        cap_kwh           = b_mwh * 1000

        all_traces = []

        for day_idx, day in enumerate(rep_days):
            T = len(day['elec_load'])

            # ---- decision variables ----
            p_ch           = cp.Variable(T, nonneg=True)   # BESS charge (kW)
            p_dis          = cp.Variable(T, nonneg=True)   # BESS discharge (kW)
            soc            = cp.Variable(T, nonneg=True)   # BESS state of charge (kWh)
            p_site_supply  = cp.Variable(T, nonneg=True)   # project electricity served to customer electric load (kW)
            p_cust_grid    = cp.Variable(T, nonneg=True)   # customer direct grid backup purchase (kW)
            p_eboiler_elec = cp.Variable(T, nonneg=True)   # electricity into e-boiler (kW)
            p_gas_used     = cp.Variable(T, nonneg=True)   # gas heat still consumed (kWh_th)
            p_grid_sell    = cp.Variable(T, nonneg=True)   # export to grid (kW)
            p_grid_buy_proj = cp.Variable(T, nonneg=True)  # project-side grid import for BESS charging (kW)
            p_curtail      = cp.Variable(T, nonneg=True)   # curtailed renewable (kW)
            z_cust_short   = cp.Variable(T, boolean=True)  # 1 if customer uses grid backup in this hour
            z_bess_charge  = cp.Variable(T, boolean=True)  # 1 if BESS is charging, 0 if discharging

            # ---- generation profiles (from hourly wind speed & irradiance) ----
            p_wind = np.zeros(T, dtype=float)
            if t_model != "None" and 'wind_speed' in day:
                p_wind = self.get_wind_power(day['wind_speed'], t_model, t_count, shock_factor=wind_shock)

            p_solar = np.zeros(T, dtype=float)
            if s_mw > 0 and 'irradiance' in day:
                # Area-based PV model using absolute irradiance (W/m^2).
                # Land rule: 5 acres per 1 MW PV -> area scales with installed MW.
                irr_wm2 = np.clip(day['irradiance'], 0.0, 1400.0)
                
                solar_land_area_m2 = s_mw * 5.0 * self.acre_to_m2

                # Effective panel-covered fraction of land area.
                # If not provided, auto-calibrate so 1 MW on 5 acres can reach rated power at 1000 W/m^2.
                pv_land_utilization = self.params.get('pv_land_utilization')
                if pv_land_utilization is None:
                    eta = max(self.params['solar_efficiency'], 1e-6)
                    pv_land_utilization = min(1.0, 1000.0 / (eta * 5.0 * self.acre_to_m2))

                p_solar_area_kw = (
                    irr_wm2 * solar_land_area_m2
                    * self.params['solar_efficiency']
                    * pv_land_utilization
                    / 1000.0
                )
                # AC output cannot exceed installed inverter capacity.
                p_solar = np.minimum(p_solar_area_kw, s_mw * 1000.0)

            # ---- constraints ----
            constraints = [
                # (C1) Project energy balance: project generation/discharge serves
                # customer load, e-boiler demand, battery charging and grid export.
                p_wind + p_solar + p_dis + p_grid_buy_proj
                    == p_site_supply + p_eboiler_elec + p_grid_sell + p_ch + p_curtail,

                # (C2) Customer electric load split:
                # project supply + customer direct grid purchase = customer demand.
                p_site_supply + p_cust_grid == day['elec_load'],

                # (C3) Heat balance: e-boiler heat + gas = total gas-heat demand
                # Note: day['gas_load'] is in kWh per hour (hourly thermal energy), treated as kWth (power) in hourly dispatch model
                eboiler_eff * p_eboiler_elec + p_gas_used == day['gas_load'],

                # (C4) Grid limits (no Big-M; price spread prevents simultaneous buy+sell)
                p_grid_sell <= export_limit_kw,
                # Customer direct grid backup is limited by import capacity.
                p_cust_grid <= grid_buy_limit_kw,
                # Project grid import and customer backup share the same import headroom.
                p_cust_grid + p_grid_buy_proj <= grid_buy_limit_kw,

                # (C5) Curtailment cannot exceed available renewable generation
                p_curtail <= p_wind + p_solar,
            ]

            # (C5.1) Hard service-priority constraint:
            # If customer uses grid backup (p_cust_grid > 0), project cannot simultaneously
            # charge BESS, export to grid, or curtail energy in that same hour.
            # Use hour-specific Big-M (M_t) instead of one global constant.
            # Tighter bounds usually improve MILP numerics and solve speed.
            max_surplus_kw = (
                grid_buy_limit_kw
                + export_limit_kw
                + bess_max_power_kw
                + np.asarray(day['elec_load'], dtype=float)
                + float(eboiler_max_kw)
                + (p_wind + p_solar)
            )
            constraints += [
                p_cust_grid <= cp.multiply(max_surplus_kw, z_cust_short),
                p_ch + p_grid_sell + p_curtail <= cp.multiply(max_surplus_kw, 1 - z_cust_short),
            ]

            # (C6) E-boiler power cap
            if eboiler_enabled:
                constraints += [p_eboiler_elec <= eboiler_max_kw]
            else:
                constraints += [p_eboiler_elec == 0]

            # (C7) BESS constraints
            if b_mwh > 0:
                soc_init = cap_kwh * 0.5   # initial and terminal SOC = 50 %

                # FIX 2: SOC dynamics starting at t=0 (previously t=0 was omitted,
                # letting the solver charge/discharge at t=0 with no state cost).
                constraints += [
                    # Hour 0: SOC after t=0 action
                    soc[0] == soc_init + p_ch[0] * self.bess_eff - p_dis[0] / self.bess_eff,
                    # Terminal SOC equals initial SOC (cyclic daily operation)
                    soc[T - 1] == soc_init,
                    # SOC window
                    soc >= cap_kwh * 0.1,
                    soc <= cap_kwh * 0.9,
                    # Power limits
                    p_ch  <= bess_max_power_kw,
                    p_dis <= bess_max_power_kw,
                    # BESS can only charge from renewable surplus, not from grid purchase.
                    # Disable grid import for battery charging to avoid arbitrage.
                    p_grid_buy_proj == 0,
                    # (C7.1) Prevent simultaneous charge and discharge:
                    # If z_bess_charge[t]=1, then discharge p_dis[t]=0; if z_bess_charge[t]=0, then charge p_ch[t]=0
                    p_ch  <= bess_max_power_kw * z_bess_charge,
                    p_dis <= bess_max_power_kw * (1 - z_bess_charge),
                ]
                for t in range(1, T):
                    constraints.append(
                        soc[t] == soc[t - 1] + p_ch[t] * self.bess_eff - p_dis[t] / self.bess_eff
                    )
            else:
                constraints += [p_ch == 0, p_dis == 0, soc == 0, p_grid_buy_proj == 0, z_bess_charge == 0]

            # ---- objective ----
            # Galetech revenue streams
            # Only electricity physically supplied by the project to customer load is billed at PPA price.
            rev_customer = cp.sum(p_site_supply * (self.params['p_galetech'] * ppa_shock))
            rev_heat     = cp.sum(eboiler_eff * p_eboiler_elec * p_heat_sell)
            rev_grid     = cp.sum(p_grid_sell * self.params['p_sell'])

            # Galetech cost streams
            # NOTE: gas purchase cost and carbon cost are borne by the CUSTOMER, not Galetech.
            # They are therefore excluded from Galetech's financial objective.
            grid_purchase_cost = cp.sum(p_grid_buy_proj * p_grid_buy_price)
            customer_grid_penalty_cost = cp.sum(p_cust_grid * p_cust_grid_penalty)
            eboiler_cost       = cp.sum(eboiler_eff * p_eboiler_elec * eboiler_var_cost)
            bess_degradation   = cp.sum(p_dis * bess_wear_cost)

            # Carbon credit revenue to Galetech:
            # monetised CO2 reduction from
            # 1) replacing gas heat demand, and
            # 2) reducing grid electricity imports.
            gas_co2_saved_t = (
                (cp.sum(day['gas_load']) - cp.sum(p_gas_used))
                * self.gas_carbon_factor
                / 1000
            )
            # Electricity decarbonisation is credited to net RE displacement of grid power.
            # Keep this affine in the objective to satisfy DCP rules.
            net_re_displacement = cp.sum(p_site_supply) - cp.sum(p_grid_buy_proj)
            grid_co2_saved_t = net_re_displacement * grid_carbon_factor / 1000
            carbon_credit = (
                carbon_credit_share
                * self.params['p_carbon']
                * (gas_co2_saved_t + grid_co2_saved_t)
            )

            obj  = cp.Maximize(
                rev_customer + rev_heat + rev_grid
                - grid_purchase_cost
                - customer_grid_penalty_cost
                - eboiler_cost
                - bess_degradation
                + carbon_credit
            )
            prob = cp.Problem(obj, constraints)

            try:
                # Use GUROBI solver directly
                prob.solve(solver=cp.GUROBI, verbose=False)

                if prob.status not in ("infeasible", "unbounded") and prob.value is not None:
                    w = day['weight']
                    annual_revenue        += prob.value * w
                    gas_co2_saved = (
                        (np.sum(day['gas_load']) - np.sum(p_gas_used.value))
                        * self.gas_carbon_factor / 1000
                    )
                    grid_co2_saved = (
                        max(0.0, np.sum(p_site_supply.value) - np.sum(p_grid_buy_proj.value))
                        * grid_carbon_factor / 1000
                    )
                    annual_co2_saved      += (gas_co2_saved + grid_co2_saved) * w
                    annual_btm_supply_kwh += np.sum(p_site_supply.value) * w
                    annual_heat_from_elec += np.sum(eboiler_eff * p_eboiler_elec.value) * w
                    annual_gas_used_kwh   += np.sum(p_gas_used.value) * w
                    annual_curtailed_kwh  += np.sum(p_curtail.value) * w
                    annual_curtail_cost   += np.sum(p_curtail.value) * self.params['p_sell'] * w

                    if return_traces:
                        for t in range(T):
                            all_traces.append({
                                "Day_Type":        f"Scenario_{day_idx + 1}",
                                "Hour":            t,
                                "Elec_Demand_kW":  day['elec_load'][t],
                                "Gas_Demand_kWh":  day['gas_load'][t],  # kWh/hour thermal energy = kWth power
                                "Wind_Gen_kW":     p_wind[t],
                                "Solar_Gen_kW":    p_solar[t],
                                "BESS_SoC_kWh":    soc.value[t] if b_mwh > 0 else 0,
                                "BESS_Charge_kW":  p_ch.value[t] if b_mwh > 0 else 0,
                                "BESS_Discharge_kW": p_dis.value[t] if b_mwh > 0 else 0,
                                "EBoiler_Elec_kW": p_eboiler_elec.value[t],
                                "Gas_Used_kWh":    p_gas_used.value[t],  # thermal energy kWh/hour
                                "BTM_Supply_kW":   p_site_supply.value[t],
                                "Customer_Grid_Backup_kW": p_cust_grid.value[t],
                                "Grid_Export_kW":  p_grid_sell.value[t],
                                "Grid_Import_kW":  p_cust_grid.value[t] + p_grid_buy_proj.value[t],
                                "Project_Grid_Import_for_BESS_kW": p_grid_buy_proj.value[t],
                                "Curtailed_kW":    p_curtail.value[t],
                            })
                else:
                    # Detailed diagnostics for infeasible/unbounded problems
                    # Use deduplication to avoid spamming same error type multiple times
                    max_elec_demand = np.max(day['elec_load'])
                    max_gas_demand = np.max(day['gas_load'])
                    max_avail_wind = np.sum(p_wind)
                    max_avail_solar = np.sum(p_solar)
                    max_avail_renewable = max_avail_wind + max_avail_solar
                    
                    if prob.status == "infeasible":
                        # Only show detailed diagnosis for the first infeasible day encountered
                        diag_key = f"infeasible_day_{day_idx}"
                        if diag_key not in self._warned_solver_exceptions:
                            avg_demand = (np.sum(day['elec_load']) + np.sum(day['gas_load']) * eboiler_eff / 1000) / T
                            
                            # Additional analysis: check if max hourly load exceeds grid limit + renewable
                            grid_backup_capacity = grid_buy_limit_kw
                            max_hourly_supply = max_avail_renewable + grid_backup_capacity + (bess_max_power_kw if b_mwh > 0 else 0)
                            shortfall = max(0, max_elec_demand - max_hourly_supply)
                            
                            # Specific diagnostic for "no storage + high demand" scenario
                            diagnosis = (
                                f"**Diagnostic Analysis (Peak Hour):**\n"
                                f"- Max hourly elec demand: {max_elec_demand:.0f} kW\n"
                                f"- Available renewable: {max_avail_renewable:.0f} kW\n"
                                f"- Grid backup (p_cust_grid) limit: {grid_backup_capacity:.0f} kW\n"
                                f"- BESS discharge capacity: {bess_max_power_kw:.0f} kW\n"
                                f"- **Total supply capacity: {max_hourly_supply:.0f} kW**\n"
                            )
                            if shortfall > 0:
                                diagnosis += f"- **⚠️  SHORTFALL: {shortfall:.0f} kW** (demand exceeds all available supply sources)\n\n"
                                diagnosis += "**Primary issue**: Peak demand exceeds all available supply (renewable + grid + storage).\n"
                            else:
                                diagnosis += "\n"
                            
                            st.warning(
                                f"❌ Day {day_idx + 1}: **INFEASIBLE** solution detected.\n\n"
                                f"**Demand Profile:**\n"
                                f"- Avg load: {avg_demand:.1f} kW\n"
                                f"- Max load: {max_elec_demand:.0f} kW\n"
                                f"- Max heat demand: {max_gas_demand:.0f} kWth → {max_gas_demand * eboiler_eff / 1000:.0f} kW equiv.\n\n"
                                f"{diagnosis}"
                                f"**Recommended fixes:**\n"
                                f"1) 📈 **Increase renewable capacity**: More solar/wind MW\n"
                                f"2) 🔋 **Add battery storage**: BESS shifts demand across hours\n"
                                f"3) 📡 **Increase grid import limit**: Raise 'grid_buy_limit_kw' if available grid capacity supports it\n"
                                f"4) 💨 **Add wind turbines**: Generates when solar doesn't (nights, cloudy)\n"
                                f"5) 🌡️ **Reduce heating demand**: Use gas boiler backup instead of E-boiler\n\n"
                                f"Config: {t_model} ×{t_count}, Solar={s_mw} MW, BESS={b_mwh} MWh"
                            )
                            self._warned_solver_exceptions.add(diag_key)
                    else:
                        # For other statuses (unbounded, etc.), show compact message only once per status type
                        status_key = f"solver_status_{prob.status}"
                        if status_key not in self._warned_solver_exceptions:
                            st.warning(
                                f"⚠️  Day {day_idx + 1}: LP status = {prob.status}. "
                                f"Config: {t_model} ×{t_count}, Solar={s_mw} MW, BESS={b_mwh} MWh"
                            )
                            self._warned_solver_exceptions.add(status_key)
            except Exception as e:
                # Avoid spamming repeated identical exception lines across many candidate configs.
                err_key = str(e)
                if err_key not in self._warned_solver_exceptions:
                    st.warning(f"Day {day_idx + 1}: solver exception — {e}")
                    self._warned_solver_exceptions.add(err_key)

        # Safety: always return the correct type
        if return_traces:
            return pd.DataFrame(all_traces) if all_traces else pd.DataFrame()
        
        # Return 7-tuple in all cases
        return (
            annual_revenue,
            annual_co2_saved,
            annual_btm_supply_kwh,
            annual_heat_from_elec,
            annual_gas_used_kwh,
            annual_curtailed_kwh,
            annual_curtail_cost,
        )

    # ------------------------------------------------------------------
    # Two-stage capacity optimisation
    # Stage 1: coarse grid sweep to find the promising region
    # Stage 2: fine search around the Stage 1 best
    # ------------------------------------------------------------------
    def two_stage_optimization(self, rep_days, turbine_choices,
                               min_turbines, max_turbines,
                               max_solar, min_bess, max_bess,
                               site_area_acre,
                               optimization_metric='Payback',
                               min_eboiler_kw=0, max_eboiler_kw=0):

        annual_customer_load_kwh = sum(np.sum(day['elec_load']) * day['weight'] for day in rep_days)

        # ---- Stage 1 ----
        t_start = max(min_turbines, 0)
        b_start = max(min_bess, 0)

        t_count_steps = range(t_start, max_turbines + 1, max(1, (max_turbines - t_start) // 3)) \
                        if max_turbines >= t_start else [t_start]
        s_steps       = range(0, max_solar + 1, max(1, max_solar // 3)) \
                        if max_solar > 0 else [0]
        b_steps       = range(b_start, max_bess + 1, max(1, (max_bess - b_start) // 3)) \
                        if max_bess >= b_start else [b_start]
        if self.params.get('enable_e_boiler', True):
            if max_eboiler_kw <= min_eboiler_kw:
                e_steps = [int(max_eboiler_kw)]
            else:
                e_step = max(100, (max_eboiler_kw - min_eboiler_kw) // 3)
                e_steps = list(range(int(min_eboiler_kw), int(max_eboiler_kw) + 1, int(e_step)))
                if e_steps[-1] != int(max_eboiler_kw):
                    e_steps.append(int(max_eboiler_kw))
        else:
            e_steps = [0]

        stage1_results = []
        stage1_skipped_feasibility = 0  # Configs skipped by feasibility pre-check
        stage1_failed_infeasible = 0     # Configs that failed LP solve (infeasible)
        
        for t_model in turbine_choices:
            counts = [0] if t_model == 'None' else list(t_count_steps)
            for t_count in counts:
                if t_count < min_turbines:
                    continue
                for s in s_steps:
                    for b in b_steps:
                        for e_kw in e_steps:
                            if b < min_bess:
                                continue
                            if t_count == 0 and s == 0:
                                continue  # 禁止无风无光配置（有无电池皆禁）
                            # New land rule: every 5 acres allows up to 2 turbines and 1 MW PV.
                            # Combined form: t_count/2 + s <= site_area_acre/5.
                            if (t_count / 2.0 + s) > (site_area_acre / 5.0):
                                continue

                            # ---- Quick feasibility pre-check ----
                            # Before running expensive LP solve, verify that max available supply >= max demand
                            # This avoids solving infeasible problems that configuration can never meet.
                            max_feasible_supply = 0.0
                            max_demand = 0.0
                            
                            for day in rep_days:
                                # Max wind generation in this day
                                p_wind_max = self.get_wind_power(day['wind_speed'], t_model, t_count).max() if t_model != "None" else 0.0
                                # Max solar generation in this day
                                if s > 0:
                                    irr_wm2 = np.clip(day['irradiance'], 0.0, 1400.0)
                                    eta = max(self.params['solar_efficiency'], 1e-6)
                                    pv_land_area_m2 = s * 5.0 * self.acre_to_m2
                                    p_solar_area_kw = irr_wm2 * pv_land_area_m2 * eta * min(1.0, 1000.0 / (eta * 5.0 * self.acre_to_m2)) / 1000.0
                                    p_solar_max = min(p_solar_area_kw.max(), s * 1000.0)
                                else:
                                    p_solar_max = 0.0
                                
                                # Max supply in this day: renewable + grid limit + battery discharge limit
                                grid_limit = self.params.get('grid_buy_limit_kw', 5000)
                                bess_discharge_max = (b * 1000 * 0.5) if b > 0 else 0.0  # assumes 2h discharge time
                                max_supply_day = p_wind_max + p_solar_max + grid_limit + bess_discharge_max
                                
                                # Max demand in this day (elec + heat converted to elec)
                                eboiler_eff = self.params.get('eboiler_efficiency', 1.0)
                                max_elec_demand = np.max(day['elec_load'])
                                max_heat_demand_as_elec = np.max(day['gas_load']) * eboiler_eff / 1000.0  # kWth to kW equiv
                                max_demand_day = max_elec_demand + max_heat_demand_as_elec
                                
                                max_feasible_supply = max(max_feasible_supply, max_supply_day)
                                max_demand = max(max_demand, max_demand_day)
                            
                            # If peak demand exceeds peak supply capacity by >5% margin, skip this config
                            # (small margin to account for rounding and model approximations)
                            if max_demand > max_feasible_supply * 1.05:
                                stage1_skipped_feasibility += 1
                                continue

                            capex, _, _, _, _ = self.get_capex(t_model, t_count, s, b, eboiler_kw=e_kw)
                            annual_opex, _, _, _ = self.get_opex(capex, t_count=t_count, s_mw=s)
                            out = self.evaluate_combination(t_model, t_count, s, b, rep_days, eboiler_kw=e_kw)

                            # Robust handling of return value
                            if out is None or isinstance(out, pd.DataFrame):
                                stage1_failed_infeasible += 1
                                continue
                            if not isinstance(out, tuple) or len(out) != 7:
                                stage1_failed_infeasible += 1
                                continue

                            revenue, co2, btm_e, heat_elec, gas_used, curt, curt_cost = out
                            net_profit = revenue - annual_opex
                            irr     = self._safe_irr(capex, net_profit)
                            npv     = npf.npv(self.discount_rate, [-capex] + [net_profit] * self.project_life_years)
                            payback = capex / net_profit if net_profit > 0 else 99.0
                            # Carbon credit is fully retained by the energy system.
                            carbon_share = self.params.get('carbon_credit_share', 1.0)
                            heat_sales_k = heat_elec * self.params.get('p_heat_sell', 0.0) / 1e3
                            cust_gas_savings_k = heat_elec * (self.params['p_gas'] - self.params.get('p_heat_sell', 0.0)) / 1e3
                            galetech_carbon_credit_k = co2 * self.params['p_carbon'] * carbon_share / 1e3
                            cust_carbon_savings_k = co2 * self.params['p_carbon'] * (1 - carbon_share) / 1e3

                            stage1_results.append({
                                'Turbine':                  f"{t_count}x {t_model}" if t_count > 0 else 'No Wind',
                                't_model_raw':              t_model,
                                't_count_raw':              t_count,
                                'Solar_MW':                 s,
                                'BESS_MWh':                 b,
                                'EBoiler_kW':               e_kw,
                                'CAPEX_M':                  capex / 1e6,
                                'OPEX_k':                   annual_opex / 1e3,
                                'Profit_k':                 net_profit / 1e3,
                                'Payback':                  payback,
                                'IRR':                      irr * 100,
                                'NPV10_M':                  npv / 1e6,
                                'CO2_T':                    co2,
                                'Elec_Offset_MWh':          btm_e / 1000,
                                'Green_Elec_Share_pct':     (btm_e / annual_customer_load_kwh * 100) if annual_customer_load_kwh > 0 else 0.0,
                                'Heat_By_EBoiler_MWh':      heat_elec / 1000,
                                'Gas_Used_MWh':             gas_used / 1000,
                                'Curtailed_MWh':            curt / 1000,
                                'Min_Elec_PPA':             0.0,
                                'Heat_Sales_k':             heat_sales_k,
                                'Galetech_Carbon_Credit_k': galetech_carbon_credit_k,
                                'Cust_Gas_Savings_k':       cust_gas_savings_k,
                                'Cust_Carbon_Savings_k':    cust_carbon_savings_k,
                            })

        df_stage1 = pd.DataFrame(stage1_results)
        if df_stage1.empty:
            # Return diagnostic info in the DataFrame so UI can explain why
            diag_df = pd.DataFrame([{
                'error_type': 'stage1_empty',
                'skipped_feasibility': stage1_skipped_feasibility,
                'failed_infeasible': stage1_failed_infeasible,
            }])
            return diag_df, None

        if optimization_metric == 'IRR':
            best_stage1 = df_stage1.sort_values('IRR', ascending=False).iloc[0]
        elif optimization_metric == 'NPV':
            best_stage1 = df_stage1.sort_values('NPV10_M', ascending=False).iloc[0]
        else:
            best_stage1 = df_stage1.sort_values('Payback', ascending=True).iloc[0]

        # ---- Stage 2: fine search ----
        best_t = int(best_stage1['t_count_raw'])
        best_s = int(best_stage1['Solar_MW'])
        best_b = int(best_stage1['BESS_MWh'])
        best_e = int(best_stage1['EBoiler_kW']) if 'EBoiler_kW' in best_stage1 else 0

        candidates = set()
        for t_model in turbine_choices:
            t_range = [0] if t_model == 'None' else list({
                max(min_turbines, best_t - 1), best_t, min(max_turbines, best_t + 1)
            })
            for t_count in t_range:
                if t_count < min_turbines:
                    continue
                for s in {max(0, best_s - 1), best_s, min(max_solar, best_s + 1)}:
                    for b in {max(min_bess, best_b - 1), best_b, min(max_bess, best_b + 1)}:
                        for e_kw in {
                            max(int(min_eboiler_kw), best_e - 500),
                            best_e,
                            min(int(max_eboiler_kw), best_e + 500)
                        }:
                            if b < min_bess:
                                continue
                            if t_count == 0 and s == 0:
                                continue  # 禁止无风无光配置（有无电池皆禁）
                            if (t_count / 2.0 + s) > (site_area_acre / 5.0):
                                continue
                            candidates.add((t_model, t_count, s, b, e_kw))

        final_results = []
        for t_model, t_count, s, b, e_kw in candidates:
            capex, _, _, _, _ = self.get_capex(t_model, t_count, s, b, eboiler_kw=e_kw)
            annual_opex, _, _, _ = self.get_opex(capex, t_count=t_count, s_mw=s)
            out = self.evaluate_combination(t_model, t_count, s, b, rep_days, eboiler_kw=e_kw)
            
            # Robust handling of return value
            if out is None or isinstance(out, pd.DataFrame):
                continue
            if not isinstance(out, tuple) or len(out) != 7:
                continue
                
            revenue, co2, btm_e, heat_elec, gas_used, curt, curt_cost = out
            net_profit = revenue - annual_opex
            irr     = self._safe_irr(capex, net_profit)
            npv     = npf.npv(self.discount_rate, [-capex] + [net_profit] * self.project_life_years)
            payback = capex / net_profit if net_profit > 0 else 99.0
            # Carbon credit is fully retained by the energy system.
            carbon_share = self.params.get('carbon_credit_share', 1.0)
            heat_sales_k = heat_elec * self.params.get('p_heat_sell', 0.0) / 1e3
            cust_gas_savings_k = heat_elec * (self.params['p_gas'] - self.params.get('p_heat_sell', 0.0)) / 1e3
            galetech_carbon_credit_k = co2 * self.params['p_carbon'] * carbon_share / 1e3
            cust_carbon_savings_k = co2 * self.params['p_carbon'] * (1 - carbon_share) / 1e3

            final_results.append({
                'Turbine':                  f"{t_count}x {t_model}" if t_count > 0 else 'No Wind',
                't_model_raw':              t_model,
                't_count_raw':              t_count,
                'Solar_MW':                 s,
                'BESS_MWh':                 b,
                'EBoiler_kW':               e_kw,
                'CAPEX_M':                  capex / 1e6,
                'OPEX_k':                   annual_opex / 1e3,
                'Profit_k':                 net_profit / 1e3,
                'Payback':                  payback,
                'IRR':                      irr * 100,
                'NPV10_M':                  npv / 1e6,
                'CO2_T':                    co2,
                'Elec_Offset_MWh':          btm_e / 1000,
                'Green_Elec_Share_pct':     (btm_e / annual_customer_load_kwh * 100) if annual_customer_load_kwh > 0 else 0.0,
                'Heat_By_EBoiler_MWh':      heat_elec / 1000,
                'Gas_Used_MWh':             gas_used / 1000,
                'Curtailed_MWh':            curt / 1000,
                'Min_Elec_PPA':             0.0,
                'Heat_Sales_k':             heat_sales_k,
                'Galetech_Carbon_Credit_k': galetech_carbon_credit_k,
                'Cust_Gas_Savings_k':       cust_gas_savings_k,
                'Cust_Carbon_Savings_k':    cust_carbon_savings_k,
            })

        df_final = pd.DataFrame(final_results)
        if df_final.empty:
            return df_stage1, best_stage1

        if optimization_metric == 'IRR':
            best_final = df_final.sort_values('IRR', ascending=False).iloc[0]
        elif optimization_metric == 'NPV':
            best_final = df_final.sort_values('NPV10_M', ascending=False).iloc[0]
        else:
            best_final = df_final.sort_values('Payback', ascending=True).iloc[0]

        return df_final, best_final


# ==========================================
# 2. Data preparation helpers
# ==========================================
def get_synthetic_weather_profiles():
    return [
        {
            'label': 'Summer',
            'wind_speed': 7 + 2.5 * np.sin((np.arange(24) - 3) * np.pi / 12),
            'irradiance': np.array([
                0, 0, 0, 0, 20, 100, 250, 450, 700, 900, 1000, 1050,
                1050, 980, 850, 650, 400, 180, 60, 10, 0, 0, 0, 0,
            ], dtype=float),
        },
        {
            'label': 'Winter',
            'wind_speed': 9 + 3.0 * np.sin((np.arange(24) + 1) * np.pi / 12),
            'irradiance': np.array([
                0, 0, 0, 0, 15, 80, 200, 380, 620, 820, 900, 950,
                950, 900, 760, 560, 320, 130, 40, 5, 0, 0, 0, 0,
            ], dtype=float),
        },
        {
            'label': 'Spring/Autumn',
            'wind_speed': 8 + 2.0 * np.cos((np.arange(24) - 2) * np.pi / 12),
            'irradiance': np.array([
                0, 0, 0, 0, 18, 90, 220, 410, 660, 860, 960, 1010,
                1010, 940, 800, 600, 350, 150, 50, 8, 0, 0, 0, 0,
            ], dtype=float),
        },
    ]


def get_weather_profile_by_index(weather_profiles, index):
    profiles = weather_profiles if weather_profiles else get_synthetic_weather_profiles()
    return profiles[index % len(profiles)]


def get_average_weather_profile(weather_profiles=None):
    profiles = weather_profiles if weather_profiles else get_synthetic_weather_profiles()
    return {
        'label': 'Average of Seasonal Profiles',
        'wind_speed': np.mean([profile['wind_speed'] for profile in profiles], axis=0),
        'irradiance': np.mean([profile['irradiance'] for profile in profiles], axis=0),
    }


def build_typical_weather_profiles(hourly_weather_df):
    seasonal_months = [
        ('Summer', [6, 7, 8]),
        ('Winter', [12, 1, 2]),
        ('Spring/Autumn', [3, 4, 5, 9, 10, 11]),
    ]
    fallback_profiles = {p['label']: p for p in get_synthetic_weather_profiles()}
    profiles = []

    weather_df = hourly_weather_df.copy()
    weather_df['timestamp'] = pd.to_datetime(weather_df['time'])
    weather_df['month'] = weather_df['timestamp'].dt.month
    weather_df['hour'] = weather_df['timestamp'].dt.hour

    for label, months in seasonal_months:
        seasonal_df = weather_df[weather_df['month'].isin(months)]
        if seasonal_df.empty:
            profiles.append(fallback_profiles[label])
            continue

        hourly_average = (
            seasonal_df.groupby('hour')[['wind_speed_10m', 'irradiance']]
            .mean()
            .reindex(range(24))
            .interpolate(limit_direction='both')
            .fillna(0.0)
        )
        profiles.append({
            'label': label,
            'wind_speed': hourly_average['wind_speed_10m'].to_numpy(dtype=float),
            'irradiance': np.clip(hourly_average['irradiance'].to_numpy(dtype=float), 0.0, None),
        })

    return profiles


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_json(url, params):
    request_url = f"{url}?{urlencode(params)}"
    try:
        with urlopen(request_url, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace')
        try:
            error_payload = json.loads(error_body)
            reason = error_payload.get('reason') or error_payload.get('message') or error_body
        except json.JSONDecodeError:
            reason = error_body or str(exc)
        raise ValueError(f"Weather API request failed: {reason}") from exc
    except URLError as exc:
        raise ValueError(f"Weather API request failed: {exc.reason}") from exc


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_last_year_typical_weather(location_query, use_tilted_gti=True):
    """
    Fetch typical weather for the previous year from Open-Meteo.
    
    Args:
        location_query: Location name to search for
        use_tilted_gti: If True, fetch Global Tilted Irradiance (GTI) at 30° tilt,
                        SE (-45°) + SW (45°) averaged. If False, use horizontal
                        shortwave radiation (GHI).
    """
    target_year = date.today().year - 1
    geo_data = fetch_json(
        'https://geocoding-api.open-meteo.com/v1/search',
        {'name': location_query, 'count': 1, 'language': 'en', 'format': 'json'},
    )
    if not geo_data.get('results'):
        raise ValueError(f"No location match found for '{location_query}'.")

    location = geo_data['results'][0]
    
    # Option 1: Fetch tilted GTI for SE and SW orientations
    if use_tilted_gti:
        try:
            # Open-Meteo azimuth uses 0° = south, -90° = east, 90° = west.
            # Therefore SE = -45° and SW = 45°.
            # Fetch SE panel (azimuth -45°, tilt 30°)
            archive_se = fetch_json(
                'https://archive-api.open-meteo.com/v1/archive',
                {
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'start_date': f'{target_year}-01-01',
                    'end_date': f'{target_year}-12-31',
                    'hourly': 'wind_speed_10m,global_tilted_irradiance',
                    'tilt': 30,
                    'azimuth': -45,
                    'wind_speed_unit': 'ms',
                    'timezone': 'auto',
                },
            )
            # Fetch SW panel (azimuth 45°, tilt 30°)
            archive_sw = fetch_json(
                'https://archive-api.open-meteo.com/v1/archive',
                {
                    'latitude': location['latitude'],
                    'longitude': location['longitude'],
                    'start_date': f'{target_year}-01-01',
                    'end_date': f'{target_year}-12-31',
                    'hourly': 'wind_speed_10m,global_tilted_irradiance',
                    'tilt': 30,
                    'azimuth': 45,
                    'wind_speed_unit': 'ms',
                    'timezone': 'auto',
                },
            )
            
            hourly_se = archive_se.get('hourly', {})
            hourly_sw = archive_sw.get('hourly', {})
            
            if not hourly_se or not hourly_sw or 'time' not in hourly_se:
                raise ValueError('GTI data unavailable; falling back to GHI.')
            
            # Average SE and SW irradiance
            gti_se = np.array(hourly_se.get('global_tilted_irradiance', []))
            gti_sw = np.array(hourly_sw.get('global_tilted_irradiance', []))
            irradiance_data = (gti_se + gti_sw) / 2.0
            
            hourly_weather_df = pd.DataFrame({
                'time': hourly_se['time'],
                'wind_speed_10m': hourly_se.get('wind_speed_10m', []),
                'irradiance': irradiance_data,
            })
            irradiance_source = 'Open-Meteo GTI (SE -45° + SW 45°, 30° tilt, averaged)'
            
        except (ValueError, KeyError):
            # Fallback to horizontal shortwave radiation if GTI unavailable
            st.warning('GTI data unavailable; using horizontal shortwave radiation instead.')
            use_tilted_gti = False
    
    # Option 2: Fallback to horizontal shortwave radiation (GHI)
    if not use_tilted_gti:
        archive_data = fetch_json(
            'https://archive-api.open-meteo.com/v1/archive',
            {
                'latitude': location['latitude'],
                'longitude': location['longitude'],
                'start_date': f'{target_year}-01-01',
                'end_date': f'{target_year}-12-31',
                'hourly': 'wind_speed_10m,shortwave_radiation',
                'wind_speed_unit': 'ms',
                'timezone': 'auto',
            },
        )
        hourly = archive_data.get('hourly', {})
        if not hourly or 'time' not in hourly:
            raise ValueError('Historical weather data is unavailable for the selected location.')

        hourly_weather_df = pd.DataFrame({
            'time': hourly['time'],
            'wind_speed_10m': hourly.get('wind_speed_10m', []),
            'irradiance': hourly.get('shortwave_radiation', []),
        })
        irradiance_source = 'Open-Meteo GHI (horizontal shortwave)'
    
    if hourly_weather_df.empty or len(hourly_weather_df) < 24:
        raise ValueError('Insufficient hourly weather data returned by the weather service.')

    return {
        'location_label': ', '.join(filter(None, [
            location.get('name'),
            location.get('admin1'),
            location.get('country'),
        ])),
        'latitude': location['latitude'],
        'longitude': location['longitude'],
        'source': irradiance_source,
        'year': target_year,
        'profiles': build_typical_weather_profiles(hourly_weather_df),
    }


def load_custom_typical_days(df=None, custom_weights=None, show_warnings=True, weather_profiles=None):
    """
    Build representative daily load profiles (electricity + gas) from
    an uploaded CSV, or fall back to three synthetic typical days.
    Wind / solar generation profiles are embedded separately.
    """
    rep_days = []
    if isinstance(weather_profiles, dict) and 'profiles' in weather_profiles:
        weather_defaults = weather_profiles.get('profiles', get_synthetic_weather_profiles())
    elif isinstance(weather_profiles, list):
        weather_defaults = weather_profiles
    else:
        weather_defaults = get_synthetic_weather_profiles()

    if df is not None:
        num_days = len(df) // 24
        average_weather_profile = get_average_weather_profile(weather_defaults)
        for i in range(num_days):
            day_data = df.iloc[i * 24 : (i + 1) * 24]
            weather_profile = (
                average_weather_profile
                if num_days == 1
                else get_weather_profile_by_index(weather_defaults, i)
            )

            if 'elec_load' in df.columns:
                elec_load = day_data['elec_load'].values.astype(float)
            else:
                if show_warnings:
                    st.warning("Column 'elec_load' not found — using default profile.")
                t = np.arange(24)
                elec_load = 1200 + 400 * np.sin((t - 8) * np.pi / 12)

            if 'gas_load' in df.columns:
                # gas_load is expected in kWh (thermal energy per hour)
                # In hourly dispatch model, this is treated as kWth (thermal power)
                gas_load = day_data['gas_load'].values.astype(float)
            else:
                if show_warnings:
                    st.warning("Column 'gas_load' not found — using zero thermal load (kWh/hour).")
                gas_load = np.zeros(24, dtype=float)

            if 'wind_speed' in df.columns:
                wind_speed = day_data['wind_speed'].values.astype(float)
            else:
                if show_warnings:
                    st.warning("Column 'wind_speed' not found — using default weather wind-speed profile.")
                wind_speed = weather_profile['wind_speed']

            if 'irradiance' in df.columns:
                irradiance = day_data['irradiance'].values.astype(float)
            else:
                if show_warnings:
                    st.warning("Column 'irradiance' not found — using default weather irradiance profile.")
                irradiance = weather_profile['irradiance']

            weight = custom_weights[i] if custom_weights and i < len(custom_weights) else 365 // num_days
            rep_days.append({
                'label': weather_profile.get('label', f'Day {i + 1}'),
                'elec_load': elec_load,
                'gas_load': gas_load,
                'wind_speed': wind_speed,
                'irradiance': irradiance,
                'weight': weight,
            })
    else:
        # Synthetic three-day profiles
        weights = custom_weights if custom_weights else [90, 90, 185]
        t = np.arange(24)
        summer_weather = get_weather_profile_by_index(weather_defaults, 0)
        winter_weather = get_weather_profile_by_index(weather_defaults, 1)
        shoulder_weather = get_weather_profile_by_index(weather_defaults, 2)
        rep_days.append({
            'label': summer_weather.get('label', 'Summer'),
            'elec_load': 1200 + 400 * np.sin((t - 8)  * np.pi / 12),
            'gas_load':  200  +  50 * np.random.rand(24),
            'wind_speed': summer_weather['wind_speed'],
            'irradiance': summer_weather['irradiance'],
            'weight':    weights[0] if len(weights) > 0 else 90,
        })
        rep_days.append({
            'label': winter_weather.get('label', 'Winter'),
            'elec_load': 800  + 200 * np.sin((t - 6)  * np.pi / 12),
            'gas_load':  2000 + 800 * np.cos((t - 12) * np.pi / 12),
            'wind_speed': winter_weather['wind_speed'],
            'irradiance': winter_weather['irradiance'],
            'weight':    weights[1] if len(weights) > 1 else 90,
        })
        rep_days.append({
            'label': shoulder_weather.get('label', 'Spring/Autumn'),
            'elec_load': 900  + 150 * np.sin(t / 4),
            'gas_load':  600  + 100 * np.random.rand(24),
            'wind_speed': shoulder_weather['wind_speed'],
            'irradiance': shoulder_weather['irradiance'],
            'weight':    weights[2] if len(weights) > 2 else 185,
        })
    return rep_days


# ==========================================
# 3. Streamlit UI
# ==========================================

def render_pre_run_preview(rep_days, solar_efficiency, weather_profiles=None):
    """Render pre-optimisation typical-day load and generation preview charts.

    Figure 1 (demand): always uses rep_days directly — actual uploaded data if available,
    otherwise the 3 seasonal profiles from load_custom_typical_days.
    Figures 2 & 3 (weather / generation): always show exactly 3 seasonal lines from weather_profiles.
    Weather data MUST come from fetched Open-Meteo data, not synthetic fallback.
    """
    if not rep_days:
        return

    preview_optimizer = GaletechAssetOptimizer({'solar_efficiency': solar_efficiency})
    hours = np.arange(24)

    # --- Build 3 seasonal weather profiles for Figures 2 & 3 ---
    # ONLY use fetched weather data, NO synthetic fallback
    if isinstance(weather_profiles, list):
        profiles = weather_profiles
    elif isinstance(weather_profiles, dict) and 'profiles' in weather_profiles:
        profiles = weather_profiles['profiles']
    else:
        st.error(
            "❌ Preview charts require fetched weather data from Open-Meteo. "
            "Weather profiles not available. Please ensure weather data has been successfully fetched."
        )
        return

    season_days = [
        {
            'label':      p.get('label', f'Season {i + 1}'),
            'wind_speed': p['wind_speed'],
            'irradiance': p['irradiance'],
        }
        for i, p in enumerate(profiles[:3])
    ]

    st.subheader("Pre-Optimisation Typical Day Preview")
    st.caption(
        "Demand chart shows uploaded representative day profiles (or default seasonal profiles if no file is uploaded). "
        "Weather and generation charts show 3 seasonal profiles (Summer / Winter / Spring-Autumn) "
        "built from the previous full year's hourly Open-Meteo data by seasonal hour-of-day averaging. "
        "Reference generation: 1 x EWT 1MW turbine + 1 MW solar PV."
    )

    # Figure 1: electricity and heat demand — use rep_days directly (uploaded data or synthetic fallback)
    fig_load, ax_load = plt.subplots(figsize=(10, 4))
    for i, day in enumerate(rep_days, start=1):
        day_label = day.get('label', f'Day {i}')
        ax_load.plot(hours, day['elec_load'], linewidth=2, label=f"{day_label} Electricity Demand")
        ax_load.plot(hours, day['gas_load'], linestyle='--', linewidth=1.8, label=f"{day_label} Heat Demand")
    ax_load.set_title("Representative Day — Electricity & Heat Demand Profiles")
    ax_load.set_xlabel("Hour of Day")
    ax_load.set_ylabel("Power (kW) / Energy (kWh per hour)")
    ax_load.set_xticks(np.arange(0, 24, 2))
    ax_load.grid(alpha=0.25)
    ax_load.legend(ncol=2, fontsize=8)
    st.pyplot(fig_load)
    plt.close(fig_load)

    # Figure 2: weather input curves — 3 seasonal lines, no annual average
    fig_weather, (ax_wind, ax_irr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for day in season_days:
        ax_wind.plot(hours, day['wind_speed'], linewidth=2, label=f"{day['label']} 10 m Wind Speed")
        ax_irr.plot(hours, day['irradiance'], linewidth=2, label=f"{day['label']} Irradiance (W/m²)")

    ax_wind.set_title("Seasonal Typical Day — 10 m Wind Speed (m/s)")
    ax_wind.set_ylabel("Wind Speed (m/s)")
    ax_wind.grid(alpha=0.25)
    ax_wind.legend(ncol=2, fontsize=8)

    ax_irr.set_title("Seasonal Typical Day — Tilted Irradiance (W/m²)")
    ax_irr.set_xlabel("Hour")
    ax_irr.set_ylabel("Irradiance (W/m²)")
    ax_irr.set_xticks(np.arange(0, 24, 2))
    ax_irr.grid(alpha=0.25)
    ax_irr.legend(ncol=2, fontsize=8)
    fig_weather.tight_layout()
    st.pyplot(fig_weather)
    plt.close(fig_weather)

    # Figure 3: seasonal wind & solar reference output — 3 seasonal lines
    fig_gen, ax_gen = plt.subplots(figsize=(10, 4))
    eta = max(float(solar_efficiency), 1e-6)
    pv_land_utilization = min(1.0, 1000.0 / (eta * 5.0 * preview_optimizer.acre_to_m2))
    solar_land_area_m2 = 1.0 * 5.0 * preview_optimizer.acre_to_m2

    for day in season_days:
        p_wind_ref = preview_optimizer.get_wind_power(day['wind_speed'], "EWT 1MW", count=1, shock_factor=1.0)
        irr_wm2 = np.clip(day['irradiance'], 0.0, 1400.0)
        p_solar_area_kw = irr_wm2 * solar_land_area_m2 * eta * pv_land_utilization / 1000.0
        p_solar_ref = np.minimum(p_solar_area_kw, 1000.0)
        ax_gen.plot(hours, p_wind_ref, linewidth=2,       label=f"{day['label']} Wind Output")
        ax_gen.plot(hours, p_solar_ref, linestyle='--', linewidth=1.8, label=f"{day['label']} Solar Output")

    ax_gen.set_title("Seasonal Typical Day — Wind & Solar Reference Output (kW)  [1 × EWT 1MW + 1 MW PV]")
    ax_gen.set_xlabel("Hour")
    ax_gen.set_ylabel("Power (kW)")
    ax_gen.set_xticks(np.arange(0, 24, 2))
    ax_gen.grid(alpha=0.25)
    ax_gen.legend(ncol=2, fontsize=8)
    st.pyplot(fig_gen)
    plt.close(fig_gen)


st.set_page_config(page_title="Galetech BOO Bankable Report", layout="wide")
st.title("📑 Galetech BOO Optimiser & Bankability Assistant")

with st.sidebar:
    st.header("📂 Data & Profile Setup")
    st.info(
        "Upload hourly CSV/Excel with columns `elec_load`, `wind_speed`, `irradiance` "
        "(optional: `gas_load`), one row per hour.\n\n"
        "**Units:**\n"
        "- `elec_load`: kW (power)\n"
        "- `gas_load`: kWh (thermal energy per hour, treated as kWth power in hourly model)\n"
        "- `wind_speed`: m/s\n"
        "- `irradiance`: W/m² (absolute value)\n\n"
        "Renewable output is calculated from wind speed and irradiance."
    )
    uploaded_file = st.file_uploader(
        "Customer Hourly Load Profile",
        type=['csv', 'xlsx'],
    )
    st.caption(
        "If weather columns are missing, the app can use fetched default climate profiles for the chosen location. "
        "If no location is selected, Dublin is used by default."
    )
    weather_location_query = st.text_input(
        "Weather location (city or place)",
        value=st.session_state.get('weather_location_query', 'Dublin'),
        help="Used to fetch last year's 10 m wind speed and surface irradiance as default weather data.",
    )
    fetch_weather_btn = st.button("🌤️ Fetch Last-Year Weather Defaults", use_container_width=True)
    df_customer   = None
    custom_weights = []

    if fetch_weather_btn:
        if not weather_location_query.strip():
            st.error("Please enter a city or location before fetching weather data.")
        else:
            with st.spinner("Fetching historical weather data and building typical-day weather profiles..."):
                try:
                    weather_payload = fetch_last_year_typical_weather(weather_location_query.strip())
                    st.session_state['weather_defaults'] = weather_payload
                    st.session_state['weather_location_query'] = weather_location_query.strip()
                    st.success(
                        f"Weather defaults loaded for {weather_payload['location_label']} "
                        f"using {weather_payload['year']} historical data."
                    )
                except Exception as exc:
                    st.error(f"Unable to fetch weather defaults: {exc}")

    # Auto-default weather when user has not fetched any location yet.
    if 'weather_defaults' not in st.session_state:
        default_location = (st.session_state.get('weather_location_query', 'Dublin') or 'Dublin').strip() or 'Dublin'
        st.info(f"⏳ Fetching weather data from Open-Meteo for {default_location}...")
        try:
            weather_payload = fetch_last_year_typical_weather(default_location)
            st.session_state['weather_defaults'] = weather_payload
            st.session_state['weather_location_query'] = default_location
            st.success(
                f"✓ Weather defaults loaded for {weather_payload['location_label']} "
                f"using {weather_payload['year']} historical data (Open-Meteo)."
            )
        except Exception as exc:
            st.error(
                f"❌ Unable to fetch weather data from Open-Meteo for '{default_location}': {exc}\n\n"
                f"Please enter a valid city name and click '🌤️ Fetch Last-Year Weather Defaults' to proceed. "
                f"The app requires real historical weather data to ensure accurate modeling."
            )
            st.stop()

    if 'weather_defaults' in st.session_state:
        weather_info = st.session_state['weather_defaults']
        st.info(
            f"Default weather source: {weather_info['location_label']} | "
            f"{weather_info['source']} | {weather_info['year']} | "
            f"Representative days built from full-year hourly data by seasonal hour-of-day averaging"
        )

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df_customer = pd.read_excel(uploaded_file)
        else:
            df_customer = pd.read_csv(uploaded_file)
        num_days = len(df_customer) // 24
        if num_days == 0:
            st.error("File must contain at least 24 rows (1 full day).")
        for i in range(num_days):
            custom_weights.append(
                st.number_input(f"Day {i + 1} weight (days/year):",
                                min_value=1, max_value=365, value=365 // num_days)
            )
    else:
        st.info("Using demo data (3 synthetic typical days)")
        custom_weights = [
            st.number_input("Summer weight:",      value=90),
            st.number_input("Winter weight:",      value=90),
            st.number_input("Spring/Autumn weight:", value=185),
        ]

    st.divider()
    st.header("💶 Commercial & Pricing")
    p_retail_input   = st.number_input("Customer current electricity price (€/MWh)", value=130.0)
    p_gas_input      = st.number_input(
        "Customer gas price (€/MWh) — borne by customer; used for customer savings reporting",
        value=50.0,
    )
    p_galetech_input = st.number_input("Target BOO PPA price — electricity (€/MWh)", value=100.0)
    p_sell_input     = st.number_input("Grid export price (€/MWh)",                  value=100.0)
    cust_grid_penalty_input = st.number_input(
        "Customer grid-backup penalty (€/MWh)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        help="Optional soft constraint (default 0): use only if you want extra service-priority pressure beyond pure price signals.",
    )
    p_carbon_input   = st.number_input("Carbon value (€/tonne CO₂)",                 value=65.0)
    target_irr_input = st.number_input("Target IRR for minimum PPA (%)",             value=10.0)

    use_e_boiler          = st.checkbox("Enable electric boiler option", value=True)
    eboiler_eff_input     = st.number_input("Electric boiler efficiency (%)",
                                            min_value=70.0, max_value=100.0, value=95.0) / 100
    eboiler_var_cost_input = st.number_input("Electric boiler variable cost (€/MWh_th)",
                                             min_value=0.0, max_value=200.0, value=8.0)
    heat_sell_price_input = st.number_input(
        "Heat sales price from E-boiler (€/MWh_th)",
        min_value=0.0,
        max_value=300.0,
        value=45.0,
        help="Default set below typical gas price so customer still saves while heat becomes project revenue.",
    )

    st.header("📌 Optimisation Strategy")
    optimization_metric = st.selectbox("Optimisation metric", ["Payback", "IRR", "NPV"], index=0)

    with st.expander("🛠️ Advanced CAPEX / OPEX assumptions", expanded=False):
        c_solar_mw       = st.number_input("Solar equipment cost (€/MW)",        value=1_000_000)
        c_bess_mwh       = st.number_input("BESS equipment cost (€/MWh)",         value=300_000)
        c_eboiler_kw     = st.number_input("E-boiler equipment cost (€/kW_e)",    value=120)
        c_eboiler_civil  = st.number_input("E-boiler civils (€/site)",            value=80_000)
        opex_eboiler_fixed = st.number_input("E-boiler fixed OPEX (€/year)",      value=15_000)
        solar_efficiency = st.number_input("Solar panel efficiency (%)",           value=20.0) / 100
        cv_solar         = st.number_input("Civils — solar (€/site)",             value=300_000)
        cv_bess          = st.number_input("Civils — BESS (€/site)",              value=300_000)
        elec_conn        = st.number_input("Electrical works (€/source)",         value=400_000)
        pm_rate          = st.number_input("Project management rate (% of CAPEX)", value=10.0) / 100
        ins_rate         = st.number_input("Insurance rate (% of CAPEX/year)",    value=1.5)  / 100
        maint_rate       = st.number_input("Maintenance rate (% of CAPEX/year)",  value=3.5)  / 100
        lease_cost       = st.number_input("Land lease (€/year)",                 value=100_000)

    st.divider()
    st.header("📐 Site & Physical Constraints")
    site_area_acre = st.number_input("Site area (acres)", min_value=0.0, max_value=100000.0, value=100.0, step=5.0)
    site_max_turbines = int((site_area_acre / 5) * 2)
    site_max_solar = int(site_area_acre / 5)
    col1, col2 = st.columns(2)
    with col1:
        min_turbines = st.number_input("Min turbines",   0, max(0, site_max_turbines), 0)
        min_bess     = st.number_input("Min BESS (MWh)", 0, 40, 4)
        max_bess     = st.slider("Max BESS (MWh)",       0, 40, 12, step=1)
    with col2:
        st.metric("Max turbines (auto)", f"{site_max_turbines}")
        st.metric("Max solar PV (auto, MW)", f"{site_max_solar}")
        export_limit_input  = st.number_input("Grid export limit (kW)", value=5000,
                                              min_value=0, max_value=20000)
        grid_buy_limit_input = st.number_input("Grid import limit (kW)", value=5000,
                                               min_value=0, max_value=20000, 
                                               help="Maximum power that can be imported from grid (network connection capacity)")
        st.caption("Land rule: every 5 acres allows up to 2 turbines and 1 MW PV")
    st.caption(
        f"Standalone maxima from site area: "
        f"Solar ≤ {int(site_area_acre // 5)} MW, "
        f"Wind ≤ {int((site_area_acre / 5) * 2)} turbines"
    )

    run_btn = st.button("🚀 Generate Bankable Report", type="primary", use_container_width=True)

# Show a "Generate Chart" button after sidebar configuration.
# Charts are only rendered when the user explicitly requests them,
# rather than being drawn automatically on every page reload.
default_weather_defaults = st.session_state.get('weather_defaults', {})
preview_btn = st.button("📊 Generate Preview Charts", use_container_width=True)
if preview_btn:
    rep_days_preview = load_custom_typical_days(
        df_customer,
        custom_weights,
        show_warnings=True,
        weather_profiles=default_weather_defaults,
    )
    render_pre_run_preview(rep_days_preview, solar_efficiency, weather_profiles=default_weather_defaults)

# ==========================================
# 4. Run optimisation and display results
# ==========================================
if run_btn or ('report_cache' in st.session_state):
    if run_btn:
        # ---- Input validation ----
        if p_galetech_input <= 0 or p_galetech_input >= p_retail_input:
            st.error(f"🛑 BOO PPA price must be between €0 and €{p_retail_input}/MWh.")
            st.stop()
        if min_bess > max_bess:
            st.error("🛑 Min BESS cannot exceed max BESS.")
            st.stop()
        if min_turbines > int((site_area_acre / 5) * 2):
            st.error("🛑 Minimum wind requirement exceeds available site area.")
            st.stop()
        if (min_turbines / 2.0) > (site_area_acre / 5.0):
            st.error("🛑 Combined land constraint violated by minimum wind setting.")
            st.stop()

        # ---- Load data ----
        rep_days = load_custom_typical_days(
            df_customer,
            custom_weights,
            weather_profiles=default_weather_defaults,
        )
        # Max wind/solar are fully determined by site acreage.
        effective_max_turbines = site_max_turbines
        effective_max_solar    = site_max_solar

        # E-boiler capacity is optimized, not user-fixed.
        # Use heat profile to set search bounds automatically.
        max_gas_load_kw = max(float(np.max(day['gas_load'])) for day in rep_days) if rep_days else 0.0
        if use_e_boiler and eboiler_eff_input > 0:
            auto_max_eboiler_kw = int(np.ceil(max_gas_load_kw / eboiler_eff_input / 500.0) * 500)
            auto_max_eboiler_kw = max(auto_max_eboiler_kw, 500)
        else:
            auto_max_eboiler_kw = 0
        auto_min_eboiler_kw = 0

        optimizer_params = {
            'p_galetech':        p_galetech_input / 1000,   # €/kWh
            'p_gas':             p_gas_input      / 1000,
            'p_heat_sell':       heat_sell_price_input / 1000,
            'p_grid_buy':        p_retail_input   / 1000,
            'p_cust_grid_penalty': cust_grid_penalty_input / 1000,
            'p_sell':            p_sell_input     / 1000,
            'p_carbon':          p_carbon_input,
            'carbon_credit_share': 1.0,
            'grid_carbon_factor':  0.35,
            'enable_e_boiler':   use_e_boiler,
            'eboiler_eff':       eboiler_eff_input,
            'eboiler_var_cost':  eboiler_var_cost_input / 1000,
            'eboiler_max_kw':    auto_max_eboiler_kw,
            'eboiler_capex_per_kw': c_eboiler_kw,
            'eboiler_civil_cost':   c_eboiler_civil,
            'eboiler_fixed_opex':   opex_eboiler_fixed,
            'export_limit_kw':   export_limit_input,
            'grid_buy_limit_kw': grid_buy_limit_input,
            'cost_solar_mw':     c_solar_mw,
            'cost_bess_mwh':     c_bess_mwh,
            'solar_efficiency':  solar_efficiency,
            'civil_solar':       cv_solar,
            'civil_bess':        cv_bess,
            'elec_per_source':   elec_conn,
            'pm_rate':           pm_rate,
            'insurance_rate':    ins_rate,
            'maintenance_rate':  maint_rate,
            'land_lease':        lease_cost,
            'site_area_acre':    site_area_acre,
        }
        optimizer = GaletechAssetOptimizer(optimizer_params)

        turbine_choices = ["None", "EWT 500kW", "EWT 1MW", "E82 2.3MW",
                           "V90 3MW", "E115 4.2MW", "E138 4.26MW"]

        with st.spinner("Running two-stage optimisation (capacity sweep → fine dispatch)…"):
            df_res, best = optimizer.two_stage_optimization(
                rep_days, turbine_choices,
                min_turbines, effective_max_turbines,
                effective_max_solar, min_bess, max_bess,
                site_area_acre,
                optimization_metric=optimization_metric,
                min_eboiler_kw=auto_min_eboiler_kw,
                max_eboiler_kw=auto_max_eboiler_kw,
            )

        if df_res.empty or best is None:
            # Check if we have diagnostic info
            error_msg = (
                "❌ **No commercially viable configurations found.**\n\n"
                "**This happens when:**\n"
                "1. **All configurations are technically infeasible** — Even with grid backup, cannot meet demand peaks\n"
                "   → Add battery storage (BESS) or more renewable capacity (wind/solar)\n"
                "2. **All configurations have negative economics** — Costs exceed revenues at current prices\n"
                "   → Increase PPA electricity price, reduce CAPEX assumptions, or reduce demand\n"
                "3. **constraints are too tight** — Site area, grid link capacity, or other limits prevent viable designs\n\n"
            )
            
            # Add diagnostic details if available
            if not df_res.empty and 'error_type' in df_res.columns:
                skipped = df_res.iloc[0].get('skipped_feasibility', 0)
                failed = df_res.iloc[0].get('failed_infeasible', 0)
                if skipped > 0 or failed > 0:
                    error_msg += f"**Diagnostic breakdown (Stage 1):**\n"
                    if skipped > 0:
                        error_msg += f"- {skipped} configs rejected by feasibility pre-check (peak demand > peak supply)\n"
                    if failed > 0:
                        error_msg += f"- {failed} configs failed dispatch optimization (LP infeasible)\n\n"
            
            error_msg += (
                "**Recommended actions:**\n"
                "- 📈 Increase customer electricity price (PPA rate)\n"
                "- 📊 Check your load profile — very high peaks may require large storage\n"
                "- 🔌 Verify grid_buy_limit_kw is sufficient for backup power\n"
                "- 💰 Reduce cost assumptions (CAPEX, OPEX, interest rate)\n"
                "- 🌍 Check that renewable resource (wind/solar) is adequate for location"
            )
            
            st.error(error_msg)
            st.stop()

        # ---- Filter viability by selected optimisation metric ----
        if optimization_metric == "IRR":
            df_viable = df_res[df_res['IRR'] > 0].copy()
            df_viable = df_viable.sort_values('IRR', ascending=False)
            viability_label = "positive IRR"
        elif optimization_metric == "NPV":
            df_viable = df_res[df_res['NPV10_M'] > 0].copy()
            df_viable = df_viable.sort_values('NPV10_M', ascending=False)
            viability_label = "positive NPV"
        else:
            df_viable = df_res[(df_res['Payback'] > 0) & (df_res['Payback'] < optimizer.project_life_years)].copy()
            df_viable = df_viable.sort_values('Payback', ascending=True)
            viability_label = "positive payback within project life"
        df_viable.reset_index(drop=True, inplace=True)

        if df_viable.empty:
            st.warning(
                f"No configurations achieved {viability_label}. "
                "Showing the closest available configuration for diagnostics."
            )
            # Fall back to best available scenario by selected metric.
            if optimization_metric == "IRR":
                df_fallback = df_res.sort_values('IRR', ascending=False).reset_index(drop=True)
            elif optimization_metric == "NPV":
                df_fallback = df_res.sort_values('NPV10_M', ascending=False).reset_index(drop=True)
            else:
                df_fallback = df_res.sort_values('Payback', ascending=True).reset_index(drop=True)
            best = df_fallback.iloc[0].copy()
            has_viable_selection = False
            st.info(
                f"Best available now: Payback={best['Payback']:.1f} yrs, "
                f"IRR={best['IRR']:.1f}%, NPV={best['NPV10_M']:.2f} M€. "
                "You can improve viability by increasing electricity price, reducing CAPEX assumptions, "
                "or relaxing site/BESS constraints."
            )
        else:
            best = df_viable.iloc[0].copy()
            has_viable_selection = True

        # ---- Compute minimum PPA for target IRR ----
        capex, _, _, _, _ = optimizer.get_capex(best['t_model_raw'], best['t_count_raw'],
                                                 best['Solar_MW'], best['BESS_MWh'],
                                                 eboiler_kw=best.get('EBoiler_kW', 0))
        annual_opex, _, _, _ = optimizer.get_opex(capex, t_count=best['t_count_raw'], s_mw=best['Solar_MW'])
        revenue   = best['Profit_k'] * 1000 + annual_opex
        btm_e     = best['Elec_Offset_MWh'] * 1000   # kWh/year
        target_irr = target_irr_input / 100
        pv_factor  = (1 - (1 + target_irr) ** -optimizer.project_life_years) / target_irr
        req_profit = capex / pv_factor if pv_factor > 0 else 0
        req_revenue = req_profit + annual_opex
        rev_non_elec = revenue - (btm_e * (p_galetech_input / 1000)) if btm_e > 0 else 0
        min_ppa_elec = (req_revenue - rev_non_elec) / btm_e * 1000 if btm_e > 0 else 0
        best['Min_Elec_PPA'] = min_ppa_elec

        # ---- Annual cash-flow table ----
        best_capex     = best['CAPEX_M'] * 1e6
        best_opex      = best['OPEX_k']  * 1e3
        best_netprofit = best['Profit_k'] * 1e3
        best_revenue_  = best_netprofit + best_opex
        cashflow_rows  = [{'Year': 0, 'CAPEX': -best_capex, 'Revenue': 0.0,
                           'OPEX': 0.0, 'NetProfit': -best_capex, 'CashFlow': -best_capex}]
        for y in range(1, optimizer.project_life_years + 1):
            cashflow_rows.append({'Year': y, 'CAPEX': 0.0, 'Revenue': best_revenue_,
                                  'OPEX': best_opex, 'NetProfit': best_netprofit,
                                  'CashFlow': best_netprofit})
        df_cashflow = pd.DataFrame(cashflow_rows)
        # Cumulative profit by year (running total of annual net profit).
        df_cashflow['CumulativeProfit'] = df_cashflow['NetProfit'].cumsum()

        # ---- Dispatch traces for best config ----
        df_traces = optimizer.evaluate_combination(
            best['t_model_raw'], best['t_count_raw'],
            best['Solar_MW'], best['BESS_MWh'],
            rep_days, return_traces=True,
            eboiler_kw=best.get('EBoiler_kW', 0),
        )
        csv_buffer = io.StringIO()
        df_traces.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')

        st.session_state['report_cache'] = {
            'df_res': df_res,
            'df_viable': df_viable,
            'best': best,
            'has_viable_selection': has_viable_selection,
            'df_cashflow': df_cashflow,
            'df_traces': df_traces,
            'csv_data': csv_data,
            'rep_days': rep_days,
            'optimizer_params': optimizer_params,
            'optimization_metric': optimization_metric,
            'target_irr_input': target_irr_input,
        }
    else:
        cache = st.session_state['report_cache']
        df_res = cache['df_res']
        df_viable = cache['df_viable']
        best = cache['best']
        has_viable_selection = cache.get('has_viable_selection', True)
        df_cashflow = cache['df_cashflow']
        df_traces = cache['df_traces']
        csv_data = cache['csv_data']
        rep_days = cache['rep_days']
        optimizer_params = cache['optimizer_params']
        optimization_metric = cache['optimization_metric']
        target_irr_input = cache['target_irr_input']
        optimizer = GaletechAssetOptimizer(optimizer_params)

    st.session_state['best_config'] = best
    st.session_state['rep_days'] = rep_days

    if has_viable_selection:
        st.success("✅ Optimisation complete. Bankable report generated.")
    else:
        st.warning("⚠️ Optimisation complete, but no configuration passed the selected metric's viability threshold.")

    # ==========================================
    # 5. Output tabs
    # ==========================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Executive Summary",
        "2. Cost Breakdown",
        "3. Benchmarking",
        "4. Monte Carlo Risk",
        "5. Auditable Pack",
    ])

    # ---------- Tab 1: Executive Summary ----------
    with tab1:
        st.header("Executive Summary")
        title_map = {'Payback': 'Shortest Payback', 'IRR': 'Highest IRR', 'NPV': 'Highest NPV'}
        st.markdown(f"### 🎯 Headline Recommendation ({title_map.get(optimization_metric, 'Payback')})")

        if optimization_metric == 'Payback':
            st.markdown("Optimal configuration to **minimise payback period** (fastest capital recovery):")
        elif optimization_metric == 'IRR':
            st.markdown("Optimal configuration to **maximise IRR** (strongest return):")
        else:
            st.markdown("Optimal configuration to **maximise NPV** (best value creation):")

        st.markdown(
            f"- **Wind:** {best['Turbine']}  |  "
            f"**Solar PV:** {best['Solar_MW']} MW  |  "
            f"**BESS:** {best['BESS_MWh']} MWh"
        )

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Payback period",   f"{best['Payback']:.1f} yrs")
        c2.metric("Project IRR",      f"{best['IRR']:.1f} %")
        c3.metric("NPV @ 10 %",       f"€ {best['NPV10_M']:.2f} M")
        c4.metric("E-boiler size",    f"{int(best.get('EBoiler_kW', 0))} kW")
        c5.metric("Customer green electricity share", f"{best.get('Green_Elec_Share_pct', 0.0):.1f} %")

        st.caption(f"Gas retained for heat: {best['Gas_Used_MWh']:.0f} MWh/yr")
        heat_mode = (
            "Electric boiler preferred"
            if best['Heat_By_EBoiler_MWh'] > best['Gas_Used_MWh']
            else "Gas boiler preferred"
        )
        st.info(f"Optimal heating dispatch: **{heat_mode}**")
        st.info(
            f"**Minimum electricity PPA price to achieve {target_irr_input:.0f} % IRR:** "
            f"€ {best['Min_Elec_PPA']:.2f} / MWh"
        )

        # ---- Customer decarbonisation & cost savings ----
        # Gas and carbon costs are borne by the customer.  The optimiser minimises
        # residual gas consumption as an environmental objective on their behalf.
        st.divider()
        st.markdown("### Customer Decarbonisation & Cost Savings")
        st.caption(
            "Gas purchase and carbon costs are the customer's responsibility. "
            "Carbon-credit value from avoided emissions is fully retained by the energy system. "
            "When E-boiler heat is sold to customer, customer gas saving is shown as net saving "
            "(avoided gas cost minus purchased heat cost). "
            "Electricity decarbonisation is calculated from net RE displacement of grid power "
            "(project supply minus project-side grid charging used for BESS). "
            "Customer-grid-backup penalty is applied as a soft service-priority constraint."
        )
        carbon_share_ui = optimizer.params.get('carbon_credit_share', 1.0)
        cust_gas_sav   = best.get('Cust_Gas_Savings_k', 0.0)    # k€/yr
        galetech_carb_rev = best.get('Galetech_Carbon_Credit_k', 0.0)  # k€/yr
        cust_carb_sav  = best.get('Cust_Carbon_Savings_k', 0.0)  # k€/yr
        heat_sales_rev = best.get('Heat_Sales_k', 0.0)            # k€/yr
        co2_displaced  = best['CO2_T']                            # tonnes CO2/yr
        heat_displaced = best['Heat_By_EBoiler_MWh']              # MWh_th/yr

        cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
        cc1.metric("CO₂ displaced",               f"{co2_displaced:,.0f} t/yr")
        cc2.metric("Gas heat displaced",          f"{heat_displaced:,.0f} MWh/yr")
        cc3.metric("Heat sales revenue",          f"€ {heat_sales_rev:,.1f} k/yr")
        cc4.metric("Galetech carbon credit",      f"€ {galetech_carb_rev:,.1f} k/yr")
        cc5.metric("Customer net heat saving",    f"€ {cust_gas_sav:,.1f} k/yr")
        cc6.metric("Customer carbon benefit",     f"€ {cust_carb_sav:,.1f} k/yr")
        st.caption(f"Carbon share to system: {carbon_share_ui * 100:.0f} %")

        total_cust_saving = cust_gas_sav + cust_carb_sav
        st.success(
            f"Total estimated customer annual saving from gas & carbon: "
            f"**€ {total_cust_saving:,.1f} k/yr**  "
            f"(gas displaced {heat_displaced:,.0f} MWh/yr · "
            f"CO₂ avoided {co2_displaced:,.0f} t/yr)"
        )

    # ---------- Tab 2: Cost Breakdown ----------
    with tab2:
        st.header("Financial Performance & Breakdown")
        df_table = df_viable if not df_viable.empty else df_res
        show_cols = ['Turbine', 'Solar_MW', 'BESS_MWh', 'Payback', 'IRR', 'NPV10_M',
                     'CAPEX_M', 'EBoiler_kW', 'Elec_Offset_MWh', 'Green_Elec_Share_pct', 'Heat_By_EBoiler_MWh',
                     'Gas_Used_MWh', 'Curtailed_MWh',
                     'Heat_Sales_k',
                     'Galetech_Carbon_Credit_k',
                     'Cust_Gas_Savings_k', 'Cust_Carbon_Savings_k']
        # Only include columns that actually exist (guards against old cached DataFrames)
        show_cols = [c for c in show_cols if c in df_table.columns]
        st.dataframe(
            df_table[show_cols].head(10).style.format({
                'CAPEX_M':                '{:.2f}',
                'EBoiler_kW':             '{:.0f}',
                'NPV10_M':                '{:.2f}',
                'IRR':                    '{:.1f}%',
                'Payback':                '{:.1f} yrs',
                'Elec_Offset_MWh':        '{:.0f}',
                'Green_Elec_Share_pct':   '{:.1f}%',
                'Heat_By_EBoiler_MWh':    '{:.0f}',
                'Gas_Used_MWh':           '{:.0f}',
                'Curtailed_MWh':          '{:.0f}',
                'Heat_Sales_k':           '{:.1f} k€',
                'Galetech_Carbon_Credit_k': '{:.1f} k€',
                'Cust_Gas_Savings_k':     '{:.1f} k€',
                'Cust_Carbon_Savings_k':  '{:.1f} k€',
            })
        )

        st.subheader("Best config — annual cash flows (constant-revenue estimate)")
        st.dataframe(
            df_cashflow.style.format({
                'CAPEX':     '€{:,.0f}',
                'Revenue':   '€{:,.0f}',
                'OPEX':      '€{:,.0f}',
                'NetProfit': '€{:,.0f}',
                'CashFlow':  '€{:,.0f}',
                'CumulativeProfit': '€{:,.0f}',
            })
        )

    # ---------- Tab 3: Benchmarking ----------
    with tab3:
        st.header("Technology Benchmarking")
        benchmarks = [dict(best) | {'Category': '⭐ Recommended optimum'}]

        no_bess = df_res[(df_res['BESS_MWh'] == 0) &
                         ((df_res['Solar_MW'] > 0) | (df_res['Turbine'] != 'No Wind'))
                        ].sort_values('Payback')
        if not no_bess.empty:
            benchmarks.append(dict(no_bess.iloc[0]) | {'Category': '❌ Renewables only (no battery)'})

        bess_only = df_res[(df_res['BESS_MWh'] > 0) &
                           (df_res['Solar_MW'] == 0) &
                           (df_res['Turbine'] == 'No Wind')
                          ].sort_values('Payback')
        if not bess_only.empty:
            benchmarks.append(dict(bess_only.iloc[0]) | {'Category': '❌ Battery only (no renewables)'})

        df_bench = pd.DataFrame(benchmarks)
        bench_cols = ['Category', 'Turbine', 'Solar_MW', 'BESS_MWh',
                      'Payback', 'IRR', 'NPV10_M', 'CAPEX_M']
        st.dataframe(
            df_bench[bench_cols].style.format({
                'CAPEX_M': '{:.2f}', 'NPV10_M': '{:.2f}',
                'IRR': '{:.1f}%',   'Payback': '{:.1f}',
            })
        )

    # ---------- Tab 4: Monte Carlo ----------
    with tab4:
        st.header("Monte Carlo Uncertainty Analysis")
        st.markdown(
            """
**Perturbed parameters in each simulation run:**

| Parameter | Distribution | Std Dev | What it captures |
|---|---|---|---|
| **Wind resource** (`wind_shock`) | Normal(1.0, 0.10) | ±10% | Inter-annual wind speed variability, turbine degradation, wake losses |
| **CAPEX** (`capex_shock`) | Normal(1.0, 0.05) | ±5% | Construction cost overrun, equipment price fluctuation |
| **PPA electricity price** (`ppa_shock`) | Normal(1.0, 0.05) | ±5% | Revenue uncertainty from contract renegotiation or market price changes |

*Each run applies independently sampled multipliers to the optimal configuration and re-solves dispatch to compute Payback, IRR, and NPV.*
            """
        )
        if st.button("🎲 Run 50 Monte Carlo simulations on optimal config"):
            mc_results = []
            progress   = st.progress(0)
            best_conf  = st.session_state['best_config']

            for i in range(50):
                # --- Perturbed uncertain parameters ---
                # Wind resource multiplier: ±10% std, captures inter-annual variability
                wind_shock  = np.random.normal(1.0, 0.10)
                # CAPEX multiplier: ±5% std, captures construction cost uncertainty
                capex_shock = np.random.normal(1.0, 0.05)
                # PPA price multiplier: ±5% std, captures revenue price uncertainty
                ppa_shock   = np.random.normal(1.0, 0.05)

                mc_capex, _, _, _, _ = optimizer.get_capex(
                    best_conf['t_model_raw'], best_conf['t_count_raw'],
                    best_conf['Solar_MW'], best_conf['BESS_MWh'], capex_shock,
                    eboiler_kw=best_conf.get('EBoiler_kW', 0),
                )
                mc_opex, _, _, _ = optimizer.get_opex(
                    mc_capex,
                    t_count=best_conf['t_count_raw'],
                    s_mw=best_conf['Solar_MW'],
                )

                out = optimizer.evaluate_combination(
                    best_conf['t_model_raw'], best_conf['t_count_raw'],
                    best_conf['Solar_MW'], best_conf['BESS_MWh'],
                    st.session_state['rep_days'],
                    wind_shock=wind_shock, ppa_shock=ppa_shock,
                    eboiler_kw=best_conf.get('EBoiler_kW', 0),
                )
                if out is None or isinstance(out, pd.DataFrame):
                    continue
                if not isinstance(out, tuple) or len(out) < 1:
                    continue
                rev = out[0]
                mc_profit    = rev - mc_opex
                mc_cashflows = [-mc_capex] + [mc_profit] * optimizer.project_life_years
                mc_irr     = optimizer._safe_irr(mc_capex, mc_profit)
                mc_npv     = npf.npv(0.10, mc_cashflows)
                mc_payback = mc_capex / mc_profit if mc_profit > 0 else 99.0

                mc_results.append({
                    'Iteration':  i,
                    'Payback':    mc_payback,
                    'IRR':        mc_irr * 100,
                    'NPV_M':      mc_npv / 1e6,
                    'Wind_Shock': wind_shock,
                })
                progress.progress((i + 1) / 50)

            progress.empty()
            if not mc_results:
                st.warning("All Monte Carlo runs failed — check solver availability.")
            else:
                df_mc = pd.DataFrame(mc_results)
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Payback distribution (P10 / P50 / P90)**")
                    p10, p50, p90 = np.percentile(df_mc['Payback'], [10, 50, 90])
                    st.write(f"- Optimistic (P10): **{p10:.1f} yrs**")
                    st.write(f"- Expected   (P50): **{p50:.1f} yrs**")
                    st.write(f"- Conservative (P90): **{p90:.1f} yrs**")
                    st.markdown(
                        """
**How to interpret these percentiles**

- **P10 (Optimistic)**: 10% of scenarios achieve this payback or better.
- **P50 (Median)**: Central outcome; half of scenarios are better and half are worse.
- **P90 (Conservative)**: 90% of scenarios are this value or better; useful for downside planning.

Use the spread between P10 and P90 as a quick risk indicator. A wider gap means higher uncertainty.
                        """
                    )
                with c2:
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.hist(df_mc['Payback'], bins=10, color='steelblue', edgecolor='white')
                    ax.axvline(p50, color='red', linestyle='--', linewidth=1.5, label='P50')
                    ax.set_xlabel('Payback (years)')
                    ax.set_title('Monte Carlo payback distribution')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        "Histogram guide: each bar shows how many simulations fall into a payback range. "
                        "The dashed red line marks the median (P50)."
                    )

    # ---------- Tab 5: Auditable Pack ----------
    with tab5:
        st.header("Auditable Calculation Pack")
        st.download_button(
            "📥 Download hourly dispatch pack (CSV)",
            data=csv_data,
            file_name=(
                f"Galetech_Audit_{best['Turbine'].replace(' ', '_')}"
                f"_{best['Solar_MW']}MW_{best['BESS_MWh']}MWh.csv"
            ),
            mime="text/csv",
        )
        if not df_traces.empty:
            total_rows = len(df_traces)
            default_hours = min(48, total_rows)
            preview_hours = int(
                st.slider(
                    "Preview horizon (hours)",
                    min_value=1,
                    max_value=total_rows,
                    value=default_hours,
                    step=1,
                )
            )
            view_mode = st.radio(
                "Chart mode",
                ["Split panels", "Single combined chart"],
                horizontal=True,
            )

            preview = df_traces.head(preview_hours).copy()
            x = np.arange(len(preview))
            # Day-boundary positions (one vertical line every 24 h)
            day_boundaries = list(range(24, len(preview), 24))
            has_wind_asset = int(best.get('t_count_raw', 0)) > 0
            has_solar_asset = float(best.get('Solar_MW', 0.0)) > 0
            has_bess_asset = float(best.get('BESS_MWh', 0.0)) > 0

            st.caption(
                f"Previewing first {preview_hours} rows out of {total_rows}. "
                "Rows come from concatenated representative-day traces (typically 24h per day-type), "
                "not necessarily one continuous real calendar timeline."
            )
            hidden_series = []
            if not has_wind_asset:
                hidden_series.append("wind generation")
            if not has_solar_asset:
                hidden_series.append("solar generation")
            if not has_bess_asset:
                hidden_series.append("BESS charge/discharge and BESS grid-import flow")
            if hidden_series:
                st.info("Hidden in charts (not installed in optimal config): " + ", ".join(hidden_series) + ".")

            if view_mode == "Split panels":
                fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

                # Panel 1: customer demand coverage
                if 'Elec_Demand_kW' in preview:
                    axes[0].plot(x, preview['Elec_Demand_kW'], label='Customer demand', linewidth=2)
                if 'BTM_Supply_kW' in preview:
                    axes[0].plot(x, preview['BTM_Supply_kW'], label='Project supply to customer', linewidth=2)
                if 'Customer_Grid_Backup_kW' in preview:
                    axes[0].plot(x, preview['Customer_Grid_Backup_kW'], label='Customer grid backup', linewidth=2)
                axes[0].set_ylabel('kW')
                axes[0].set_title('Customer Supply Split')
                axes[0].grid(alpha=0.25)
                axes[0].legend(loc='upper right', fontsize=8)

                # Panel 2: renewable and battery operation
                if 'Wind_Gen_kW' in preview and has_wind_asset:
                    axes[1].plot(x, preview['Wind_Gen_kW'], label='Wind generation', linewidth=2)
                if 'Solar_Gen_kW' in preview and has_solar_asset:
                    axes[1].plot(x, preview['Solar_Gen_kW'], label='Solar generation', linewidth=2)
                if 'BESS_Discharge_kW' in preview and has_bess_asset:
                    axes[1].plot(x, preview['BESS_Discharge_kW'], label='BESS discharge', linewidth=2)
                if 'BESS_Charge_kW' in preview and has_bess_asset:
                    axes[1].plot(x, -preview['BESS_Charge_kW'], label='BESS charge (-)', linewidth=2)
                axes[1].axhline(0, color='black', linewidth=0.8, alpha=0.6)
                axes[1].set_ylabel('kW')
                axes[1].set_title('Generation and BESS Operation')
                axes[1].grid(alpha=0.25)
                axes[1].legend(loc='upper right', fontsize=8)

                # Panel 3: grid and curtailment flows
                if 'Grid_Export_kW' in preview:
                    axes[2].plot(x, preview['Grid_Export_kW'], label='Grid export', linewidth=2)
                if 'Project_Grid_Import_for_BESS_kW' in preview and has_bess_asset:
                    axes[2].plot(x, preview['Project_Grid_Import_for_BESS_kW'], label='Project grid import for BESS', linewidth=2)
                if 'Curtailed_kW' in preview:
                    axes[2].plot(x, preview['Curtailed_kW'], label='Curtailment', linewidth=2)
                axes[2].set_ylabel('kW')
                for ax_i in axes:
                    for db in day_boundaries:
                        ax_i.axvline(db, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
                axes[2].set_xlabel('Hour  (each step = 1 h; dashed lines = day boundaries)')
                axes[2].set_title('Grid Interaction and Curtailment')
                axes[2].grid(alpha=0.25)
                axes[2].legend(loc='upper right', fontsize=8)

                st.pyplot(fig)
                plt.close(fig)
            else:
                fig, ax = plt.subplots(figsize=(12, 5))
                series_to_plot = [
                    ('Elec_Demand_kW', 'Customer demand'),
                    ('BTM_Supply_kW', 'Project supply to customer'),
                    ('Customer_Grid_Backup_kW', 'Customer grid backup'),
                    ('Wind_Gen_kW', 'Wind generation'),
                    ('Solar_Gen_kW', 'Solar generation'),
                    ('BESS_Discharge_kW', 'BESS discharge'),
                    ('Grid_Export_kW', 'Grid export'),
                    ('Project_Grid_Import_for_BESS_kW', 'Project grid import for BESS'),
                    ('Curtailed_kW', 'Curtailment'),
                ]
                for col, label in series_to_plot:
                    if col == 'Wind_Gen_kW' and not has_wind_asset:
                        continue
                    if col == 'Solar_Gen_kW' and not has_solar_asset:
                        continue
                    if col in ['BESS_Discharge_kW', 'Project_Grid_Import_for_BESS_kW'] and not has_bess_asset:
                        continue
                    if col in preview:
                        ax.plot(x, preview[col], linewidth=1.8, label=label)
                if 'BESS_Charge_kW' in preview and has_bess_asset:
                    ax.plot(x, -preview['BESS_Charge_kW'], linewidth=1.8, label='BESS charge (-)')

                ax.axhline(0, color='black', linewidth=0.8, alpha=0.6)
                ax.set_ylabel('kW')
                for db in day_boundaries:
                    ax.axvline(db, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
                ax.set_xlabel('Hour  (each step = 1 h; dashed lines = day boundaries)')
                ax.set_title('Hourly Dispatch Preview (Combined)')
                ax.grid(alpha=0.25)
                ax.legend(loc='upper right', fontsize=8, ncol=2)
                st.pyplot(fig)
                plt.close(fig)

            # --- BESS SOC profile ---
            if 'BESS_SoC_kWh' in preview.columns and best['BESS_MWh'] > 0:
                cap_kwh = best['BESS_MWh'] * 1000
                st.subheader("Battery State of Charge (SoC)")
                fig_soc, ax_soc = plt.subplots(figsize=(12, 3))
                ax_soc.plot(x, preview['BESS_SoC_kWh'], color='steelblue', linewidth=2, label='SoC (kWh)')
                ax_soc.fill_between(x, preview['BESS_SoC_kWh'], alpha=0.15, color='steelblue')
                ax_soc.axhline(cap_kwh * 0.9, color='red',   linewidth=1, linestyle='--', label='90% limit')
                ax_soc.axhline(cap_kwh * 0.5, color='grey',  linewidth=1, linestyle=':',  label='50% (initial)')
                ax_soc.axhline(cap_kwh * 0.1, color='orange',linewidth=1, linestyle='--', label='10% limit')
                ax_soc.set_ylim(0, cap_kwh * 1.05)
                ax_soc.set_ylabel('SoC (kWh)')
                for db in day_boundaries:
                    ax_soc.axvline(db, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
                ax_soc.set_xlabel('Hour  (each step = 1 h; dashed lines = day boundaries)')
                ax_soc.set_title(f'BESS State of Charge — capacity {best["BESS_MWh"]} MWh ({cap_kwh:.0f} kWh)')
                ax_soc.grid(alpha=0.25)
                ax_soc.legend(loc='upper right', fontsize=8)
                st.pyplot(fig_soc)
                plt.close(fig_soc)
            else:
                st.info("SOC chart is only available when the selected optimal configuration includes BESS (BESS_MWh > 0).")