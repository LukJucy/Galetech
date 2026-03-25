import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy_financial as npf
import io

# ==========================================
# 1. Core optimization engine (includes gas/carbon and grid constraint)
# ==========================================
class GaletechAssetOptimizer:
    def __init__(self, params, wind_solar_ref_data=None):
        # optimizer configuration dictionary
        self.params = params
        # battery round-trip efficiency (fraction)
        self.bess_eff = 0.95
        # usable cycle life of battery
        self.bess_cycles = 8000 
        # carbon factor: kg CO2 per 1 kWh of natural gas burn
        self.gas_carbon_factor = 0.202
        self.project_life_years = 20
        self.discount_rate = 0.10
        
        # 内置风光发电参考数据（kW per 1MW/1台）
        # 结构: {'wind_turbine_model': [hourly output], 'solar_per_mw': [hourly output]}
        self.wind_solar_ref_data = wind_solar_ref_data or {}
        
        self.turbine_models = {
            "None": {"mw": 0, "curve": [0]*13, "equip_cost": 0, "civil_cost": 0},
            "EWT 500kW": {"mw": 0.5, "curve": [0, 0, 3, 15, 35, 50, 120, 220, 340, 480, 500, 500, 500], "equip_cost": 800000, "civil_cost": 1000000},
            "EWT 1MW": {"mw": 1.0, "curve": [0, 35, 45, 54, 63, 100, 200, 300, 480, 610, 700, 850, 1000], "equip_cost": 1300000, "civil_cost": 1200000},
            "E82 2.3MW": {"mw": 2.3, "curve": [0, 0, 3, 25, 82, 174, 321, 532, 815, 1180, 1580, 1900, 2300], "equip_cost": 2500000, "civil_cost": 1500000},
            "V90 3MW": {"mw": 3.0, "curve": [0, 0, 0, 4, 77, 190, 353, 581, 886, 1273, 1710, 2200, 3000], "equip_cost": 3100000, "civil_cost": 1800000},
            "E115 4.2MW": {"mw": 4.2, "curve": [0, 0, 9, 57, 163, 351, 628, 1008, 1501, 2092, 2733, 3300, 4200], "equip_cost": 4200000, "civil_cost": 2200000},
            "E138 4.26MW": {"mw": 4.26, "curve": [0, 0, 2, 69, 250, 540, 952, 1506, 2173, 2865, 3474, 3900, 4260], "equip_cost": 4500000, "civil_cost": 2500000}
        }

    def get_wind_power(self, wind_speed, model_name, count, shock_factor=1.0):
        if count == 0 or model_name == "None": return np.zeros_like(wind_speed)
        curve = self.turbine_models[model_name]["curve"]
        w_speed_capped = np.clip(wind_speed, 0, 12)
        return np.interp(w_speed_capped, np.arange(13), curve) * count * shock_factor

    def get_capex(self, t_model, t_count, s_mw, b_mwh, capex_shock=1.0):
        w_mw = self.turbine_models[t_model]["mw"] * t_count
        equip_wind = self.turbine_models[t_model]["equip_cost"] * t_count
        equip_solar = s_mw * self.params['cost_solar_mw']
        equip_bess = b_mwh * self.params['cost_bess_mwh']
        equip_total = equip_wind + equip_solar + equip_bess
        
        civils = 0
        if t_count > 0: civils += self.turbine_models[t_model]["civil_cost"] * t_count
        if s_mw > 0: civils += self.params['civil_solar']
        if b_mwh > 0: civils += self.params['civil_bess']
        
        sources_count = sum([w_mw > 0, s_mw > 0, b_mwh > 0])
        elec = sources_count * self.params['elec_per_source']
        
        subtotal = equip_total + civils + elec
        pm_cost = subtotal * self.params['pm_rate']
        
        return (subtotal + pm_cost) * capex_shock, equip_total, civils, elec, pm_cost

    def get_opex(self, total_capex):
        insurance = total_capex * self.params['insurance_rate']
        maintenance = total_capex * self.params['maintenance_rate']
        lease = self.params['land_lease']
        return insurance + maintenance + lease, insurance, maintenance, lease

    def evaluate_combination(self, t_model, t_count, s_mw, b_mwh, rep_days, return_traces=False, wind_shock=1.0, ppa_shock=1.0):
        # t_model: wind turbine type key
        # t_count: number of turbines installed
        # s_mw: solar capacity (MW)
        # b_mwh: battery capacity (MWh)
        # rep_days: list of representative daily profiles for wind/solar/load
        # return_traces: if true, returns dispatch trace DataFrame
        # wind_shock: wind power multiplier for scenario stress-testing
        # ppa_shock: PPA price multiplier for scenario stress-testing

        annual_revenue = 0  # accumulated scenario-weighted revenue
        annual_co2_saved = 0  # total CO2 tonnes avoided
        annual_btm_supply_kwh = 0  # behind-the-meter electricity supplied in kWh
        annual_btm_gas_kwh = 0  # gas avoided/supplied equivalence
        annual_curtailed_kwh = 0  # curtailed energy in kWh
        annual_curtail_cost = 0  # cost of curtailed energy in Euros

        bess_wear_cost = (b_mwh * self.params['cost_bess_mwh']) / (self.bess_cycles * b_mwh * 1000) if b_mwh > 0 else 0
        carbon_premium = self.gas_carbon_factor * self.params['p_carbon'] / 1000

        export_limit_kw = self.params.get('export_limit_kw', 5000)
        bess_max_power_kw = (b_mwh * 1000) * 0.5 if b_mwh > 0 else 0

        all_traces = []

        for day_idx, day in enumerate(rep_days):
            T = len(day['elec_load'])
            p_ch = cp.Variable(T, nonneg=True)
            p_dis = cp.Variable(T, nonneg=True)
            soc = cp.Variable(T, nonneg=True)
            p_btm_supply = cp.Variable(T, nonneg=True)
            p_gas_replaced = cp.Variable(T, nonneg=True)
            p_grid_sell = cp.Variable(T, nonneg=True)
            p_grid_buy = cp.Variable(T, nonneg=True)
            p_curtail = cp.Variable(T, nonneg=True)
            z_grid = cp.Variable(T, nonneg=True)
            z_bess = cp.Variable(T, nonneg=True) if b_mwh > 0 else None

            p_wind = np.zeros(T)
            if t_model in self.wind_solar_ref_data and t_model != "None":
                p_wind = self.wind_solar_ref_data[t_model] * t_count * wind_shock

            p_solar = np.zeros(T)
            if 'solar_per_mw' in self.wind_solar_ref_data and s_mw > 0:
                p_solar = self.wind_solar_ref_data['solar_per_mw'] * s_mw * self.params['solar_efficiency']

            grid_buy_limit_kw = export_limit_kw * 0.8

            constraints = [
                p_wind + p_solar + p_dis + p_grid_buy == p_btm_supply + p_grid_sell + p_ch + p_curtail,
                p_btm_supply >= day['elec_load'] + p_gas_replaced,
                p_gas_replaced >= day['gas_load'],
                p_gas_replaced <= day['gas_load'],
                p_grid_sell <= export_limit_kw,
                p_grid_buy <= grid_buy_limit_kw,
                p_grid_sell <= export_limit_kw * z_grid,
                p_grid_buy <= grid_buy_limit_kw * (1 - z_grid),
                z_grid >= 0,
                z_grid <= 1,
                p_curtail <= p_wind + p_solar
            ]

            if b_mwh > 0:
                cap_kwh = b_mwh * 1000
                constraints += [
                    soc <= cap_kwh * 0.9,
                    soc >= cap_kwh * 0.1,
                    soc[0] == cap_kwh * 0.5,
                    soc[T-1] == cap_kwh * 0.5,
                    p_ch <= bess_max_power_kw,
                    p_dis <= bess_max_power_kw,
                    p_ch <= bess_max_power_kw * (1 - z_bess),
                    p_dis <= bess_max_power_kw * z_bess,
                    z_bess >= 0,
                    z_bess <= 1
                ]
                for t in range(1, T):
                    constraints.append(soc[t] == soc[t-1] + p_ch[t] * self.bess_eff - p_dis[t] / self.bess_eff)
            else:
                constraints += [p_ch == 0, p_dis == 0, soc == 0]

            rev_customer = cp.sum(p_btm_supply * (self.params['p_galetech'] * ppa_shock))
            rev_grid = cp.sum(p_grid_sell * self.params['p_sell'])
            val_carbon = cp.sum(p_gas_replaced * carbon_premium)
            grid_purchase_cost = cp.sum(p_grid_buy * (self.params['p_sell'] * 0.8))

            obj = cp.Maximize(rev_customer + rev_grid + val_carbon - grid_purchase_cost - cp.sum(p_dis * bess_wear_cost))
            prob = cp.Problem(obj, constraints)

            try:
                if 'GUROBI' in cp.installed_solvers():
                    prob.solve(solver=cp.GUROBI)
                else:
                    prob.solve(solver=cp.ECOS)
                if prob.status not in ["infeasible", "unbounded"] and prob.value is not None:
                    annual_revenue += prob.value * day['weight']
                    annual_co2_saved += np.sum(p_gas_replaced.value) * self.gas_carbon_factor / 1000 * day['weight']
                    annual_btm_supply_kwh += np.sum(p_btm_supply.value) * day['weight']
                    annual_btm_gas_kwh += np.sum(p_gas_replaced.value) * day['weight']
                    annual_curtailed_kwh += np.sum(p_curtail.value) * day['weight']
                    annual_curtail_cost += np.sum(p_curtail.value) * self.params['p_sell'] * day['weight']

                    if return_traces:
                        for t in range(T):
                            all_traces.append({
                                "Day_Type": f"Scenario_{day_idx+1}", "Hour": t,
                                "Elec_Demand_kW": day['elec_load'][t], "Gas_Demand_kW": day['gas_load'][t],
                                "Wind_Gen_kW": p_wind[t], "Solar_Gen_kW": p_solar[t],
                                "BESS_SoC_kWh": soc.value[t] if b_mwh > 0 else 0,
                                "BTM_Supply_kW": p_btm_supply.value[t], "Grid_Export_kW": p_grid_sell.value[t],
                                "Curtailed_kW": p_curtail.value[t]
                            })
            except Exception:
                pass

        if return_traces:
            return pd.DataFrame(all_traces)
        return annual_revenue, annual_co2_saved, annual_btm_supply_kwh, annual_btm_gas_kwh, annual_curtailed_kwh, annual_curtail_cost

    def two_stage_optimization(self, rep_days, turbine_choices, min_turbines, max_turbines, min_solar, max_solar, min_bess, max_bess, export_limit_kw, optimization_metric='Payback'):
        # Stage 1: coarse capacity sweep
        stage1_results = []
        t_start = max(min_turbines, 0)
        s_start = max(min_solar, 0)
        b_start = max(min_bess, 0)
        t_count_steps = range(t_start, max_turbines + 1, max(1, max(1, max_turbines - t_start) // 3)) if max_turbines >= t_start else [t_start]
        s_steps = range(s_start, max_solar + 1, max(1, max(1, max_solar - s_start) // 3)) if max_solar >= s_start else [s_start]
        b_steps = range(b_start, max_bess + 1, max(1, max(1, max_bess - b_start) // 3)) if max_bess >= b_start else [b_start]

        for t_model in turbine_choices:
            counts = [0] if t_model == 'None' else t_count_steps
            for t_count in counts:
                if t_count < min_turbines: continue
                if t_model != 'None' and t_count == 0 and min_turbines > 0:
                    continue
                for s in s_steps:
                    if s < min_solar: continue
                    for b in b_steps:
                        if b < min_bess: continue
                        if t_count == 0 and s == 0 and b == 0: continue
                        capex, _, _, _, _ = self.get_capex(t_model, t_count, s, b)
                        annual_opex, _, _, _ = self.get_opex(capex)
                        out = self.evaluate_combination(t_model, t_count, s, b, rep_days)
                        if out is None: continue
                        revenue, co2, btm_e, btm_g, curt, curt_cost = out
                        net_profit = revenue - annual_opex
                        irr = npf.irr([-capex] + [net_profit] * self.project_life_years) if net_profit > 0 else 0
                        npv = npf.npv(self.discount_rate, [-capex] + [net_profit] * self.project_life_years)
                        payback = capex / net_profit if net_profit > 0 else 99
                        stage1_results.append({'Turbine': f"{t_count}x {t_model}" if t_count > 0 else 'No Wind',
                                               't_model_raw': t_model, 't_count_raw': t_count,
                                               'Solar_MW': s, 'BESS_MWh': b,
                                               'CAPEX_M': capex/1e6, 'OPEX_k': annual_opex/1e3, 'Profit_k': net_profit/1e3,
                                               'Payback': payback, 'IRR': irr*100, 'NPV10_M': npv/1e6,
                                               'CO2_T': co2, 'Elec_Offset_MWh': btm_e/1000, 'Gas_Offset_MWh': btm_g/1000,
                                               'Curtailed_MWh': curt/1000, 'Min_Elec_PPA': 0}
                                              )

        df_stage1 = pd.DataFrame(stage1_results)
        if df_stage1.empty:
            return pd.DataFrame(), None

        if optimization_metric == 'IRR':
            best_stage1 = df_stage1.sort_values('IRR', ascending=False).iloc[0]
        elif optimization_metric == 'NPV':
            best_stage1 = df_stage1.sort_values('NPV10_M', ascending=False).iloc[0]
        else:
            best_stage1 = df_stage1.sort_values('Payback', ascending=True).iloc[0]

        # Stage 2: fine search around the first-stage best
        candidates = []
        best_t = int(best_stage1['t_count_raw'])
        best_s = int(best_stage1['Solar_MW'])
        best_b = int(best_stage1['BESS_MWh'])

        for t_model in turbine_choices:
            counts = [0] if t_model == 'None' else list({max(min_turbines, best_t - 1), best_t, min(max_turbines, best_t + 1)})
            for t_count in counts:
                if t_count < min_turbines: continue
                for s in [max(min_solar, best_s - 1), best_s, min(max_solar, best_s + 1)]:
                    if s < min_solar: continue
                    for b in [max(min_bess, best_b - 1), best_b, min(max_bess, best_b + 1)]:
                        if b < min_bess: continue
                        if t_count == 0 and s == 0 and b == 0: continue
                        candidates.append((t_model, t_count, s, b))

        final_results = []
        for t_model, t_count, s, b in set(candidates):
            capex, _, _, _, _ = self.get_capex(t_model, t_count, s, b)
            annual_opex, _, _, _ = self.get_opex(capex)
            out = self.evaluate_combination(t_model, t_count, s, b, rep_days)
            if out is None: continue
            revenue, co2, btm_e, btm_g, curt, curt_cost = out
            net_profit = revenue - annual_opex
            irr = npf.irr([-capex] + [net_profit] * self.project_life_years) if net_profit > 0 else 0
            npv = npf.npv(self.discount_rate, [-capex] + [net_profit] * self.project_life_years)
            payback = capex / net_profit if net_profit > 0 else 99
            final_results.append({'Turbine': f"{t_count}x {t_model}" if t_count > 0 else 'No Wind',
                                  't_model_raw': t_model, 't_count_raw': t_count,
                                  'Solar_MW': s, 'BESS_MWh': b,
                                  'CAPEX_M': capex/1e6, 'OPEX_k': annual_opex/1e3, 'Profit_k': net_profit/1e3,
                                  'Payback': payback, 'IRR': irr*100, 'NPV10_M': npv/1e6,
                                  'CO2_T': co2, 'Elec_Offset_MWh': btm_e/1000, 'Gas_Offset_MWh': btm_g/1000,
                                  'Curtailed_MWh': curt/1000, 'Min_Elec_PPA': 0}
                                 )

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

    def optimize_milp(self, rep_days, t_model="V90 3MW", max_turbines=10, max_solar=10, max_bess=10, require_wind=False, require_bess=False):
        """
        MILP co-optimization of sizing and dispatch. Requires Gurobi solver in CVXPY.
        """
        ...

    def optimize_milp(self, rep_days, t_model="V90 3MW", max_turbines=10, max_solar=10, max_bess=10, require_wind=False, require_bess=False):
        """
        MILP co-optimization of sizing and dispatch. Requires Gurobi solver in CVXPY.
        """
        # 变量设置
        t_count = cp.Variable(integer=True, nonneg=True)
        s_mw = cp.Variable(integer=True, nonneg=True)
        b_mwh = cp.Variable(integer=True, nonneg=True)

        # 风、光出力参考
        if t_model not in self.wind_solar_ref_data and t_model != "None":
            raise ValueError(f"Unknown turbine model: {t_model}")

        p_wind_profile = self.wind_solar_ref_data.get(t_model, np.zeros(24))
        p_solar_profile = self.wind_solar_ref_data.get("solar_per_mw", np.zeros(24))

        D = len(rep_days)
        T = 24

        # 运行变量（按天*小时）
        p_ch = cp.Variable((D, T), nonneg=True)
        p_dis = cp.Variable((D, T), nonneg=True)
        soc = cp.Variable((D, T), nonneg=True)
        p_btm_elec = cp.Variable((D, T), nonneg=True)
        p_gas_replaced = cp.Variable((D, T), nonneg=True)
        p_grid_sell = cp.Variable((D, T), nonneg=True)
        p_grid_buy = cp.Variable((D, T), nonneg=True)
        p_curtail = cp.Variable((D, T), nonneg=True)

        z_bess = cp.Variable((D, T), nonneg=True)
        z_grid = cp.Variable((D, T), nonneg=True)

        constraints = [
            t_count <= max_turbines,
            s_mw <= max_solar,
            b_mwh <= max_bess,
            t_count >= 1 if require_wind else t_count >= 0,
            b_mwh >= 1 if require_bess else b_mwh >= 0,
            t_count <= max_turbines,
            s_mw <= max_solar,
            b_mwh <= max_bess
        ]

        export_limit_kw = self.params.get('export_limit_kw', 5000)
        grid_buy_limit_kw = export_limit_kw * 0.8
        bess_max_power_kw = b_mwh * 1000 * 0.5
        cap_kwh = b_mwh * 1000

        for d, day in enumerate(rep_days):
            # 风光实际输出 (随大小线性)
            p_wind = p_wind_profile * t_count
            p_solar = p_solar_profile * s_mw * self.params['solar_efficiency']

            constraints += [
                p_wind + p_solar + p_dis[d, :] + p_grid_buy[d, :] == p_btm_elec[d, :] + p_gas_replaced[d, :] + p_grid_sell[d, :] + p_ch[d, :] + p_curtail[d, :],
                p_btm_elec[d, :] >= day['elec_load'],
                p_btm_elec[d, :] <= day['elec_load'],
                p_gas_replaced[d, :] >= day['gas_load'],
                p_gas_replaced[d, :] <= day['gas_load'],
                p_grid_sell[d, :] <= export_limit_kw,
                p_grid_buy[d, :] <= grid_buy_limit_kw,
                p_curtail[d, :] <= p_wind + p_solar,
                p_grid_sell[d, :] <= export_limit_kw * z_grid[d, :],
                p_grid_buy[d, :] <= grid_buy_limit_kw * (1 - z_grid[d, :]),
                z_grid[d, :] >= 0,
                z_grid[d, :] <= 1
            ]

            if b_mwh.value is not None and b_mwh.value < 1:
                constraints += [p_ch[d, :] == 0, p_dis[d, :] == 0, soc[d, :] == 0]
            else:
                constraints += [
                    soc[d, :] <= cap_kwh * 0.9,
                    soc[d, :] >= cap_kwh * 0.1,
                    p_ch[d, :] <= bess_max_power_kw,
                    p_dis[d, :] <= bess_max_power_kw,
                    p_ch[d, :] <= bess_max_power_kw * (1 - z_bess[d, :]),
                    p_dis[d, :] <= bess_max_power_kw * z_bess[d, :],
                    z_bess[d, :] >= 0,
                    z_bess[d, :] <= 1
                ]

                for t in range(1, T):
                    constraints.append(soc[d, t] == soc[d, t-1] + p_ch[d, t]*self.bess_eff - p_dis[d, t]/self.bess_eff)

        # 目标函数（年收益）
        annual_revenue = 0
        for d, day in enumerate(rep_days):
            weight = day.get('weight', 1)
            annual_revenue += (cp.sum(p_btm_elec[d, :] * self.params['p_galetech']) +
                               cp.sum(p_gas_replaced[d, :] * (self.params['p_gas'] + self.gas_carbon_factor * self.params['p_carbon'] / 1000)) +
                               cp.sum(p_grid_sell[d, :] * self.params['p_sell']) -
                               cp.sum(p_grid_buy[d, :] * self.params['p_sell'] * 0.8) -
                               cp.sum(p_dis[d, :] * (b_mwh * self.params['cost_bess_mwh'] / (self.bess_cycles * b_mwh * 1000) if b_mwh > 0 else 0))) * weight

        prob = cp.Problem(cp.Maximize(annual_revenue), constraints)
        prob.solve(solver=cp.GUROBI, verbose=False)

        return {
            't_model': t_model,
            't_count': int(round(t_count.value)) if t_count.value is not None else 0,
            'Solar_MW': int(round(s_mw.value)) if s_mw.value is not None else 0,
            'BESS_MWh': int(round(b_mwh.value)) if b_mwh.value is not None else 0,
            'annual_revenue': prob.value
        }
        
        # per-cycle battery wear cost (€/kWh discharged)
        bess_wear_cost = (b_mwh * self.params['cost_bess_mwh']) / (self.bess_cycles * b_mwh * 1000) if b_mwh > 0 else 0
        
        # core economic input for gas displacement value (€/kWh)
        gas_avoided_cost = self.params['p_gas'] + (self.gas_carbon_factor * self.params['p_carbon'] / 1000)
        
        export_limit_kw = self.params.get('export_limit_kw', 5000)
        bess_max_power_kw = (b_mwh * 1000) * 0.5 if b_mwh > 0 else 0

        all_traces = []

        for day_idx, day in enumerate(rep_days):
            T = len(day['elec_load'])
            p_ch, p_dis, soc = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
            p_btm_elec = cp.Variable(T, nonneg=True) 
            p_gas_replaced = cp.Variable(T, nonneg=True) 
            p_grid_sell = cp.Variable(T, nonneg=True)
            p_grid_buy = cp.Variable(T, nonneg=True)  # Grid purchase (backup power)
            p_curtail = cp.Variable(T, nonneg=True)
            # Battery mutual exclusion variables (z[t]=1 for discharging, z[t]=0 for charging)
            z_bess = cp.Variable(T, nonneg=True) if b_mwh > 0 else None
            
            p_wind = np.zeros(T)
            if t_model in self.wind_solar_ref_data and t_model != "None":
                p_wind = self.wind_solar_ref_data[t_model] * t_count  # kW output * turbine count
            
            p_solar = np.zeros(T)
            if 'solar_per_mw' in self.wind_solar_ref_data and s_mw > 0:
                p_solar = self.wind_solar_ref_data['solar_per_mw'] * s_mw * self.params['solar_efficiency']  # kW per MW * capacity * efficiency
            
            # Define grid purchase upper limit (contractual grid capacity, default 80% of export limit)
            grid_buy_limit_kw = export_limit_kw * 0.8
            
            # Grid mutual exclusion variable (z_grid[t]=1 for selling, z_grid[t]=0 for buying)
            z_grid = cp.Variable(T, nonneg=True)
            
            constraints = [
                # Energy balance: supply = demand
                p_wind + p_solar + p_dis + p_grid_buy == p_btm_elec + p_gas_replaced + p_grid_sell + p_ch + p_curtail,
                # Must satisfy customer loads (lower bound)
                p_btm_elec >= day['elec_load'],
                p_gas_replaced >= day['gas_load'],
                # Cannot exceed customer loads (upper bound)
                p_gas_replaced <= day['gas_load'],
                # Grid constraints
                p_grid_sell <= export_limit_kw,
                p_grid_buy <= grid_buy_limit_kw,
                # Grid mutual exclusion: cannot buy and sell simultaneously (Big-M formulation)
                p_grid_sell <= export_limit_kw * z_grid,
                p_grid_buy <= grid_buy_limit_kw * (1 - z_grid),
                # Binary constraint for grid mode (selling or buying)
                z_grid >= 0,
                z_grid <= 1,
                # Renewable curtailment constraint: curtailment cannot exceed available generation
                p_curtail <= p_wind + p_solar
            ]
            
            if b_mwh > 0:
                cap_kwh = b_mwh * 1000
                constraints += [
                    # SOC constraints
                    soc <= cap_kwh * 0.9, 
                    soc >= cap_kwh * 0.1,
                    soc[0] == cap_kwh * 0.5, 
                    soc[T-1] == cap_kwh * 0.5,
                    # Battery power constraints
                    p_ch <= bess_max_power_kw, 
                    p_dis <= bess_max_power_kw,
                    # Battery mutual exclusion: cannot charge and discharge simultaneously (Big-M formulation)
                    p_ch <= bess_max_power_kw * (1 - z_bess),
                    p_dis <= bess_max_power_kw * z_bess,
                    # z_bess=1: discharging mode, z_bess=0: charging mode
                    # Binary constraint for battery mode
                    z_bess >= 0,
                    z_bess <= 1
                ]
                for t in range(1, T): 
                    constraints.append(soc[t] == soc[t-1] + p_ch[t]*self.bess_eff - p_dis[t]/self.bess_eff)
            else:
                constraints += [p_ch == 0, p_dis == 0, soc == 0]

            # Revenue components: customer electricity sale + gas displacement equivalent + grid export - grid purchase cost
            rev_elec = cp.sum(p_btm_elec * (self.params['p_galetech'] * ppa_shock))
            rev_gas = cp.sum(p_gas_replaced * (gas_avoided_cost * ppa_shock))
            rev_grid = cp.sum(p_grid_sell * self.params['p_sell'])
            # Grid purchase cost (assume 80% of sell price as wholesale rate)
            grid_purchase_cost = cp.sum(p_grid_buy * (self.params['p_sell'] * 0.8))
            
            obj = cp.Maximize(rev_elec + rev_gas + rev_grid - grid_purchase_cost - cp.sum(p_dis * bess_wear_cost))
            prob = cp.Problem(obj, constraints)
            
            try:
                prob.solve(solver=cp.GUROBI)
                if prob.status not in ["infeasible", "unbounded"]:
                    annual_revenue += prob.value * day['weight']
                    annual_co2_saved += np.sum(p_gas_replaced.value) * self.gas_carbon_factor / 1000 * day['weight']
                    annual_btm_elec_kwh += np.sum(p_btm_elec.value) * day['weight']
                    annual_btm_gas_kwh += np.sum(p_gas_replaced.value) * day['weight']
                    
                    curtailed_daily = np.sum(p_curtail.value)
                    annual_curtailed_kwh += curtailed_daily * day['weight']
                    annual_curtail_cost += curtailed_daily * self.params['p_sell'] * day['weight']

                    if return_traces:
                        for t in range(T):
                            all_traces.append({
                                "Day_Type": f"Scenario_{day_idx+1}", "Hour": t,
                                "Elec_Demand_kW": day['elec_load'][t], "Gas_Demand_kW": day['gas_load'][t],
                                "Wind_Gen_kW": p_wind[t], "Solar_Gen_kW": p_solar[t],
                                "BESS_SoC_kWh": soc.value[t] if b_mwh>0 else 0,
                                "Elec_Supply_kW": p_btm_elec.value[t], "Gas_Replaced_kW": p_gas_replaced.value[t],
                                "Grid_Export_kW": p_grid_sell.value[t], "Curtailed_kW": p_curtail.value[t]
                            })
            except: pass

        if return_traces: return pd.DataFrame(all_traces)
        return annual_revenue, annual_co2_saved, annual_btm_elec_kwh, annual_btm_gas_kwh, annual_curtailed_kwh, annual_curtail_cost

# ==========================================
# 2. Data preparation engine
# ==========================================
def load_custom_typical_days(df=None, custom_weights=None):
    """
    加载客户的负荷数据（电、气每小时需求）。
    风光发电参考数据是内置的，不需要上传。
    """
    rep_days = []
    if df is not None:
        # 从客户上传的数据加载负荷
        num_days = len(df) // 24
        for i in range(num_days):
            day_data = df.iloc[i*24 : (i+1)*24]
            
            # 读取电负荷
            if 'elec_load' in df.columns:
                elec_load = day_data['elec_load'].values
            else:
                st.warning("未找到 'elec_load' 列，使用默认值")
                t = np.arange(24)
                elec_load = 1200 + 400*np.sin((t-8)*np.pi/12)
            
            # 读取气负荷
            if 'gas_load' in df.columns:
                gas_load = day_data['gas_load'].values
            else:
                st.warning("未找到 'gas_load' 列，使用默认值")
                t = np.arange(24)
                gas_load = 200 + 50*np.random.rand(24)
            
            rep_days.append({
                'elec_load': elec_load,
                'gas_load': gas_load,
                'weight': custom_weights[i] if custom_weights else 365/num_days
            })
    else:
        # 默认数据：3个典型日
        weights = custom_weights if custom_weights else [90, 90, 185]
        t = np.arange(24)
        rep_days.append({
            'elec_load': 1200 + 400*np.sin((t-8)*np.pi/12), 
            'gas_load': 200 + 50*np.random.rand(24), 
            'weight': weights[0] if len(weights)>0 else 90
        })
        rep_days.append({
            'elec_load': 800 + 200*np.sin((t-6)*np.pi/12), 
            'gas_load': 2000 + 800*np.cos((t-12)*np.pi/12), 
            'weight': weights[1] if len(weights)>1 else 90
        })
        rep_days.append({
            'elec_load': 900 + 150*np.sin(t/4), 
            'gas_load': 600 + 100*np.random.rand(24), 
            'weight': weights[2] if len(weights)>2 else 185
        })
    return rep_days

def load_wind_solar_ref_data():
    """
    Load wind and solar power reference data from CSV file.
    The CSV file contains hourly power outputs per turbine model and per MW of solar.
    """
    import os
    csv_path = os.path.join(os.path.dirname(__file__), 'Typical_Daily_Power_Profile_2019.csv')
    
    ref_data = {}
    
    try:
        # Read the CSV file with wind turbine hourly outputs
        df = pd.read_csv(csv_path)
        
        # Extract turbine columns (skip the 'Hour' column)
        turbine_models = [col for col in df.columns if col != 'Hour']
        
        for model in turbine_models:
            ref_data[model] = df[model].values
            
        # Add solar reference data (kW per MW at each hour)
        # This is based on typical 1MW PV system output profile
        ref_data['solar_per_mw'] = np.array([0, 0, 0, 0, 19.9, 96.0, 239.7, 448.4, 706.3, 979.5, 1203.2, 1333.4, 1348.4, 1245.7, 1042.3, 775.5, 509.0, 286.2, 125.5, 32.9, 0, 0, 0, 0])
        
    except FileNotFoundError:
        st.warning(f"Reference data file not found: {csv_path}. Using default values.")
        # Fallback to default data if CSV not found
        ref_data = {
            "V90 3MW": np.array([1022.742384, 1033.757863, 1031.353288, 1046.004247, 1043.791562, 1031.096, 1028.068603, 1030.901151, 1011.62874, 976.9499452, 985.9115342, 973.6334521, 975.104274, 962.8912877, 961.9958356, 963.9593973, 946.1557808, 954.2829863, 918.2227123, 935.9376438, 939.8587671, 965.5370685, 975.8890137, 1007.926795]),
            "EWT 500kW": np.array([170.457064, 172.2929772, 171.8922147, 174.3340412, 173.9652603, 171.8493333, 171.3447672, 171.8168585, 168.60479, 162.8249909, 164.318589, 162.272242, 162.517379, 160.4818813, 160.3326393, 160.6598996, 157.6926301, 159.0471644, 153.0371187, 155.9896073, 156.6431279, 160.9228448, 162.648169, 167.9877992]),
            "E82 2.3MW": np.array([784.1024944, 792.547695, 790.7041875, 801.9365894, 800.2401975, 790.5069333, 788.185929, 790.3575491, 775.582034, 748.994958, 755.8655096, 746.4523133, 747.5799434, 738.2166539, 737.5301406, 739.0355379, 725.3860986, 731.6169562, 703.9707461, 717.5521936, 720.5583881, 740.2450859, 748.1815772, 772.7438762]),
            "E138 4.26MW": np.array([1452.294185, 1467.936165, 1464.521669, 1485.326031, 1482.184018, 1464.15632, 1459.857416, 1463.879634, 1436.512811, 1387.268922, 1399.994379, 1382.559502, 1384.648069, 1367.305629, 1366.034087, 1368.822344, 1343.541209, 1355.081841, 1303.876251, 1329.031454, 1334.599449, 1371.062637, 1385.762399, 1431.256049]),
            "EWT 1MW": np.array([340.914128, 344.5859543, 343.7844293, 348.6680823, 347.9305207, 343.6986667, 342.6895343, 343.633717, 337.20958, 325.6499817, 328.6371781, 324.544484, 325.034758, 320.9637626, 320.6652785, 321.3197991, 315.3852603, 318.0943288, 306.0742374, 311.9792146, 313.2862557, 321.8456895, 325.2963379, 335.9755983]),
            "E115 4.2MW": np.array([1431.839338, 1447.261008, 1443.894603, 1464.405946, 1461.308187, 1443.5344, 1439.296044, 1443.261611, 1416.280236, 1367.729923, 1380.276148, 1363.086833, 1365.145984, 1348.047803, 1346.79417, 1349.543156, 1324.618093, 1335.996181, 1285.511797, 1310.312701, 1315.802274, 1351.751896, 1366.244619, 1411.097513]),
            "solar_per_mw": np.array([0, 0, 0, 0, 19.9, 96.0, 239.7, 448.4, 706.3, 979.5, 1203.2, 1333.4, 1348.4, 1245.7, 1042.3, 775.5, 509.0, 286.2, 125.5, 32.9, 0, 0, 0, 0])
        }
    
    return ref_data

# ==========================================
# 3. Streamlit UI and report generation
# ==========================================
st.set_page_config(page_title="Galetech BOO Bankable Report", layout="wide")
st.title("📑 Galetech BOO Optimiser & Bankability Assistant")

with st.sidebar:
    st.header("📂 Data & Profile Setup")
    st.info("ℹ️ Wind & Solar output data are embedded in the model. Please only upload hourly electricity and gas load profiles.")
    uploaded_file = st.file_uploader("Upload Customer Hourly Load Profile (CSV/Excel with columns: elec_load, gas_load; one row per hour)", type=['csv', 'xlsx'])
    df_customer = None
    custom_weights = []
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df_customer = pd.read_excel(uploaded_file)
        else:
            df_customer = pd.read_csv(uploaded_file)
        num_days = len(df_customer) // 24
        for i in range(num_days): custom_weights.append(st.number_input(f"Day {i+1} Weight:", min_value=1, max_value=365, value=365//num_days))
    else:
        st.info("Using Demo Data (3 Typical Days)")
        custom_weights = [st.number_input("Summer Weight:", value=90), st.number_input("Winter Weight:", value=90), st.number_input("Spring/Autumn Weight:", value=185)]

    st.divider()
    st.header("💷 Commercial & Constraints")
    p_retail_input = st.number_input("Customer Current Elec Price (€/MWh)", value=130.0) 
    p_gas_input = st.number_input("Customer Current Gas Price (€/MWh)", value=50.0) 
    p_galetech_input = st.number_input("Target BOO PPA Price (Elec) (€/MWh)", value=100.0)
    p_sell_input = st.number_input("Grid Export Price (€/MWh)", value=50.0)
    p_carbon_input = st.number_input("Carbon Value (€/Tonne)", value=65.0)
    target_irr_input = st.number_input("Target IRR for Minimum PPA (%)", value=10.0)

    st.header("📌 Strategy Filters")
    require_wind = st.checkbox("Require at least one wind turbine", value=False)
    require_bess = st.checkbox("Require at least one battery storage", value=False)
    optimization_metric = st.selectbox("Select optimization metric", ["Payback","IRR","NPV"], index=0)

    with st.expander("🛠️ Advanced CAPEX Assumptions", expanded=False):
        c_solar_mw = st.number_input("Solar Equip Cost (€/MW)", value=1000000)
        c_bess_mwh = st.number_input("BESS Equip Cost (€/MWh)", value=300000)
        solar_efficiency = st.number_input("Solar Efficiency (%)", value=20.0) / 100  # Convert to decimal
        cv_solar = st.number_input("Civils - Solar (€/Site)", value=300000)
        cv_bess = st.number_input("Civils - BESS (€/Site)", value=300000)
        elec_conn = st.number_input("Electrical Works (€/Source)", value=400000)
        pm_rate = st.number_input("Project Mngt Rate (%)", value=10.0) / 100
        ins_rate = st.number_input("Insurance Rate (% CAPEX)", value=1.5) / 100
        maint_rate = st.number_input("Maintenance Rate (% CAPEX)", value=3.5) / 100
        lease_cost = st.number_input("Land Lease (€/Year)", value=100000)

    st.divider()
    st.header("📐 Physical Search Limits")
    min_turbines = st.number_input("Min Turbines", 0, 40, 0)
    min_solar = st.number_input("Min Solar (MW)", 0, 40, 0)
    min_bess = st.number_input("Min BESS (MWh)", 0, 40, 0)
    max_turbines = st.slider("Max Turbines", 0, 40, 10)
    max_solar = st.slider("Max Solar PV (MW)", 0, 40, 10, step=2)
    max_bess = st.slider("Max BESS (MWh)", 0, 40, 10, step=1)
    export_limit_input = st.number_input("Grid Export Limit (kW)", value=5000, min_value=0, max_value=20000)
    run_btn = st.button("🚀 Generate Bankable Report", type="primary", use_container_width=True)

if run_btn:
    if p_galetech_input <= 0 or p_galetech_input >= p_retail_input:
        st.error(f"🛑 Commercial Error: The BOO PPA Price must be between €0 and €{p_retail_input}.")
        st.stop()
        
    rep_days = load_custom_typical_days(df_customer, custom_weights)
    
    # 加载内置的风光参考数据
    wind_solar_ref_data = load_wind_solar_ref_data()
    
    optimizer_params = {
        'p_galetech': p_galetech_input / 1000, 'p_gas': p_gas_input / 1000, 
        'p_sell': p_sell_input / 1000, 'p_carbon': p_carbon_input,
        'export_limit_kw': export_limit_input,
        'cost_solar_mw': c_solar_mw, 'cost_bess_mwh': c_bess_mwh, 'solar_efficiency': solar_efficiency,
        'civil_solar': cv_solar, 'civil_bess': cv_bess,
        'elec_per_source': elec_conn, 'pm_rate': pm_rate, 'insurance_rate': ins_rate, 'maintenance_rate': maint_rate, 'land_lease': lease_cost
    }
    optimizer = GaletechAssetOptimizer(optimizer_params, wind_solar_ref_data)
    
    turbine_choices = ["None", "EWT 500kW", "EWT 1MW", "E82 2.3MW", "V90 3MW", "E115 4.2MW", "E138 4.26MW"]

    with st.spinner("Executing Two-Stage Optimization (capacity first, dispatch second)..."):
        df_res, best = optimizer.two_stage_optimization(
            rep_days, turbine_choices,
            min_turbines, max_turbines,
            min_solar, max_solar,
            min_bess, max_bess,
            export_limit_input,
            optimization_metric=optimization_metric
        )

    if df_res.empty or best is None:
        st.error("No commercially viable configurations found. Try adjusting limits or parameters.")
        st.stop()

    # Stage 2 final dispatch trace for best option
    best_trace = optimizer.evaluate_combination(
        best['t_model_raw'], best['t_count_raw'], best['Solar_MW'], best['BESS_MWh'], rep_days, return_traces=True
    )

    # compute min_ppa for best option
    capex, _, _, _, _ = optimizer.get_capex(best['t_model_raw'], best['t_count_raw'], best['Solar_MW'], best['BESS_MWh'])
    annual_opex, _, _, _ = optimizer.get_opex(capex)
    revenue = best['Profit_k']*1000 + annual_opex
    btm_e = best['Elec_Offset_MWh']*1000
    target_irr = target_irr_input / 100
    pv_factor = (1 - (1 + target_irr)**-optimizer.project_life_years) / target_irr
    req_annual_profit = capex / pv_factor if pv_factor > 0 else 0
    req_total_revenue = req_annual_profit + annual_opex
    rev_fixed_other = revenue - (btm_e * (p_galetech_input/1000)) if btm_e > 0 else 0
    min_ppa_elec = (req_total_revenue - rev_fixed_other) / btm_e * 1000 if btm_e > 0 else 0
    best['Min_Elec_PPA'] = min_ppa_elec

    df_res['Min_Elec_PPA'] = df_res['Min_Elec_PPA'].fillna(0)
    df_res.loc[df_res['t_model_raw']==best['t_model_raw'],'Min_Elec_PPA'] = min_ppa_elec
    best['Min_Elec_PPA'] = min_ppa_elec

    # ----- 追加：最优方案年度现金流报表 -----
    best_capex = best['CAPEX_M'] * 1e6
    best_opex = best['OPEX_k'] * 1e3
    best_netprofit = best['Profit_k'] * 1e3
    best_revenue = best_netprofit + best_opex
    cashflow_rows = [{'Year':0, 'CAPEX': -best_capex, 'Revenue': 0.0, 'OPEX': 0.0, 'NetProfit': -best_capex, 'CashFlow': -best_capex}]
    for y in range(1, optimizer.project_life_years + 1):
        cashflow_rows.append({'Year': y, 'CAPEX': 0.0, 'Revenue': best_revenue, 'OPEX': best_opex, 'NetProfit': best_netprofit, 'CashFlow': best_netprofit})
    df_cashflow = pd.DataFrame(cashflow_rows)

    # ==============================================================
    # Core logic update: optional sorting strategy (Payback / IRR / NPV)
    # ==============================================================
    df_viable = df_res[(df_res['Payback'] > 0) & (df_res['Payback'] < optimizer.project_life_years)]
    if optimization_metric == "IRR":
        df_viable = df_viable.sort_values('IRR', ascending=False)
    elif optimization_metric == "NPV":
        df_viable = df_viable.sort_values('NPV10_M', ascending=False)
    else:
        df_viable = df_viable.sort_values('Payback', ascending=True)
    df_viable = df_viable.reset_index(drop=True)

    if df_viable.empty:
        st.error("No commercially viable configurations found. Try adjusting limits or parameters.")
        st.stop()

    best = df_viable.iloc[0]
    st.session_state['best_config'] = best 
    st.session_state['rep_days'] = rep_days
    
    df_traces = optimizer.evaluate_combination(best['t_model_raw'], best['t_count_raw'], best['Solar_MW'], best['BESS_MWh'], rep_days, return_traces=True)
    csv_buffer = io.StringIO()
    df_traces.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')

    st.success("✅ Optimization Complete. Bankable Report Generated.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Executive Summary", "2. Cost Breakdown", "3. Benchmarking", "4. Monte Carlo Risk", "5. Auditable Pack"])
    
    with tab1:
        st.header("Executive Summary")
        title_map = {'Payback':'Shortest Payback','IRR':'Highest IRR','NPV':'Highest NPV'}
        st.markdown(f"### 🎯 Headline Technical Recommendation ({title_map.get(optimization_metric,'Payback')})")
        if optimization_metric == 'Payback':
            st.markdown("The optimal renewable solution for this site to **minimize the payback period (fastest capital recovery)** is:")
        elif optimization_metric == 'IRR':
            st.markdown("The optimal renewable solution for this site to **maximize IRR (strongest return)** is:")
        else:
            st.markdown("The optimal renewable solution for this site to **maximize NPV (best value creation)** is:")
        st.markdown(f"- **Wind Asset:** {best['Turbine']} | **Solar PV:** {best['Solar_MW']} MW | **BESS:** {best['BESS_MWh']} MWh")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Payback Period", f"{best['Payback']:.1f} Years")
        col2.metric("Project IRR", f"{best['IRR']:.1f} %")
        col3.metric("NPV @ 10%", f"€ {best['NPV10_M']:.2f} M")
        col4.metric("Gas Offset", f"{int(best['Gas_Offset_MWh'])} MWh")

        st.info(f"**Suggested Minimum Elec PPA Price to achieve {target_irr_input}% IRR:** € {best['Min_Elec_PPA']:.2f} / MWh")
        
    with tab2:
        st.header("Financial Performance & Breakdown")
        show_cols = ['Turbine', 'Solar_MW', 'BESS_MWh', 'Payback', 'IRR', 'NPV10_M', 'CAPEX_M', 'Elec_Offset_MWh', 'Gas_Offset_MWh', 'Curtailed_MWh']
        st.dataframe(df_viable[show_cols].head(10).style.format({
            'CAPEX_M': '{:.2f}', 'NPV10_M': '{:.2f}', 'IRR': '{:.1f}%', 'Payback': '{:.1f} Yrs',
            'Elec_Offset_MWh': '{:.0f}', 'Gas_Offset_MWh': '{:.0f}', 
            'Curtailed_MWh': '{:.0f}'
        }))

        st.subheader("最佳方案年度现金流 (估算)")
        st.markdown("（注：阶段2输出为每年恒定估计，接口与财务展现）")
        st.dataframe(df_cashflow.style.format({
            'CAPEX': '€{:,.0f}', 'Revenue': '€{:,.0f}', 'OPEX': '€{:,.0f}', 'NetProfit': '€{:,.0f}', 'CashFlow': '€{:,.0f}'
        }))

    with tab3:
        st.header("Technology Benchmarking")
        benchmarks = []
        benchmarks.append(best.to_dict()); benchmarks[-1]['Category'] = "⭐ Recommended Optimum"
        
        no_bess = df_res[(df_res['BESS_MWh'] == 0) & ((df_res['Solar_MW'] > 0) | (df_res['Turbine'] != "No Wind"))].sort_values('Payback', ascending=True)
        if not no_bess.empty:
            b = no_bess.iloc[0].to_dict(); b['Category'] = "❌ Renewables Only (No Battery)"; benchmarks.append(b)
            
        bess_only = df_res[(df_res['BESS_MWh'] > 0) & (df_res['Solar_MW'] == 0) & (df_res['Turbine'] == "No Wind")].sort_values('Payback', ascending=True)
        if not bess_only.empty:
            b = bess_only.iloc[0].to_dict(); b['Category'] = "❌ Battery Only (No Renewables)"; benchmarks.append(b)

        df_bench = pd.DataFrame(benchmarks)
        st.dataframe(df_bench[['Category', 'Turbine', 'Solar_MW', 'BESS_MWh', 'Payback', 'IRR', 'NPV10_M', 'CAPEX_M']].style.format({'CAPEX_M': '{:.2f}', 'NPV10_M': '{:.2f}', 'IRR': '{:.1f}%', 'Payback': '{:.1f}'}))

        # 业务化热力图（指标 vs 规模）
        st.subheader("可营业性热力图（同风机类型，Solar vs BESS）")
        selected_wind = best['t_model_raw']
        sub = df_res[df_res['t_model_raw'] == selected_wind]
        if sub.empty:
            sub = df_res

        for metric, cmap in [('Payback', 'viridis'), ('IRR', 'RdYlGn'), ('NPV10_M', 'coolwarm')]:
            pivot = sub.pivot_table(index='BESS_MWh', columns='Solar_MW', values=metric, aggfunc='mean')
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap=cmap)
            ax.set_title(f"{metric} 热力图 (风机={selected_wind})")
            ax.set_ylabel('BESS MWh')
            ax.set_xlabel('Solar MW')
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns.astype(int), rotation=45)
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index.astype(int))
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric)
            st.pyplot(fig)

    with tab4:
        st.header("Monte Carlo Uncertainty Analysis")
        if st.button("🎲 Run 50 Monte Carlo Simulations on Optimal Config"):
            mc_results = []
            progress = st.progress(0)
            best_conf = st.session_state['best_config']
            
            for i in range(50):
                wind_shock = np.random.normal(1.0, 0.10)
                capex_shock = np.random.normal(1.0, 0.05) 
                ppa_shock = np.random.normal(1.0, 0.05)   
                
                capex, _, _, _, _ = optimizer.get_capex(best_conf['t_model_raw'], best_conf['t_count_raw'], best_conf['Solar_MW'], best_conf['BESS_MWh'], capex_shock)
                annual_opex, _, _, _ = optimizer.get_opex(capex)
                
                rev, co2, btm_e, btm_g, curt, curt_cost = optimizer.evaluate_combination(
                    best_conf['t_model_raw'], best_conf['t_count_raw'], best_conf['Solar_MW'], best_conf['BESS_MWh'], 
                    st.session_state['rep_days'], wind_shock=wind_shock, ppa_shock=ppa_shock
                )
                
                profit = rev - annual_opex
                cashflows = [-capex] + [profit] * 20
                irr = npf.irr(cashflows) if profit > 0 else 0
                npv = npf.npv(0.10, cashflows)
                payback = capex / profit if profit > 0 else 99
                
                mc_results.append({'Iteration': i, 'Payback': payback, 'IRR': irr*100, 'Wind_Shock': wind_shock})
                progress.progress((i+1)/50)
                
            progress.empty()
            df_mc = pd.DataFrame(mc_results)
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Payback Distribution (P90 / P50 / P10)**")
                p90, p50, p10 = np.percentile(df_mc['Payback'], [90, 50, 10]) # for payback lower is better, P90 is conservative high value
                st.write(f"- **Conservative (P90):** {p90:.1f} Years")
                st.write(f"- **Expected (P50):** {p50:.1f} Years")
                st.write(f"- **Optimistic (P10):** {p10:.1f} Years")
            with c2:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df_mc['Payback'], bins=10, color='orange', edgecolor='black')
                ax.axvline(p50, color='red', linestyle='dashed', linewidth=1, label='P50 Expected')
                ax.set_title('Monte Carlo Payback Outcomes')
                ax.legend()
                st.pyplot(fig)

    with tab5:
        st.header("Auditable Calculation Pack")
        st.download_button("📥 Download Hourly Dispatch Pack (CSV)", data=csv_data, file_name=f"Galetech_Audit_Pack_{best['Turbine']}_{best['Solar_MW']}MW_{best['BESS_MWh']}MWh.csv", mime="text/csv")