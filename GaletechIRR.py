import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy_financial as npf
import io

# ==========================================
# 1. Core optimization engine (includes grid constraint and curtailment tracking)
# ==========================================
class GaletechAssetOptimizer:
    def __init__(self, params):
        self.params = params
        self.bess_eff = 0.95
        self.bess_cycles = 8000 
        self.gas_carbon_factor = 0.202 
        self.project_life_years = 20
        self.discount_rate = 0.10
        
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
        annual_curtailed_kwh = 0  # curtailed energy in kWh
        annual_curtail_cost = 0  # cost of curtailed energy in Euros
        
        bess_wear_cost = (b_mwh * self.params['cost_bess_mwh']) / (self.bess_cycles * b_mwh * 1000) if b_mwh > 0 else 0
        carbon_premium = self.gas_carbon_factor * self.params['p_carbon'] / 1000
        
        # Physical constraints: grid export cap and battery charge/discharge C-rate
        export_limit_kw = self.params.get('export_limit_kw', 5000)
        bess_max_power_kw = (b_mwh * 1000) * 0.5 if b_mwh > 0 else 0

        all_traces = []

        for day_idx, day in enumerate(rep_days):
            T = len(day['elec_load'])
            p_ch, p_dis, soc = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
            p_btm_supply, p_gas_replaced, p_grid_sell = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)
            p_curtail = cp.Variable(T, nonneg=True) 
            
            p_wind = day['wind_powers'].get(t_model, np.zeros(24)) * t_count
            p_solar = s_mw * day['solar_power_per_mw'] * self.params['solar_efficiency']  # day['solar_power_per_mw'] is kW output for 1MW solar, s_mw in MW
            
            constraints = [
                p_wind + p_solar + p_dis == p_btm_supply + p_grid_sell + p_ch + p_curtail,
                p_btm_supply <= day['elec_load'] + p_gas_replaced,
                p_gas_replaced <= day['gas_load'],
                p_grid_sell <= export_limit_kw 
            ]
            
            if b_mwh > 0:
                cap_kwh = b_mwh * 1000
                constraints += [
                    soc <= cap_kwh * 0.9, soc >= cap_kwh * 0.1,
                    soc[0] == cap_kwh * 0.5, soc[T-1] == cap_kwh * 0.5,
                    p_ch <= bess_max_power_kw, p_dis <= bess_max_power_kw
                ]
                for t in range(1, T): constraints.append(soc[t] == soc[t-1] + p_ch[t]*self.bess_eff - p_dis[t]/self.bess_eff)
            else:
                constraints += [p_ch == 0, p_dis == 0, soc == 0]

            rev_customer = cp.sum(p_btm_supply * (self.params['p_galetech'] * ppa_shock))
            rev_grid = cp.sum(p_grid_sell * self.params['p_sell'])
            val_carbon = cp.sum(p_gas_replaced * carbon_premium)
            
            obj = cp.Maximize(rev_customer + rev_grid + val_carbon - cp.sum(p_dis * bess_wear_cost))
            prob = cp.Problem(obj, constraints)
            
            try:
                prob.solve(solver=cp.OSQP)
                if prob.status not in ["infeasible", "unbounded"]:
                    annual_revenue += prob.value * day['weight']
                    annual_co2_saved += np.sum(p_gas_replaced.value) * self.gas_carbon_factor / 1000 * day['weight']
                    annual_btm_supply_kwh += np.sum(p_btm_supply.value) * day['weight']
                    
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
                                "BTM_Supply_kW": p_btm_supply.value[t], "Grid_Export_kW": p_grid_sell.value[t],
                                "Curtailed_kW": p_curtail.value[t]
                            })
            except: pass

        if return_traces: return pd.DataFrame(all_traces)
        return annual_revenue, annual_co2_saved, annual_btm_supply_kwh, annual_curtailed_kwh, annual_curtail_cost

# ==========================================
# 2. Data processing engine
# ==========================================
def load_custom_typical_days(df=None, custom_weights=None):
    rep_days = []
    if df is not None:
        num_days = len(df) // 24
        for i in range(num_days):
            day_data = df.iloc[i*24 : (i+1)*24]
            # 风机列：除了Hour, elec_load, gas_load, 1MW installed Solar PV外的列
            wind_cols = [col for col in df.columns if col not in ['Hour', 'elec_load', 'gas_load', '1MW installed Solar PV'] and 'Solar' not in col]
            rep_days.append({
                'elec_load': day_data['elec_load'].values if 'elec_load' in day_data.columns else np.zeros(24),
                'gas_load': day_data['gas_load'].values if 'gas_load' in day_data.columns else np.zeros(24),
                'wind_powers': {col: day_data[col].values for col in wind_cols},
                'solar_power_per_mw': day_data['1MW installed Solar PV'].values if '1MW installed Solar PV' in day_data.columns else np.zeros(24),
                'weight': custom_weights[i] if custom_weights else 365/num_days
            })
    else:
        weights = custom_weights if custom_weights else [90, 90, 185]
        t = np.arange(24)
        # 默认数据：保持elec_load和gas_load，wind_powers为模拟，solar_power_per_mw为模拟
        default_wind_powers = {
            "EWT 500kW": np.clip(3 + 2*np.sin(t/4), 0, None) * 500,  # 模拟kW输出
            "EWT 1MW": np.clip(3 + 2*np.sin(t/4), 0, None) * 1000,
            "E82 2.3MW": np.clip(8 + 3*np.cos(t/4), 0, None) * 2300,
            "V90 3MW": np.clip(8 + 3*np.cos(t/4), 0, None) * 3000,
            "E115 4.2MW": np.clip(5 + 2*np.sin(t/6), 0, None) * 4200,
            "E138 4.26MW": np.clip(5 + 2*np.sin(t/6), 0, None) * 4260
        }
        default_solar = np.clip(np.sin((t-6)*np.pi/12), 0, 1) * 1000  # 1MW光伏的kW输出
        rep_days.append({
            'elec_load': 1200 + 400*np.sin((t-8)*np.pi/12), 
            'gas_load': 200 + 50*np.random.rand(24), 
            'wind_powers': default_wind_powers, 
            'solar_power_per_mw': default_solar, 
            'weight': weights[0] if len(weights)>0 else 90
        })
        rep_days.append({
            'elec_load': 800 + 200*np.sin((t-6)*np.pi/12), 
            'gas_load': 2000 + 800*np.cos((t-12)*np.pi/12), 
            'wind_powers': default_wind_powers, 
            'solar_power_per_mw': default_solar * 0.4, 
            'weight': weights[1] if len(weights)>1 else 90
        })
        rep_days.append({
            'elec_load': 900 + 150*np.sin(t/4), 
            'gas_load': 600 + 100*np.random.rand(24), 
            'wind_powers': default_wind_powers, 
            'solar_power_per_mw': default_solar * 0.7, 
            'weight': weights[2] if len(weights)>2 else 185
        })
    return rep_days

# ==========================================
# 3. Streamlit UI and report generation
# ==========================================
st.set_page_config(page_title="Galetech BOO Bankable Report", layout="wide")
st.title("📑 Galetech BOO Optimiser & Bankability Assistant")

with st.sidebar:
    st.header("📂 Data & Profile Setup")
    uploaded_file = st.file_uploader("Upload Profile (CSV, multiples of 24 rows)", type=['csv'])
    df_customer = None
    custom_weights = []
    if uploaded_file is not None:
        df_customer = pd.read_csv(uploaded_file)
        num_days = len(df_customer) // 24
        for i in range(num_days): custom_weights.append(st.number_input(f"Day {i+1} Weight:", min_value=1, max_value=365, value=365//num_days))
    else:
        st.info("Using Demo Data (3 Typical Days)")
        custom_weights = [st.number_input("Summer Weight:", value=90), st.number_input("Winter Weight:", value=90), st.number_input("Spring/Autumn Weight:", value=185)]

    st.divider()
    st.header("💷 Commercial & Constraints")
    p_retail_input = st.number_input("Customer Current Retail Price (€/MWh)", value=130.0) 
    p_galetech_input = st.number_input("Target BOO PPA Price (€/MWh)", value=100.0)
    p_sell_input = st.number_input("Grid Export Price (€/MWh)", value=50.0)
    p_carbon_input = st.number_input("Carbon Value (€/Tonne)", value=65.0)
    target_irr_input = st.number_input("Target IRR for Minimum PPA (%)", value=10.0)
    
    # Add physical constraint control for grid export capacity
    export_limit_input = st.number_input("Grid Export Limit (kW)", value=5000, help="Maximum power allowed to be exported to the grid at any hour.")

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
    max_turbines = st.slider("Max Turbines", 0, 5, 1)
    max_solar = st.slider("Max Solar PV (MW)", 0, 20, 10, step=2)
    max_bess = st.slider("Max BESS (MWh)", 0, 20, 10, step=2)
    run_btn = st.button("🚀 Generate Bankable Report", type="primary", use_container_width=True)

if run_btn:
    if p_galetech_input <= 0 or p_galetech_input >= p_retail_input:
        st.error(f"🛑 Commercial Error: The BOO PPA Price must be between €0 and €{p_retail_input}.")
        st.stop()
        
    rep_days = load_custom_typical_days(df_customer, custom_weights)
    optimizer_params = {
        'p_galetech': p_galetech_input / 1000, 'p_sell': p_sell_input / 1000, 'p_carbon': p_carbon_input,
        'export_limit_kw': export_limit_input,
        'cost_solar_mw': c_solar_mw, 'cost_bess_mwh': c_bess_mwh, 'solar_efficiency': solar_efficiency, 'civil_solar': cv_solar, 'civil_bess': cv_bess,
        'elec_per_source': elec_conn, 'pm_rate': pm_rate, 'insurance_rate': ins_rate, 'maintenance_rate': maint_rate, 'land_lease': lease_cost
    }
    optimizer = GaletechAssetOptimizer(optimizer_params)
    
    turbine_choices = ["None", "EWT 500kW", "EWT 1MW", "E82 2.3MW", "V90 3MW", "E115 4.2MW", "E138 4.26MW"]
    res_list = []
    
    with st.spinner("Executing Mathematical Optimization (Co-optimising sizing, export & curtailment)..."):
        # Adjust loop to allow battery-only configuration for benchmarking comparisons
        for t_model in turbine_choices:
            counts = [0] if t_model == "None" else range(1, max_turbines + 1)
            for t_count in counts:
                for s in range(0, max_solar + 1, max(1, max_solar//5)): 
                    for b in range(0, max_bess + 1, max(1, max_bess//5)): 
                        if t_count == 0 and s == 0 and b == 0: continue 
                            
                        capex, eq, civ, elec, pm = optimizer.get_capex(t_model, t_count, s, b)
                        annual_opex, ins, maint, lease = optimizer.get_opex(capex)
                        
                        # -------------------------------------------------------------
                        # Fix: unpack 5 return variables from evaluator to avoid ValueError
                        # -------------------------------------------------------------
                        revenue, co2, btm_kwh, curtail_kwh, curtail_cost = optimizer.evaluate_combination(t_model, t_count, s, b, rep_days)
                        
                        net_profit = revenue - annual_opex
                        cashflows = [-capex] + [net_profit] * optimizer.project_life_years
                        irr = npf.irr(cashflows) if net_profit > 0 else 0
                        npv = npf.npv(optimizer.discount_rate, cashflows)
                        payback = capex / net_profit if net_profit > 0 else 99
                        
                        target_irr = target_irr_input / 100
                        pv_factor = (1 - (1 + target_irr)**-optimizer.project_life_years) / target_irr
                        req_annual_profit = capex / pv_factor if pv_factor > 0 else 0
                        min_ppa_mwh = ((req_annual_profit + annual_opex - (revenue - btm_kwh*(p_galetech_input/1000))) / btm_kwh * 1000) if btm_kwh > 0 else 0
                        
                        config_name = f"{t_count}x {t_model}" if t_count > 0 else "No Wind"
                        res_list.append({
                            'Turbine': config_name, 'Solar_MW': s, 'BESS_MWh': b,
                            'CAPEX_M': capex/1e6, 'OPEX_k': annual_opex/1e3, 'Profit_k': net_profit/1e3, 
                            'Payback': payback, 'IRR': irr*100, 'NPV10_M': npv/1e6,
                            'CO2_T': co2, 'BTM_Supply_MWh': btm_kwh/1000, 
                            'Curtailed_MWh': curtail_kwh/1000, 'Curtail_Cost_k': curtail_cost/1000, # include curtailment tracking
                            'Min_PPA_Price': min_ppa_mwh,
                            't_model_raw': t_model, 't_count_raw': t_count
                        })

    df_res = pd.DataFrame(res_list)
    df_viable = df_res[df_res['Payback'] < 15].sort_values('NPV10_M', ascending=False).reset_index(drop=True)

    if df_viable.empty:
        st.error("No commercially viable configurations found. Try adjusting limits or parameters.")
        st.stop()

    best = df_viable.iloc[0]
    st.session_state['best_config'] = best # cache best configuration for Monte Carlo analysis
    st.session_state['rep_days'] = rep_days
    
    df_traces = optimizer.evaluate_combination(best['t_model_raw'], best['t_count_raw'], best['Solar_MW'], best['BESS_MWh'], rep_days, return_traces=True)
    csv_buffer = io.StringIO()
    df_traces.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')

    st.success("✅ Optimization Complete. Bankable Report Generated.")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. Executive Summary", "2. Cost Breakdown", "3. Benchmarking", "4. Monte Carlo Risk", "5. Auditable Pack"])
    
    with tab1:
        st.header("Executive Summary")
        st.markdown(f"### 🎯 Headline Technical Recommendation")
        st.markdown(f"- **Wind Asset:** {best['Turbine']} | **Solar PV:** {best['Solar_MW']} MW | **BESS:** {best['BESS_MWh']} MWh")
        col1, col2, col3 = st.columns(3)
        col1.metric("NPV @ 10%", f"€ {best['NPV10_M']:.2f} M")
        col2.metric("Project IRR", f"{best['IRR']:.1f} %")
        col3.metric("Annual Curtailed Energy", f"{int(best['Curtailed_MWh'])} MWh", "Lost Opportunity")

        st.info(f"**Suggested Minimum PPA Price to achieve {target_irr_input}% IRR:** € {best['Min_PPA_Price']:.2f} / MWh")
        
    with tab2:
        st.header("Financial Performance & Breakdown")
        show_cols = ['Turbine', 'Solar_MW', 'BESS_MWh', 'CAPEX_M', 'NPV10_M', 'IRR', 'Curtailed_MWh', 'Min_PPA_Price']
        st.dataframe(df_viable[show_cols].head(10).style.format({'CAPEX_M': '{:.2f}', 'NPV10_M': '{:.2f}', 'IRR': '{:.1f}%', 'Curtailed_MWh': '{:.0f}', 'Min_PPA_Price': '€ {:.2f}'}))

    with tab3:
        st.header("Technology Benchmarking")
        st.markdown("As per project requirements, comparing the recommended optimum alongside benchmark cases to make incremental value transparent.")
        
        benchmarks = []
        # 1. Optimal
        benchmarks.append(best.to_dict())
        benchmarks[-1]['Category'] = "⭐ Recommended Optimum"
        
        # 2. Renewables without Battery
        no_bess = df_res[(df_res['BESS_MWh'] == 0) & ((df_res['Solar_MW'] > 0) | (df_res['Turbine'] != "No Wind"))].sort_values('NPV10_M', ascending=False)
        if not no_bess.empty:
            b = no_bess.iloc[0].to_dict()
            b['Category'] = "❌ Renewables Only (No Battery)"
            benchmarks.append(b)
            
        # 3. Battery without Renewables
        bess_only = df_res[(df_res['BESS_MWh'] > 0) & (df_res['Solar_MW'] == 0) & (df_res['Turbine'] == "No Wind")].sort_values('NPV10_M', ascending=False)
        if not bess_only.empty:
            b = bess_only.iloc[0].to_dict()
            b['Category'] = "❌ Battery Only (No Renewables)"
            benchmarks.append(b)

        df_bench = pd.DataFrame(benchmarks)
        st.dataframe(df_bench[['Category', 'Turbine', 'Solar_MW', 'BESS_MWh', 'CAPEX_M', 'IRR', 'NPV10_M', 'Curtailed_MWh']].style.format({'CAPEX_M': '{:.2f}', 'NPV10_M': '{:.2f}', 'IRR': '{:.1f}%', 'Curtailed_MWh': '{:.0f}'}))
        st.caption("Notice how adding a battery may reduce IRR slightly due to CAPEX, but is critical in reducing curtailment and maximizing NPV.")

    with tab4:
        st.header("Monte Carlo Uncertainty Analysis")
        st.markdown("Simulate risk across wind resource, capital costs, and PPA price assumptions to identify robust outcomes.")
        
        if st.button("🎲 Run 50 Monte Carlo Simulations on Optimal Config"):
            mc_results = []
            progress = st.progress(0)
            best_conf = st.session_state['best_config']
            
            for i in range(50):
                # sample uncertainty shock factors
                wind_shock = np.random.normal(1.0, 0.10) # wind resource variability ±10%
                capex_shock = np.random.normal(1.0, 0.05) # CAPEX over/under run ±5%
                ppa_shock = np.random.normal(1.0, 0.05)   # customer PPA price volatility ±5%
                
                # recompute with shocks
                capex, _, _, _, _ = optimizer.get_capex(best_conf['t_model_raw'], best_conf['t_count_raw'], best_conf['Solar_MW'], best_conf['BESS_MWh'], capex_shock)
                annual_opex, _, _, _ = optimizer.get_opex(capex)
                
                # -------------------------------------------------------------
                # Fix: Monte Carlo loop also unpacks 5 return values from evaluator
                # -------------------------------------------------------------
                rev, co2, btm, curt, curt_cost = optimizer.evaluate_combination(best_conf['t_model_raw'], best_conf['t_count_raw'], best_conf['Solar_MW'], best_conf['BESS_MWh'], st.session_state['rep_days'], wind_shock=wind_shock, ppa_shock=ppa_shock)
                
                profit = rev - annual_opex
                cashflows = [-capex] + [profit] * 20
                irr = npf.irr(cashflows) if profit > 0 else 0
                npv = npf.npv(0.10, cashflows)
                
                mc_results.append({'Iteration': i, 'NPV10_M': npv/1e6, 'IRR': irr*100, 'Wind_Shock': wind_shock})
                progress.progress((i+1)/50)
                
            progress.empty()
            df_mc = pd.DataFrame(mc_results)
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("**NPV Distribution (P90 / P50 / P10)**")
                p90, p50, p10 = np.percentile(df_mc['NPV10_M'], [10, 50, 90]) # P90 means 90% chance to exceed this value (conservative)
                st.write(f"- **Conservative (P90):** € {p90:.2f} M")
                st.write(f"- **Expected (P50):** € {p50:.2f} M")
                st.write(f"- **Optimistic (P10):** € {p10:.2f} M")
            with c2:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.hist(df_mc['NPV10_M'], bins=10, color='skyblue', edgecolor='black')
                ax.axvline(p50, color='red', linestyle='dashed', linewidth=1, label='P50 Expected')
                ax.set_title('Monte Carlo NPV Outcomes')
                ax.legend()
                st.pyplot(fig)

    with tab5:
        st.header("Auditable Calculation Pack")
        st.markdown("Download the fully transparent, hourly dispatch model. Tracing energy flows from weather inputs to BTM supply, grid export, curtailment, and battery states.")
        st.download_button("📥 Download Hourly Dispatch Pack (CSV)", data=csv_data, file_name=f"Galetech_Audit_Pack_{best['Turbine']}_{best['Solar_MW']}MW_{best['BESS_MWh']}MWh.csv", mime="text/csv")