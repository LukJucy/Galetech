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
        # t_model: selected wind turbine model key
        # t_count: number of turbines
        # s_mw: solar capacity in MW
        # b_mwh: battery energy capacity in MWh
        # rep_days: representative days with weather and load profiles
        # return_traces: if True return hourly dispatch DataFrame
        # wind_shock: wind power scaling factor for sensitivity tests
        # ppa_shock: PPA price scaling factor for sensitivity tests

        annual_revenue = 0  # yearly revenue aggregated over scenarios
        annual_co2_saved = 0  # yearly CO2 savings in tonnes from gas offset
        annual_btm_elec_kwh = 0  # yearly behind-the-meter electricity supplied
        annual_btm_gas_kwh = 0  # yearly gas load shifted/avoided
        annual_curtailed_kwh = 0  # yearly energy curtailed
        annual_curtail_cost = 0  # yearly lost revenue from curtailment
        
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
            p_curtail = cp.Variable(T, nonneg=True) 
            
            p_wind = np.zeros(T)
            if t_model in self.wind_solar_ref_data and t_model != "None":
                p_wind = self.wind_solar_ref_data[t_model] * t_count  # kW output * turbine count
            
            p_solar = np.zeros(T)
            if 'solar_per_mw' in self.wind_solar_ref_data and s_mw > 0:
                p_solar = self.wind_solar_ref_data['solar_per_mw'] * s_mw * self.params['solar_efficiency']  # kW per MW * capacity * efficiency
            
            constraints = [
                p_wind + p_solar + p_dis == p_btm_elec + p_gas_replaced + p_grid_sell + p_ch + p_curtail,
                p_btm_elec <= day['elec_load'],
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

            # Revenue components: customer electricity sale + gas displacement equivalent + grid export
            rev_elec = cp.sum(p_btm_elec * (self.params['p_galetech'] * ppa_shock))
            rev_gas = cp.sum(p_gas_replaced * (gas_avoided_cost * ppa_shock))
            rev_grid = cp.sum(p_grid_sell * self.params['p_sell'])
            
            obj = cp.Maximize(rev_elec + rev_gas + rev_grid - cp.sum(p_dis * bess_wear_cost))
            prob = cp.Problem(obj, constraints)
            
            try:
                prob.solve(solver=cp.OSQP)
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
    加载风光发电参考数据（从模型内置）。
    基于 Typical_Daily_Power_Profile_2019.xlsx 中的实际数据。
    """
    # 实际数据应该从Excel中读取，这里用示例数据
    # 在实际应用中，可以让管理员上传此文件给模型配置
    ref_data = {
        "EWT 500kW": np.array([0]*24) * 0.5 / 1.0,  # 每台kW输出
        "EWT 1MW": np.array([0]*24),
        "E82 2.3MW": np.array([0]*24) * 2.3,
        "V90 3MW": np.array([1022.7, 1033.8, 1031.4, 1046.0, 1043.8, 1031.1, 1028.1, 1030.9, 1011.6, 976.9, 985.9, 973.6, 975.1, 962.9, 961.9, 963.9, 946.2, 954.3, 918.2, 935.9, 939.9, 965.5, 975.9, 1007.9]),  # 根据提供的数据
        "E115 4.2MW": np.array([0]*24) * 4.2,
        "E138 4.26MW": np.array([0]*24) * 4.26,
        "solar_per_mw": np.array([0, 0, 0, 0, 19.9, 96.0, 239.7, 448.4, 706.3, 979.5, 1203.2, 1333.4, 1348.4, 1245.7, 1042.3, 775.5, 509.0, 286.2, 125.5, 32.9, 0, 0, 0, 0])  # 根据提供的数据
    }
    return ref_data

# ==========================================
# 3. Streamlit UI and report generation
# ==========================================
st.set_page_config(page_title="Galetech BOO Bankable Report", layout="wide")
st.title("📑 Galetech BOO Optimiser & Bankability Assistant")

with st.sidebar:
    st.header("📂 Data & Profile Setup")
    uploaded_file = st.file_uploader("Upload Customer Hourly Load Profile (CSV/Excel: elec_load, gas_load per hour, multiples of 24 rows)", type=['csv', 'xlsx'])
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
    export_limit_input = st.number_input("Grid Export Limit (kW)", value=5000)

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
    max_turbines = st.slider("Max Turbines", 0, 10, 1)
    max_solar = st.slider("Max Solar PV (MW)", 0, 40, 10, step=2)
    max_bess = st.slider("Max BESS (MWh)", 0, 40, 10, step=1)
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
    res_list = []
    
    with st.spinner("Executing Mathematical Optimization (Co-optimising sizing, export & curtailment)..."):
        for t_model in turbine_choices:
            counts = [0] if t_model == "None" else range(1, max_turbines + 1)
            for t_count in counts:
                for s in range(0, max_solar + 1, max(1, max_solar//5)): 
                    for b in range(0, max_bess + 1, max(1, max_bess//5)): 
                        if t_count == 0 and s == 0 and b == 0: continue 
                        if require_wind and t_count == 0: continue
                        if require_bess and b == 0: continue
                        
                        capex, eq, civ, elec, pm = optimizer.get_capex(t_model, t_count, s, b)
                        annual_opex, ins, maint, lease = optimizer.get_opex(capex)
                        
                        revenue, co2, btm_e, btm_g, curt, curt_cost = optimizer.evaluate_combination(t_model, t_count, s, b, rep_days)
                        
                        net_profit = revenue - annual_opex
                        cashflows = [-capex] + [net_profit] * optimizer.project_life_years
                        irr = npf.irr(cashflows) if net_profit > 0 else 0
                        npv = npf.npv(optimizer.discount_rate, cashflows)
                        payback = capex / net_profit if net_profit > 0 else 99
                        
                        target_irr = target_irr_input / 100
                        pv_factor = (1 - (1 + target_irr)**-optimizer.project_life_years) / target_irr
                        req_annual_profit = capex / pv_factor if pv_factor > 0 else 0
                        
                        req_total_revenue = req_annual_profit + annual_opex
                        gas_avoided_cost = (p_gas_input + (0.202 * p_carbon_input)) / 1000
                        rev_fixed_other = revenue - (btm_e * (p_galetech_input/1000))
                        req_elec_rev = req_total_revenue - rev_fixed_other
                        min_ppa_elec = (req_elec_rev / btm_e * 1000) if btm_e > 0 else 0
                        
                        config_name = f"{t_count}x {t_model}" if t_count > 0 else "No Wind"
                        res_list.append({
                            'Turbine': config_name, 'Solar_MW': s, 'BESS_MWh': b,
                            'CAPEX_M': capex/1e6, 'OPEX_k': annual_opex/1e3, 'Profit_k': net_profit/1e3, 
                            'Payback': payback, 'IRR': irr*100, 'NPV10_M': npv/1e6,
                            'CO2_T': co2, 'Elec_Offset_MWh': btm_e/1000, 'Gas_Offset_MWh': btm_g/1000, 
                            'Curtailed_MWh': curt/1000, 'Min_Elec_PPA': min_ppa_elec,
                            't_model_raw': t_model, 't_count_raw': t_count
                        })

    df_res = pd.DataFrame(res_list)
    
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