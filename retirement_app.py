import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------- Page config ----------------------
st.set_page_config(page_title="Retirement Strategy Planner", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Voyage of Creativity" styling
st.markdown("""
<style>
    .big-font { font-size: 20px !important; }
    .metric-box {
        background-color: #f0f2f6;
        border-left: 5px solid #4e8cff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        color: #333333 !important;
    }
    .metric-box h3, .metric-box p, .metric-box span {
        color: #333333 !important;
    }
    .stHeader { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.title("Retirement Strategy Planner")
st.markdown("### Visualize your cash flow, compare Social Security strategies, and stress-test your portfolio.")

# ---------------------- Sidebar Inputs ----------------------
with st.sidebar:
    st.header("1. Personal Details")
    current_age = st.number_input("Your Age", 40, 85, 58, 1)
    spouse_age = st.number_input("Spouse Age", 30, 85, 49, 1)
    retire_age = st.number_input("Retirement Age", 50, 75, 61, 1)
    horizon_age = st.number_input("Projection End Age", 75, 105, 90, 1)
    infl = st.slider("Inflation / COLA (%)", 0.0, 6.0, 2.0, 0.1) / 100.0

    st.header("2. Monthly Expenses (Today's $)")
    with st.expander("Housing (Mortgage, Tax, Rent)"):
        mortgage_payment = st.number_input("Mortgage Payment ($)", 0.0, 10000.0, 557.35, 10.0)
        mortgage_years_left = st.number_input("Mortgage Years Left", 0, 40, 14, 1)
        property_tax = st.number_input("Property Tax ($/mo)", 0.0, 5000.0, 500.0, 10.0)
        house_maintenance = st.number_input("Maintenance ($/mo)", 0.0, 5000.0, 250.0, 10.0)
        
        st.caption("International Living / Snowbirding")
        months_abroad = st.slider("Months Abroad per Year", 0, 12, 0, 1)
        thai_ratio = st.slider("Abroad Cost Ratio vs US", 0.2, 1.0, 0.45, 0.05) if months_abroad > 0 else 1.0
        rent_monthly = st.number_input("Rent Cost Abroad ($/mo)", 0.0, 10000.0, 2000.0, 50.0) if months_abroad > 0 else 0.0
        rental_income_home = st.number_input("Rental Income from US Home ($/mo while away)", 0.0, 10000.0, 0.0, 100.0) if months_abroad > 0 else 0.0
        rent_grows_with_cola = st.checkbox("Rent grows w/ COLA", value=True) if months_abroad > 0 else False

    with st.expander("Living Essentials (Food, Util, Trans)"):
        food = st.number_input("Food ($)", 0.0, 5000.0, 1000.0, 10.0)
        utilities = st.number_input("Utilities ($)", 0.0, 5000.0, 280.0, 10.0, help="Includes heat, electric, water, etc.")
        transportation = st.number_input("Transportation ($)", 0.0, 5000.0, 450.0, 10.0)
        insurance_other = st.number_input("Other Insurance ($)", 0.0, 5000.0, 250.0, 10.0)

    with st.expander("Healthcare (Premiums + OOP)"):
        hdhp_annual = st.number_input("HDHP Premium ($/yr/person, pre-65)", 0.0, 30000.0, 6000.0, 100.0)
        med_oop_total = st.number_input("Lifetime Medicare OOP ($/person, 65+)", 0.0, 500000.0, 160000.0, 1000.0)
        med_oop_real_escalation = st.slider("Med Costs Real Growth (%)", 0.0, 10.0, 3.0, 0.1) / 100.0

    with st.expander("Discretionary & Goals"):
        discretionary = st.number_input("Discretionary & Buffer ($/mo)", 0.0, 50000.0, 1000.0, 100.0, help="Combined monthly budget for non-essentials, travel, hobbies, and extra goals.")
        travel = st.number_input("Travel ($)", 0.0, 10000.0, 375.0, 10.0)

    st.header("3. Income & Assets")
    with st.expander("Social Security"):
        you_ss_62 = st.number_input("Your SS @ 62 ($/mo)", 0.0, 10000.0, 2577.0, 10.0)
        you_ss_67 = st.number_input("Your SS @ 67 ($/mo)", 0.0, 10000.0, 3681.0, 10.0)
        you_ss_70 = st.number_input("Your SS @ 70 ($/mo)", 0.0, 15000.0, float(3681.0 * 1.24), 10.0)
        spouse_spousal_benefit = st.number_input("Spousal Benefit ($/mo)", 0.0, 10000.0, float(3681 * 0.5), 10.0)
        spouse_claim_age_spouse = st.number_input("Spouse Claim Age", 62, 70, 67, 1)
        primary_ss_plan = st.selectbox("Scenario Plan", ["Claim at 62", "Claim at 67", "Claim at 70"], index=1)

    with st.expander("Portfolio Assumptions"):
        start_principal = st.number_input("Current Portfolio ($)", 0.0, 20000000.0, 800000.0, 1000.0)
        withdraw_pct = st.slider("Safe Withdrawal Rate (%)", 0.0, 10.0, 4.0, 0.1, help="Annual % of initial portfolio value to withdraw.") / 100.0
        
    with st.expander("Taxes"):
        withdraw_tax_rate = st.slider("Portfolio Tax Rate (%)", 0.0, 50.0, 15.0, 0.5) / 100.0
        ss_taxable_pct = st.slider("SS Taxable Portion (%)", 0.0, 100.0, 85.0, 1.0) / 100.0
        ss_marginal_rate = st.slider("SS Marginal Tax (%)", 0.0, 50.0, 12.0, 0.5) / 100.0

    st.header("4. Simulation Parameters")
    stock_alloc = st.slider("Asset Allocation (% Stocks)", 0, 100, 75, 5, help="Remainder is assumed to be US 10Y Bonds.") / 100.0
    sims = st.slider("Simulations", 100, 5000, 1000, 100)
    mc_seed_input = st.number_input("Random Seed", -1, 999999, 42, 1)

# ---------------------- Logic: Calculation ----------------------
@st.cache_data
def load_historical_data():
    try:
        # Columns: Date,SP500,Dividend,Earnings,Consumer Price Index,Long Interest Rate,Real Price,Real Dividend,Real Earnings,PE10
        df = pd.read_csv("market_data.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Calculate Monthly Returns
        # Stock: (P_t + D_t/12) / P_{t-1} - 1
        df["Stock_Price"] = df["SP500"]
        df["Stock_Div"] = df["Dividend"]
        
        # Bond: Yield Income + Price Change (Duration approx 7)
        # Yield is in %. Monthly Yield = Yield/1200.
        # Price Change approx -Duration * Delta_Yield.
        df["Bond_Yield"] = df["Long Interest Rate"]
        duration = 7.0
        
        # Inflation: CPI_t / CPI_{t-1} - 1
        df["CPI"] = df["Consumer Price Index"]
        
        # Vectorized calc
        prev_price = df["Stock_Price"].shift(1)
        prev_yield = df["Bond_Yield"].shift(1)
        prev_cpi = df["CPI"].shift(1)
        
        # Stock Nominal Return
        # Shiller Div is annualized.
        df["Stock_Ret_Nom"] = (df["Stock_Price"] + df["Stock_Div"]/12.0) / prev_price - 1.0
        
        # Bond Nominal Return
        # Yield change in decimal = (y_t - y_{t-1})/100
        yield_change = (df["Bond_Yield"] - prev_yield) / 100.0
        income = prev_yield / 1200.0
        df["Bond_Ret_Nom"] = income - (duration * yield_change)
        
        # Inflation
        df["Infl_Rate"] = df["CPI"] / prev_cpi - 1.0
        
        # Real Returns
        df["Stock_Ret_Real"] = (1 + df["Stock_Ret_Nom"]) / (1 + df["Infl_Rate"]) - 1.0
        df["Bond_Ret_Real"] = (1 + df["Bond_Ret_Nom"]) / (1 + df["Infl_Rate"]) - 1.0
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading market data: {e}")
        return None

hist_data = load_historical_data()

ages = np.arange(retire_age, horizon_age + 1)

def escalating_series_initial(total, r, years):
    if r == 0: return total / years
    return total * r / ((1 + r) ** years - 1)

def build_expenses_monthly():
    # Base columns
    cols = [
        "Mortgage", "Property Tax", "Rent Abroad", "House Maint", "Rental Income Credit", # Housing
        "Food", "Utilities", "Transportation", "Insurance Other", # Essentials
        "Discretionary", "Travel", "Buffer", # Discretionary
        "HDHP (You)", "HDHP (Spouse)", "Medicare OOP (You)", "Medicare OOP (Spouse)" # Health
    ]
    df = pd.DataFrame(0.0, index=ages, columns=cols)
    
    abroad_frac = months_abroad / 12.0
    # Weighted average cost factor for variable expenses
    # (Months Home * 1.0 + Months Abroad * Ratio) / 12
    blend_factor = (1.0 - abroad_frac) + (abroad_frac * thai_ratio)
    
    # Calculate costs per year
    for a in ages:
        yrs = a - retire_age
        f = (1 + infl) ** yrs
        
        # Housing
        df.loc[a, "Mortgage"] = (mortgage_payment if a < current_age + mortgage_years_left else 0.0)
        df.loc[a, "Property Tax"] = property_tax * f
        df.loc[a, "House Maint"] = house_maintenance * f
        
        rent_growth = f if rent_grows_with_cola else 1.0
        # Rent expense (only incurred for months abroad)
        df.loc[a, "Rent Abroad"] = rent_monthly * abroad_frac * rent_growth
        
        # Rental Income (only earned for months abroad) - grows with inflation
        # We store as positive value here, subtract later
        df.loc[a, "Rental Income Credit"] = rental_income_home * abroad_frac * f
        
        # Essentials (Blended)
        df.loc[a, "Food"] = food * blend_factor * f
        df.loc[a, "Utilities"] = utilities * blend_factor * f
        df.loc[a, "Transportation"] = transportation * blend_factor * f
        df.loc[a, "Insurance Other"] = insurance_other * blend_factor * f
        
        # Discretionary (Blended)
        df.loc[a, "Discretionary"] = discretionary * blend_factor * f
        
        # Travel (Not blended, assumed fully applicable)
        df.loc[a, "Travel"] = travel * f
        
        # Healthcare (HDHP)
        df.loc[a, "HDHP (You)"] = (hdhp_annual / 12 * f) if a < 65 else 0.0
        spouse_is_under_65 = (spouse_age + (a - current_age)) < 65
        df.loc[a, "HDHP (Spouse)"] = (hdhp_annual / 12 * f) if spouse_is_under_65 else 0.0

    # Healthcare (Medicare OOP)
    N = 90 - 65 + 1
    A = escalating_series_initial(med_oop_total, med_oop_real_escalation, N)
    spouse_start_age = current_age + (65 - spouse_age)
    
    you_real = pd.Series(0.0, index=ages)
    sp_real = pd.Series(0.0, index=ages)
    for a in ages:
        if a >= 65:
            k = a - 65
            you_real[a] = A * ((1 + med_oop_real_escalation) ** k)
        if a >= spouse_start_age:
            k = a - spouse_start_age
            sp_real[a] = A * ((1 + med_oop_real_escalation) ** k)

    for a in ages:
        f = (1 + infl) ** (a - retire_age)
        df.loc[a, "Medicare OOP (You)"] = (you_real[a] * f) / 12
        df.loc[a, "Medicare OOP (Spouse)"] = (sp_real[a] * f) / 12

    return df

exp_m = build_expenses_monthly()
inflation_factors = (1 + infl) ** (ages - retire_age)

    # ---------------------- SS Income ----------------------
def ss_series(claim_age, base_val, sp_claim_age, sp_benefit):
    out = pd.Series(0.0, index=ages)
    # You
    for a in ages:
        if a >= claim_age:
            out[a] += base_val * ((1 + infl) ** (a - claim_age))
    # Spouse
    sp_claim_you_age = current_age + (sp_claim_age - spouse_age)
    for a in ages:
        if a >= sp_claim_you_age:
            out[a] += sp_benefit * ((1 + infl) ** (a - sp_claim_you_age))
    return out

def get_net_ss(gross_monthly):
    tax = gross_monthly * ss_taxable_pct * ss_marginal_rate
    return gross_monthly - tax, tax

ss62_gross = ss_series(62, you_ss_62, spouse_claim_age_spouse, spouse_spousal_benefit)
ss67_gross = ss_series(67, you_ss_67, spouse_claim_age_spouse, spouse_spousal_benefit)
ss70_gross = ss_series(70, you_ss_70, spouse_claim_age_spouse, spouse_spousal_benefit)

ss62_net, ss62_tax = get_net_ss(ss62_gross)
ss67_net, ss67_tax = get_net_ss(ss67_gross)
ss70_net, ss70_tax = get_net_ss(ss70_gross)

# Active Scenario
if primary_ss_plan == "Claim at 62":
    active_ss_net, active_ss_tax = ss62_net, ss62_tax
elif primary_ss_plan == "Claim at 67":
    active_ss_net, active_ss_tax = ss67_net, ss67_tax
else:
    active_ss_net, active_ss_tax = ss70_net, ss70_tax

# Taxes Calculation
# Spending includes the "Desired Extra" so we are calculating tax needed to cover ALL desired spending
# NOTE: We must subtract Rental Income Credit, not add it.
expenses_only = exp_m.drop(columns=["Rental Income Credit"])
total_spend_pre_tax = expenses_only.sum(axis=1) - exp_m["Rental Income Credit"]

# Monthly tax needs
shortfall_monthly = np.maximum(0.0, total_spend_pre_tax + active_ss_tax - active_ss_net)

if withdraw_tax_rate >= 1.0:
    gross_withdraw = np.zeros_like(shortfall_monthly)
else:
    gross_withdraw = np.where(shortfall_monthly > 0, shortfall_monthly / (1 - withdraw_tax_rate), 0.0)

withdraw_tax = gross_withdraw * withdraw_tax_rate
exp_m["Estimated Taxes"] = active_ss_tax + withdraw_tax

# ---------------------- Grouping for Display ----------------------
# Create a cleaner dataframe for plotting
exp_grouped = pd.DataFrame(index=ages)
# Housing = Mortgage + Tax + Rent Abroad + Maint - Rental Income
exp_grouped["Housing"] = (exp_m["Mortgage"] + exp_m["Property Tax"] + 
                          exp_m["Rent Abroad"] + exp_m["House Maint"] - 
                          exp_m["Rental Income Credit"])
# Ensure housing expense doesn't go below zero if income is high? 
# Usually we want net cash flow. If negative, it effectively subsidizes other costs.
# But for a stacked bar chart of "Where money goes", a negative bar is weird.
# Let's floor it at zero for the visual bar, OR allow it to reduce the stack height naturally if the plotting library supports it.
# Matplotlib bar bottoms are simpler if positive.
# Let's keep it simple: Net Housing Cost.
exp_grouped["Housing"] = exp_grouped["Housing"].apply(lambda x: max(0.0, x))

exp_grouped["Healthcare"] = exp_m[["HDHP (You)", "HDHP (Spouse)", "Medicare OOP (You)", "Medicare OOP (Spouse)"]].sum(axis=1)
# Breakdown of Essentials
exp_grouped["Food"] = exp_m["Food"]
exp_grouped["Utilities"] = exp_m["Utilities"]
exp_grouped["Transportation"] = exp_m["Transportation"]
exp_grouped["Insurance"] = exp_m["Insurance Other"]

exp_grouped["Flexible Spending"] = exp_m[["Discretionary", "Travel"]].sum(axis=1)
exp_grouped["Taxes"] = exp_m["Estimated Taxes"]
# ---------------------- VISUALIZATION ----------------------

# Global Scenario Selector
st.markdown("---")
st.header("Analysis Dashboard")
analysis_scenario = st.radio(
    "Select Social Security Strategy to Analyze:",
    ["Claim at 62", "Claim at 67", "Claim at 70"],
    horizontal=True,
    index=1
)

# Determine Active Data based on Selection
if analysis_scenario == "Claim at 62":
    current_ss_net = ss62_net
    current_ss_tax = ss62_tax
    current_color = "#1976d2"
elif analysis_scenario == "Claim at 67":
    current_ss_net = ss67_net
    current_ss_tax = ss67_tax
    current_color = "#0d47a1"
else:
    current_ss_net = ss70_net
    current_ss_tax = ss70_tax
    current_color = "#002171"

# 1. Monthly Expenses
st.subheader("1. Monthly Cash Flow Breakdown")
st.markdown("Your projected monthly expenses vs. spending needs.")

fig1, ax1 = plt.subplots(figsize=(12, 6))
groups = ["Housing", "Healthcare", "Food", "Utilities", "Transportation", "Insurance", "Taxes", "Flexible Spending"]
colors = ["#3f51b5", "#f44336", "#2e7d32", "#4caf50", "#8bc34a", "#009688", "#607d8b", "#ff9800"]
bottom = np.zeros(len(ages))

for col, color in zip(groups, colors):
    vals = exp_grouped[col].values
    ax1.bar(ages, vals, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
    bottom += vals

ax1.set_xlabel("Age")
ax1.set_ylabel("Monthly ($)")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax1.grid(axis="y", linestyle=":", alpha=0.3)
st.pyplot(fig1)


# 2. Annual Comparison
st.subheader("2. Annual Income vs. Spending")
st.markdown(f"**Scenario: {analysis_scenario}**")

# Helper to recalc taxes and spending for the selected scenario
def get_scenario_annuals(ss_net_m, ss_tax_m):
    # Recalculate taxes based on this specific SS stream
    local_shortfall = np.maximum(0.0, total_spend_pre_tax + ss_tax_m - ss_net_m)
    
    if withdraw_tax_rate >= 1.0:
        local_gross_wd = np.zeros_like(local_shortfall)
    else:
        local_gross_wd = np.where(local_shortfall > 0, local_shortfall / (1 - withdraw_tax_rate), 0.0)
        
    local_wd_tax = local_gross_wd * withdraw_tax_rate
    local_total_tax = ss_tax_m + local_wd_tax
    
    # Total Spending Need = Pre-Tax Spend + Total Taxes
    local_spend_need = (total_spend_pre_tax + local_total_tax) * 12.0
    return local_spend_need

# Calculate specifics for current selection
spend_annual_current = get_scenario_annuals(current_ss_net, current_ss_tax)
safe_draw_annual = (start_principal * withdraw_pct) * inflation_factors
ss_inc_y = current_ss_net * 12.0

fig2, ax2 = plt.subplots(figsize=(14, 6))
x = np.arange(len(ages))

# Income Stack
ax2.bar(x, ss_inc_y, label="SS Income (Net)", color=current_color, alpha=0.9, width=0.6)
ax2.bar(x, safe_draw_annual, bottom=ss_inc_y, label=f"Portfolio Draw ({withdraw_pct:.1%})", color="#4caf50", alpha=0.9, width=0.6)

# Spending Line
ax2.plot(x, spend_annual_current, color="#d32f2f", linewidth=3, marker="o", markersize=4, label="Total Spending Need (w/ Tax)")

# Shortfall Fill
total_inc = ss_inc_y + safe_draw_annual
ax2.fill_between(x, total_inc, spend_annual_current, where=(spend_annual_current > total_inc), 
                interpolate=True, color="red", alpha=0.15, hatch="///", label="Shortfall")

ax2.set_title(f"Cash Flow Projection: {analysis_scenario}", fontsize=14, fontweight="bold")
ax2.set_xlabel("Age")
ax2.set_ylabel("Annual Amount ($)")
ax2.set_xticks(x[::2])
ax2.set_xticklabels(ages[::2])
ax2.grid(axis="y", linestyle=":", alpha=0.4)
ax2.legend(loc="upper left")
st.pyplot(fig2)


# 3. Monte Carlo
st.subheader("3. Portfolio Reliability (Monte Carlo)")
st.markdown(f"Stress-testing: **{analysis_scenario}** using historical market data (1871–Present).")

# Run Full Simulation Logic
def run_full_mc(principal, req_annual, years, sims, stock_pct, seed):
    rng = np.random.default_rng(int(seed) if seed >= 0 else None)
    total_months = years * 12
    
    # 1. Construct Historical Weighted Real Returns
    if hist_data is None:
        return np.zeros((sims, total_months+1)), np.zeros(sims)
        
    # Mix
    hist_real = hist_data["Stock_Ret_Real"].values * stock_pct + \
                hist_data["Bond_Ret_Real"].values * (1 - stock_pct)
    
    n_hist = len(hist_real)
    block_size = 12 # 1 year blocks
    
    # 2. Block Bootstrap
    n_blocks = int(np.ceil(total_months / block_size))
    starts = rng.integers(0, n_hist - block_size + 1, size=(sims, n_blocks))
    
    sim_returns = np.zeros((sims, n_blocks * block_size))
    for i in range(sims):
        path = []
        for start_idx in starts[i]:
            path.append(hist_real[start_idx : start_idx + block_size])
        sim_returns[i, :] = np.concatenate(path)
        
    sim_returns = sim_returns[:, :total_months]
    
    # 3. Re-inflate
    monthly_infl = (1 + infl) ** (1/12) - 1
    sim_nominal = (1 + sim_returns) * (1 + monthly_infl) - 1
    
    # Monthly Req
    req_monthly = np.repeat(req_annual / 12.0, 12)[:total_months]
    
    balances = np.zeros((sims, total_months + 1))
    balances[:, 0] = principal
    
    for m in range(total_months):
        balances[:, m+1] = (balances[:, m] * (1 + sim_nominal[:, m])) - req_monthly[m]
    
    ruined = np.any(balances <= 0, axis=1)
    return balances, ruined

# Helper for calculation
def get_gross_req(ss_net, ss_tax):
    shortfall = np.maximum(0.0, total_spend_pre_tax + ss_tax - ss_net)
    if withdraw_tax_rate >= 1.0:
        gross = np.zeros_like(shortfall)
    else:
        gross = np.where(shortfall > 0, shortfall / (1 - withdraw_tax_rate), 0.0)
    return gross * 12.0

# Run MC for Selected Scenario
gross_req = get_gross_req(current_ss_net, current_ss_tax)
mc_bals, mc_ruined = run_full_mc(start_principal, gross_req, len(ages), sims, stock_alloc, mc_seed_input)

success_rate = 1.0 - mc_ruined.mean()
median_end = np.median(mc_bals[:, -1])
median_end_str = f"${median_end:,.0f}" if median_end > 0 else "$0 (Depleted)"

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Success Rate</h3>
        <span class="big-font" style="color: {'green' if success_rate > 0.8 else 'red'}">
            {success_rate:.1%}
        </span>
        <p>Likelihood portfolio lasts to age {horizon_age}.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <h3>Median Ending Balance</h3>
        <span class="big-font">
            {median_end_str}
        </span>
        <p>Expected nominal wealth at age {horizon_age}.</p>
    </div>
    """, unsafe_allow_html=True)

if mc_bals is not None:
    months_total = mc_bals.shape[1]
    x_months = np.arange(months_total) / 12.0 + retire_age
    
    p50 = np.percentile(mc_bals, 50, axis=0)
    p10 = np.percentile(mc_bals, 10, axis=0)
    p90 = np.percentile(mc_bals, 90, axis=0)

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    # Plot sample paths
    for i in range(min(50, sims)):
        ax3.plot(x_months, mc_bals[i, :], color="#aeb6bf", alpha=0.10)

    # Plot percentiles
    ax3.plot(x_months, p50, color=current_color, linewidth=2.5, label="Median Path")
    ax3.fill_between(x_months, p10, p90, color=current_color, alpha=0.15, label="10th-90th Percentile")

    ax3.axhline(0, color="#b71c1c", linestyle="--", linewidth=1.8)
    ax3.set_ylim(bottom= -start_principal * 0.2, top=np.percentile(mc_bals, 95))
    ax3.ticklabel_format(style='plain', axis='y', useOffset=False)
    ax3.set_ylabel("Portfolio Balance ($)")
    ax3.set_xlabel("Age")
    ax3.set_title(f"Simulation: {analysis_scenario} (Success: {success_rate:.1%})")
    ax3.legend(loc="upper left")
    st.pyplot(fig3, use_container_width=True)

st.caption(f"Simulation based on {sims} runs using **Historical Block Bootstrap** (sampling 1-year blocks from Shiller data 1871-Present).")
st.caption("This method captures historical volatility clustering and crashes (e.g., 1929, 2008) better than a simple bell curve.")