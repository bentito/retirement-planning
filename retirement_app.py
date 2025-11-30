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
        
        st.caption("Thailand / Rent Option")
        thailand_half = st.checkbox("Spend 6mo/yr in Thailand", value=False)
        thai_ratio = st.slider("Thailand Cost Ratio vs US", 0.2, 1.0, 0.45, 0.05) if thailand_half else 1.0
        rent_monthly = st.number_input("Local Rent if renting ($/mo)", 0.0, 10000.0, 2000.0, 50.0) if thailand_half else 0.0
        rent_grows_with_cola = st.checkbox("Rent grows w/ COLA", value=True) if thailand_half else False

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
        discretionary = st.number_input("General Discretionary ($)", 0.0, 10000.0, 450.0, 10.0)
        travel = st.number_input("Travel ($)", 0.0, 10000.0, 375.0, 10.0)
        buffer = st.number_input("Misc Buffer ($)", 0.0, 10000.0, 250.0, 10.0)
        desired_extra_spending_monthly = st.number_input("Desired Extra Spending Goal ($/mo)", 0.0, 50000.0, 750.0, 10.0) # Changed to monthly, default / 12

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

def blended(val, ratio):
    return val * (0.5 + 0.5 * ratio)

def escalating_series_initial(total, r, years):
    if r == 0: return total / years
    return total * r / ((1 + r) ** years - 1)

def build_expenses_monthly():
    # Base columns
    cols = [
        "Mortgage", "Property Tax", "Rent", "House Maint", # Housing
        "Food", "Utilities", "Transportation", "Insurance Other", # Essentials
        "Discretionary", "Travel", "Buffer", # Discretionary
        "HDHP (You)", "HDHP (Spouse)", "Medicare OOP (You)", "Medicare OOP (Spouse)" # Health
    ]
    df = pd.DataFrame(0.0, index=ages, columns=cols)
    
    # Calculate costs per year
    for a in ages:
        yrs = a - retire_age
        f = (1 + infl) ** yrs
        
        # Housing
        df.loc[a, "Mortgage"] = (mortgage_payment if a < current_age + mortgage_years_left else 0.0)
        df.loc[a, "Property Tax"] = property_tax * f
        df.loc[a, "House Maint"] = house_maintenance * f
        
        ratio = thai_ratio if thailand_half else 1.0
        rent_factor = 0.5 if thailand_half else 0.0
        rent_growth = f if rent_grows_with_cola else 1.0
        df.loc[a, "Rent"] = rent_monthly * rent_factor * rent_growth
        
        # Essentials
        df.loc[a, "Food"] = (blended(food, ratio) if thailand_half else food) * f
        df.loc[a, "Utilities"] = (blended(utilities, ratio) if thailand_half else utilities) * f
        df.loc[a, "Transportation"] = (blended(transportation, ratio) if thailand_half else transportation) * f
        df.loc[a, "Insurance Other"] = (blended(insurance_other, ratio) if thailand_half else insurance_other) * f
        
        # Discretionary
        df.loc[a, "Discretionary"] = (blended(discretionary, ratio) if thailand_half else discretionary) * f
        df.loc[a, "Travel"] = travel * f
        df.loc[a, "Buffer"] = buffer * f
        
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

# Add "Desired Extra Spending"
exp_m["Desired Extra Spending"] = desired_extra_spending_monthly * inflation_factors

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
total_spend_pre_tax = exp_m.sum(axis=1)
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
exp_grouped["Housing"] = exp_m[["Mortgage", "Property Tax", "Rent", "House Maint"]].sum(axis=1)
exp_grouped["Healthcare"] = exp_m[["HDHP (You)", "HDHP (Spouse)", "Medicare OOP (You)", "Medicare OOP (Spouse)"]].sum(axis=1)
# Breakdown of Essentials
exp_grouped["Food"] = exp_m["Food"]
exp_grouped["Utilities"] = exp_m["Utilities"]
exp_grouped["Transportation"] = exp_m["Transportation"]
exp_grouped["Insurance"] = exp_m["Insurance Other"]

exp_grouped["Flexible Spending"] = exp_m[["Discretionary", "Travel", "Buffer", "Desired Extra Spending"]].sum(axis=1)
exp_grouped["Taxes"] = exp_m["Estimated Taxes"]

# ---------------------- VISUALIZATION ----------------------

# 1. Monthly Expenses
st.subheader("1. Where does the money go? (Monthly Expenses)")
st.markdown("Breakdown of your projected monthly cash outflow. **Taxes are a mandatory expense.** 'Flexible Spending' includes your Desired Extra Spending goal.")

fig1, ax1 = plt.subplots(figsize=(12, 6))
groups = ["Housing", "Healthcare", "Food", "Utilities", "Transportation", "Insurance", "Taxes", "Flexible Spending"]
# Palette: Housing (Indigo), Healthcare (Red), Food (Green), Utilities (Light Green), Trans (Lime), Ins (Teal), Taxes (Blue Gray), Flex (Orange)
colors = ["#3f51b5", "#f44336", "#2e7d32", "#4caf50", "#8bc34a", "#009688", "#607d8b", "#ff9800"]
bottom = np.zeros(len(ages))

for col, color in zip(groups, colors):
    vals = exp_grouped[col].values
    ax1.bar(ages, vals, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
    bottom += vals

# Add income lines overlay
ax1.plot(ages, ss62_net.values, color="black", linestyle="-", linewidth=2, label="SS@62 (Net)")
ax1.plot(ages, ss67_net.values, color="black", linestyle="--", linewidth=2, label="SS@67 (Net)")

ax1.set_xlabel("Age")
ax1.set_ylabel("Monthly ($)")
ax1.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax1.grid(axis="y", linestyle=":", alpha=0.3)
st.pyplot(fig1)


# 2. Annual Comparison
st.subheader("2. Can I afford it? (SS Claiming Strategy Comparison)")
st.markdown("Compare your cash flow sustainability under three different Social Security claiming ages: **62, 67, and 70**.")

spend_annual = exp_grouped.sum(axis=1) * 12.0
# Safe Draw
safe_draw_annual = (start_principal * withdraw_pct) * inflation_factors

# Three Scenarios
scenarios = [
    ("Claim at 62", ss62_net * 12.0, "#1976d2"),
    ("Claim at 67", ss67_net * 12.0, "#0d47a1"),
    ("Claim at 70", ss70_net * 12.0, "#002171")
]

fig2, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
x = np.arange(len(ages))

for ax, (label, ss_vals, color) in zip(axes, scenarios):
    # Income Stack
    ax.bar(x, ss_vals, label="SS Income (Net)", color=color, alpha=0.9)
    ax.bar(x, safe_draw_annual, bottom=ss_vals, label=f"Portfolio Draw ({withdraw_pct:.1%})", color="#4caf50", alpha=0.9)
    
    # Spending Line
    ax.plot(x, spend_annual, color="#d32f2f", linewidth=3, label="Spending Need")
    
    # Shortfall
    total_inc = ss_vals + safe_draw_annual
    ax.fill_between(x, total_inc, spend_annual, where=(spend_annual > total_inc), 
                    interpolate=True, color="red", alpha=0.15, hatch="///", label="Shortfall")
    
    ax.set_title(label, fontsize=14, fontweight="bold")
    ax.set_xlabel("Age")
    ax.set_xticks(x[::5])
    ax.set_xticklabels(ages[::5])
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    
    if label == "Claim at 62":
        ax.set_ylabel("Annual Amount ($)")
        ax.legend(loc="upper left")

st.pyplot(fig2)


# 3. Monte Carlo
st.subheader("3. Will I run out? (Reliability)")
st.markdown(f"Stress-testing your portfolio withdrawals based on the **{primary_ss_plan}** strategy selected in the sidebar.")

# Run Full Simulation
# Need Gross Withdrawals (Total Spend + Taxes - SS)
gross_req_annual = np.maximum(0.0, total_spend_pre_tax + active_ss_tax - active_ss_net) * 12.0

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
    block_size = 12 # 1 year blocks to preserve seasonality/autocorrelation
    
    # 2. Block Bootstrap
    # We need (sims, total_months)
    # How many blocks?
    n_blocks = int(np.ceil(total_months / block_size))
    
    # Random start indices for blocks
    # Valid starts are 0 to n_hist - block_size
    starts = rng.integers(0, n_hist - block_size + 1, size=(sims, n_blocks))
    
    # Construct paths
    sim_returns = np.zeros((sims, n_blocks * block_size))
    for i in range(sims):
        # Stitch blocks
        path = []
        for start_idx in starts[i]:
            path.append(hist_real[start_idx : start_idx + block_size])
        sim_returns[i, :] = np.concatenate(path)
        
    # Trim to exact length
    sim_returns = sim_returns[:, :total_months]
    
    # 3. Re-inflate with User's projected inflation
    # (1 + Real) * (1 + UserInfl) - 1
    # User infl is annual. Monthly = (1+infl)^(1/12) - 1
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

mc_bals, mc_ruined = run_full_mc(start_principal, gross_req_annual.values, len(ages), sims, stock_alloc, mc_seed_input)

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
        <p>Likelihood your portfolio lasts until age {horizon_age}.</p>
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
    prob_ruin = mc_ruined.mean()

    fig3, ax3 = plt.subplots(figsize=(20, 8))
    # Plot sample paths (first 50)
    for i in range(min(50, sims)):
        ax3.plot(x_months, mc_bals[i, :], color="#aeb6bf", alpha=0.10) # Light grey for subtle background paths

    # Plot percentiles
    ax3.plot(x_months, p50, color="#1565c0", linewidth=2.5, label="Median Path") # Distinct dark blue
    ax3.fill_between(x_months, p10, p90, color="#1565c0", alpha=0.15, label="10th-90th Percentile") # Light blue fill

    ax3.axhline(0, color="#b71c1c", linestyle="--", linewidth=1.8) # Darker, thicker red for zero line
    ax3.set_ylim(bottom= -start_principal * 0.2, top=np.percentile(mc_bals, 95)) # Zoom in a bit
    ax3.ticklabel_format(style='plain', axis='y', useOffset=False) # Remove scientific notation
    ax3.set_ylabel("Portfolio Balance ($)")
    ax3.set_xlabel("Age")
    ax3.set_title(f"Monte Carlo Simulation: {primary_ss_plan} (Success Rate: {success_rate:.1%})")
    ax3.legend(loc="upper left")
    st.pyplot(fig3, use_container_width=True)
    
st.caption(f"Simulation based on {sims} runs using **Historical Block Bootstrap** (sampling 1-year blocks from Shiller data 1871-Present).")
st.caption("This method captures historical volatility clustering and crashes (e.g., 1929, 2008) better than a simple bell curve.")