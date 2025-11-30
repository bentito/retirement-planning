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
        annual_spend_from_portfolio = st.number_input("Desired Extra Spending Goal ($/yr)", 0.0, 500000.0, 30000.0, 1000.0)

    st.header("3. Income & Assets")
    with st.expander("Social Security"):
        you_ss_62 = st.number_input("Your SS @ 62 ($/mo)", 0.0, 10000.0, 2577.0, 10.0)
        you_ss_67 = st.number_input("Your SS @ 67 ($/mo)", 0.0, 10000.0, 3681.0, 10.0)
        spouse_spousal_benefit = st.number_input("Spousal Benefit ($/mo)", 0.0, 10000.0, float(3681 * 0.5), 10.0)
        spouse_claim_age_spouse = st.number_input("Spouse Claim Age", 62, 70, 67, 1)
        primary_ss_plan = st.selectbox("Scenario Plan", ["Claim at 62", "Claim at 67"], index=1)

    with st.expander("Portfolio Assumptions"):
        start_principal = st.number_input("Current Portfolio ($)", 0.0, 20000000.0, 800000.0, 1000.0)
        # Tiers for the chart
        tier1 = st.number_input("Chart Tier 1 ($)", 0.0, 10000000.0, 500000.0, 1000.0)
        tier2 = st.number_input("Chart Tier 2 ($)", 0.0, 10000000.0, 750000.0, 1000.0)
        tier3 = st.number_input("Chart Tier 3 ($)", 0.0, 10000000.0, 1750000.0, 1000.0)
        withdraw_pct = st.slider("Chart Draw Rate (%)", 0.0, 10.0, 5.0, 0.1) / 100.0
        
    with st.expander("Taxes"):
        withdraw_tax_rate = st.slider("Portfolio Tax Rate (%)", 0.0, 50.0, 15.0, 0.5) / 100.0
        ss_taxable_pct = st.slider("SS Taxable Portion (%)", 0.0, 100.0, 85.0, 1.0) / 100.0
        ss_marginal_rate = st.slider("SS Marginal Tax (%)", 0.0, 50.0, 12.0, 0.5) / 100.0

    st.header("4. Simulation")
    sims = st.slider("Simulations", 100, 5000, 1000, 100)
    mean_ann = st.slider("Mean Return (%)", -5.0, 15.0, 7.0, 0.1) / 100.0
    vol_ann = st.slider("Volatility (%)", 0.0, 50.0, 15.0, 0.5) / 100.0
    mc_seed_input = st.number_input("Random Seed", -1, 999999, 42, 1)

# ---------------------- Logic: Calculation ----------------------
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
exp_m["Desired Extra Spending"] = (annual_spend_from_portfolio / 12.0) * inflation_factors

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

ss62_net, ss62_tax = get_net_ss(ss62_gross)
ss67_net, ss67_tax = get_net_ss(ss67_gross)

# Active Scenario
if primary_ss_plan == "Claim at 62":
    active_ss_net, active_ss_tax = ss62_net, ss62_tax
else:
    active_ss_net, active_ss_tax = ss67_net, ss67_tax

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
st.subheader("2. Can I afford it? (Annual Analysis)")
st.markdown("Comparing your **Total Annual Spending Need** (including taxes & goals) vs. **Income Potential** (SS + Portfolio Draws).")

spend_annual = exp_grouped.sum(axis=1) * 12.0
ss62_y = ss62_net * 12.0
ss67_y = ss67_net * 12.0

tiers = [tier1, tier2, tier3]
draws = np.array([t * withdraw_pct for t in tiers])
l1, l2, l3 = draws[0], max(draws[1]-draws[0],0), max(draws[2]-draws[1],0)

fig2, ax2 = plt.subplots(figsize=(12, 6))
w = 0.25
x = np.arange(len(ages))

# Spending Bar
ax2.bar(x - w - 0.05, spend_annual, w, label="Total Spending Need", color="#607d8b", alpha=0.8)

# Helper for stacking income
def plot_stack(x_offset, ss_vals, label, color):
    # SS Base
    ax2.bar(x + x_offset, ss_vals, w, label=label, color=color, alpha=0.9)
    # Layers
    bot = ss_vals.copy()
    ax2.bar(x + x_offset, np.full_like(bot, l1), w, bottom=bot, color="#ffb74d", label=f"Draw on ${tiers[0]/1e3:,.0f}k")
    bot += l1
    ax2.bar(x + x_offset, np.full_like(bot, l2), w, bottom=bot, color="#ff9800", label=f"+ to ${tiers[1]/1e3:,.0f}k")
    bot += l2
    ax2.bar(x + x_offset, np.full_like(bot, l3), w, bottom=bot, color="#f57c00", label=f"+ to ${tiers[2]/1e6:,.1f}M")

plot_stack(0, ss62_y, "SS@62 Net", "#1976d2") # Blue
plot_stack(w + 0.05, ss67_y, "SS@67 Net", "#0d47a1") # Dark Blue

# MC Median Lines
# We need to simulate monthly for accuracy
def run_mc_draws(principal, years, sims, mean, vol, pct):
    total_months = years * 12
    # Simple vectorization
    mu_m = (1 + mean)**(1/12) - 1
    vol_m = vol / np.sqrt(12)
    rng = np.random.default_rng(42) # fixed seed for consistency of lines
    rets = rng.normal(mu_m, vol_m, (sims, total_months))
    
    balances = np.full(sims, principal)
    annual_draws = np.zeros((sims, years))
    target_monthly = (principal * pct) / 12.0
    
    for m in range(total_months):
        balances *= (1 + rets[:, m])
        drawn = np.minimum(balances, target_monthly)
        balances -= drawn
        annual_draws[:, m//12] += drawn
    return np.percentile(annual_draws, 50, axis=0)

y_len = len(ages)
# Only run lines for non-hist simple normal for speed/clarity on this chart
median_l1 = run_mc_draws(tier1, y_len, 500, mean_ann, vol_ann, withdraw_pct)
median_l2 = run_mc_draws(tier2, y_len, 500, mean_ann, vol_ann, withdraw_pct)
median_l3 = run_mc_draws(tier3, y_len, 500, mean_ann, vol_ann, withdraw_pct)

ax2.plot(x, median_l1, color="#3e2723", lw=2, linestyle=":", label=f"Median Realized (${tiers[0]/1e3:,.0f}k)")
ax2.plot(x, median_l2, color="#3e2723", lw=2, linestyle="--", label=f"Median Realized (${tiers[1]/1e3:,.0f}k)")
ax2.plot(x, median_l3, color="#3e2723", lw=2, linestyle="-", label=f"Median Realized (${tiers[2]/1e6:,.1f}M)")

ax2.set_xticks(x[::2])
ax2.set_xticklabels(ages[::2])
ax2.set_ylabel("Annual Amount ($)")
ax2.set_xlabel("Age")

# Dedupe legend
hand, lab = ax2.get_legend_handles_labels()
by_label = dict(zip(lab, hand))
ax2.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1, 1))
ax2.grid(axis="y", linestyle=":", alpha=0.4)
st.pyplot(fig2)


# 3. Monte Carlo
st.subheader("3. Will I run out? (Reliability)")

# Run Full Simulation
# Need Gross Withdrawals (Total Spend + Taxes - SS)
gross_req_annual = np.maximum(0.0, total_spend_pre_tax + active_ss_tax - active_ss_net) * 12.0

def run_full_mc(principal, req_annual, years, sims, mean, vol, seed):
    rng = np.random.default_rng(int(seed) if seed >= 0 else None)
    total_months = years * 12
    mu_m = (1 + mean)**(1/12) - 1
    vol_m = vol / np.sqrt(12)
    
    # Monthly Returns
    rets = rng.normal(mu_m, vol_m, (sims, total_months))
    
    # Monthly Req
    req_monthly = np.repeat(req_annual / 12.0, 12)[:total_months]
    
    balances = np.zeros((sims, total_months + 1))
    balances[:, 0] = principal
    
    for m in range(total_months):
        balances[:, m+1] = (balances[:, m] * (1 + rets[:, m])) - req_monthly[m]
        # Allow negative to track "magnitude of ruin", but for ruin check use <= 0
    
    ruined = np.any(balances <= 0, axis=1)
    return balances, ruined

mc_bals, mc_ruined = run_full_mc(start_principal, gross_req_annual.values, len(ages), sims, mean_ann, vol_ann, mc_seed_input)

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

# Plot MC Paths
fig3, ax3 = plt.subplots(figsize=(12, 5))
# Plot sample paths (first 50)
months = np.arange(mc_bals.shape[1]) / 12.0 + retire_age
for i in range(min(50, sims)):
    ax3.plot(months, mc_bals[i, :], color="gray", alpha=0.1)

# Plot percentiles
p50 = np.percentile(mc_bals, 50, axis=0)
p10 = np.percentile(mc_bals, 10, axis=0)
p90 = np.percentile(mc_bals, 90, axis=0)

ax3.plot(months, p50, color="blue", linewidth=2, label="Median Path")
ax3.fill_between(months, p10, p90, color="blue", alpha=0.1, label="10th-90th Percentile")

ax3.axhline(0, color="red", linestyle="--", linewidth=1)
ax3.set_ylim(bottom= -start_principal * 0.2, top=np.percentile(mc_bals, 95)) # Zoom in a bit
ax3.set_ylabel("Portfolio Balance ($)")
ax3.set_xlabel("Age")
ax3.legend(loc="upper left")
st.pyplot(fig3)

st.caption(f"Simulation based on {sims} runs. Assumes {primary_ss_plan} and nominal returns ~{mean_ann:.1%}, vol ~{vol_ann:.1%}.")