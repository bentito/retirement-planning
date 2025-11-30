
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------- Page config ----------------------
st.set_page_config(page_title="Retirement Planner (Monthly + Annual)", layout="wide")
st.title("Retirement Planner — Monthly (stacked expenses + SS lines) and Annual (bars + layers) + Monte Carlo")

# ---------------------- Sidebar Inputs ----------------------
with st.sidebar:
    st.header("General")
    current_age = st.number_input("Your current age", 40, 85, 58, 1)
    spouse_age = st.number_input("Spouse current age", 30, 85, 49, 1)
    retire_age = st.number_input("Your retirement age", 50, 75, 61, 1)
    horizon_age = st.number_input("Project to age", 75, 105, 90, 1)
    infl = st.slider("Inflation / COLA (annual %)", 0.0, 6.0, 2.0, 0.1) / 100.0

    st.header("Taxes")
    withdraw_tax_rate = st.slider(
        "Effective tax rate on portfolio withdrawals (% of gross draw)",
        0.0, 50.0, 15.0, 0.5
    ) / 100.0
    ss_taxable_pct = st.slider(
        "Percent of Social Security treated as taxable income",
        0.0, 100.0, 85.0, 1.0
    ) / 100.0
    ss_marginal_rate = st.slider(
        "Marginal tax rate applied to taxable Social Security (%)",
        0.0, 50.0, 12.0, 0.5
    ) / 100.0

    st.header("Home & Regular Monthly Costs (today's $)")
    mortgage_payment = st.number_input("Mortgage payment ($/mo)", 0.0, 10000.0, 557.35, 1.0)
    mortgage_years_left = st.number_input("Years left on mortgage", 0, 40, 14, 1)
    property_tax = st.number_input("Property tax ($/mo)", 0.0, 5000.0, 500.0, 1.0)
    food = st.number_input("Food ($/mo)", 0.0, 5000.0, 1000.0, 1.0)
    house_maintenance = st.number_input("House maintenance ($/mo)", 0.0, 5000.0, 250.0, 1.0)
    utilities = st.number_input("Utilities ($/mo)", 0.0, 5000.0, 200.0, 1.0)
    wood = st.number_input("Wood heat ($/mo when older)", 0.0, 5000.0, 80.0, 1.0)
    wood_start_age = st.number_input("Wood cost begins at your age", 60, 100, 75, 1)
    transportation = st.number_input("Transportation ($/mo)", 0.0, 5000.0, 450.0, 1.0)
    insurance_other = st.number_input("Insurance other ($/mo)", 0.0, 5000.0, 250.0, 1.0)
    discretionary = st.number_input("Discretionary ($/mo)", 0.0, 10000.0, 450.0, 1.0)
    travel = st.number_input("Travel ($/mo)", 0.0, 10000.0, 375.0, 1.0)
    buffer = st.number_input("Misc buffer ($/mo)", 0.0, 10000.0, 250.0, 1.0)

    st.header("Healthcare")
    hdhp_annual = st.number_input("HDHP premium per person ($/yr)", 0.0, 30000.0, 6000.0, 100.0)
    med_oop_total = st.number_input("Above-Medicare out-of-pocket per person, 65→90 (today's $)", 0.0, 500000.0, 160000.0, 1000.0)
    med_oop_real_escalation = st.slider("Medicare OOP real escalation (%/yr)", 0.0, 10.0, 3.0, 0.1) / 100.0

    st.header("Thailand / Renting (optional)")
    thailand_half = st.checkbox("Spend half the year in Thailand", value=False)
    thai_ratio = st.slider("Thailand cost ratio vs US", 0.2, 1.0, 0.45, 0.05) if thailand_half else 1.0
    rent_monthly = st.number_input("Local rent (2bd/2ba) if renting home ($/mo)", 0.0, 10000.0, 2000.0, 50.0) if thailand_half else 0.0
    rent_grows_with_cola = st.checkbox("Rent grows with COLA", value=True) if thailand_half else False

    st.header("Social Security (monthly, today's $)")
    you_ss_62 = st.number_input("Your SS @62 ($/mo)", 0.0, 10000.0, 2577.0, 10.0)
    you_ss_67 = st.number_input("Your SS @67 ($/mo)", 0.0, 10000.0, 3681.0, 10.0)
    spouse_spousal_benefit = st.number_input("Spousal benefit (monthly at spouse claim age)", 0.0, 10000.0, float(3681 * 0.5), 10.0)
    spouse_claim_age_spouse = st.number_input("Spouse Social Security claim age (spouse age)", 62, 70, 67, 1)
    primary_ss_plan = st.selectbox(
        "Primary Social Security plan for taxes & Monte Carlo",
        options=["Claim at 62", "Claim at 67"],
        index=1
    )

    st.header("Portfolio Layers (for Annual chart)")
    tier1 = st.number_input("Layer 1 principal ($)", 0.0, 10000000.0, 500000.0, 1000.0)
    tier2 = st.number_input("Layer 2 principal ($)", 0.0, 10000000.0, 750000.0, 1000.0)
    tier3 = st.number_input("Layer 3 principal ($)", 0.0, 10000000.0, 1750000.0, 1000.0)
    withdraw_pct = st.slider("Simple draw % for annual bar chart", 0.0, 10.0, 5.0, 0.1) / 100.0

    st.header("Monte Carlo (portfolio path)")
    start_principal = st.number_input("Starting portfolio ($)", 0.0, 10000000.0, 800000.0, 1000.0)
    annual_spend_from_portfolio = st.number_input("Extra net cash goal from portfolio ($/yr, today's $)", 0.0, 500000.0, 30000.0, 1000.0)
    uploaded_returns = st.file_uploader("Upload monthly return series CSV (single column; decimals like 0.0123). Optional.", type=["csv"])
    use_hist = uploaded_returns is not None
    mean_ann = 0.07
    vol_ann = 0.15
    if not use_hist:
        st.caption("No return file uploaded — using defaults: mean 7%/yr, stdev 15%/yr, normal iid.")
        mean_ann = st.slider("Assumed mean return (%/yr)", -5.0, 15.0, 7.0, 0.1) / 100.0
        vol_ann = st.slider("Assumed volatility (%/yr)", 0.0, 50.0, 15.0, 0.5) / 100.0
    sims = st.slider("Monte Carlo simulations", 100, 10000, 1000, 100)
    mc_seed_input = st.number_input("Monte Carlo random seed (-1 uses a fresh seed each run)", -1, 2_000_000_000, 42, 1)

# ---------------------- Base axes ----------------------
ages = np.arange(retire_age, horizon_age + 1)

hist_monthly_returns = None
if use_hist and uploaded_returns is not None:
    try:
        uploaded_returns.seek(0)
        hist_df = pd.read_csv(uploaded_returns)
        hist_monthly_returns = hist_df.iloc[:, 0].dropna().astype(float).values
        if len(hist_monthly_returns) < 12:
            st.error("Need at least 12 monthly return values in the uploaded CSV. Falling back to parametric returns.")
            hist_monthly_returns = None
    except Exception as exc:
        st.error(f"Could not read uploaded return file ({exc}). Falling back to parametric returns.")
        hist_monthly_returns = None
    finally:
        uploaded_returns.seek(0)
    if hist_monthly_returns is None:
        use_hist = False

def blended(val, ratio):
    return val * (0.5 + 0.5 * ratio)

def escalating_series_initial(total, r, years):
    if r == 0:
        return total / years
    return total * r / ((1 + r) ** years - 1)

# ---------------------- MONTHLY EXPENSES (stacked bars) ----------------------
def build_expenses_monthly():
    cols = ["Mortgage","Property Tax","Rent","Food","House Maintenance","Utilities","Wood",
            "Transportation","Insurance Other","Discretionary","Travel","Buffer",
            "HDHP (You)","HDHP (Spouse)","Medicare OOP (You)","Medicare OOP (Spouse)"]
    df = pd.DataFrame(0.0, index=ages, columns=cols)
    for a in ages:
        yrs = a - retire_age
        f = (1 + infl) ** yrs
        # Mortgage is nominal (fixed), so we do NOT multiply by f.
        df.loc[a, "Mortgage"] = (mortgage_payment if a < current_age + mortgage_years_left else 0)
        df.loc[a, "Property Tax"] = property_tax * f
        ratio = thai_ratio if thailand_half else 1.0
        rent_factor = 0.5 if thailand_half else 0.0
        rent_growth = f if rent_grows_with_cola else 1.0
        df.loc[a, "Rent"] = rent_monthly * rent_factor * rent_growth
        df.loc[a, "Food"] = (blended(food, ratio) if thailand_half else food) * f
        df.loc[a, "Utilities"] = (blended(utilities, ratio) if thailand_half else utilities) * f
        df.loc[a, "Transportation"] = (blended(transportation, ratio) if thailand_half else transportation) * f
        df.loc[a, "Insurance Other"] = (blended(insurance_other, ratio) if thailand_half else insurance_other) * f
        df.loc[a, "Discretionary"] = (blended(discretionary, ratio) if thailand_half else discretionary) * f
        df.loc[a, "Travel"] = travel * f
        df.loc[a, "Buffer"] = buffer * f
        df.loc[a, "House Maintenance"] = house_maintenance * f
        df.loc[a, "Wood"] = (wood if a >= wood_start_age else 0) * f
        # HDHP monthly until 65 for each person
        df.loc[a, "HDHP (You)"] = (hdhp_annual / 12 * f) if a < 65 else 0.0
        spouse_is_under_65 = (spouse_age + (a - current_age)) < 65
        df.loc[a, "HDHP (Spouse)"] = (hdhp_annual / 12 * f) if spouse_is_under_65 else 0.0

    # Medicare OOP monthly (real escalator -> nominal -> /12)
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
extra_net_monthly = pd.Series((annual_spend_from_portfolio / 12.0) * inflation_factors, index=ages)
exp_m["Extra Net Portfolio Cash"] = extra_net_monthly

# ---------------------- SS income lines (monthly, combined) ----------------------
def ss_series_base_monthly_combined(claim_age_you, you_base, spouse_claim_age_spouse, spouse_benefit):
    """Return combined monthly Social Security (gross, before tax)."""
    out = pd.Series(0.0, index=ages, dtype=float)
    for a in ages:
        if a >= claim_age_you:
            out[a] += you_base * ((1 + infl) ** (a - claim_age_you))

    spouse_claim_you_age = current_age + (spouse_claim_age_spouse - spouse_age)
    for a in ages:
        if a >= spouse_claim_you_age:
            out[a] += spouse_benefit * ((1 + infl) ** (a - spouse_claim_you_age))
    return out

def ss_net_and_tax(monthly_gross):
    effective_rate = ss_taxable_pct * ss_marginal_rate
    tax = monthly_gross * effective_rate
    net = monthly_gross - tax
    return net, tax

ss62_m_gross = ss_series_base_monthly_combined(62, you_ss_62, spouse_claim_age_spouse, spouse_spousal_benefit)
ss67_m_gross = ss_series_base_monthly_combined(67, you_ss_67, spouse_claim_age_spouse, spouse_spousal_benefit)

ss62_m, ss62_tax_m = ss_net_and_tax(ss62_m_gross)
ss67_m, ss67_tax_m = ss_net_and_tax(ss67_m_gross)

if primary_ss_plan == "Claim at 62":
    active_ss_net_m = ss62_m
    active_ss_tax_m = ss62_tax_m
    mc_ss_label = "Claim at 62"
else:
    active_ss_net_m = ss67_m
    active_ss_tax_m = ss67_tax_m
    mc_ss_label = "Claim at 67"

spend_annual_pre_tax = exp_m.sum(axis=1) * 12.0
ss_tax_annual_active = active_ss_tax_m * 12.0
ss_net_annual_active = active_ss_net_m * 12.0

shortfall_net_annual = np.maximum(0.0, spend_annual_pre_tax + ss_tax_annual_active - ss_net_annual_active)
if np.isclose(1.0 - withdraw_tax_rate, 0):
    st.error("Effective tax rate on portfolio withdrawals must be below 100%. Adjust the slider in Taxes.")
    gross_withdraw_annual = np.zeros_like(shortfall_net_annual)
else:
    gross_withdraw_annual = np.where(shortfall_net_annual > 0,
                                     shortfall_net_annual / (1 - withdraw_tax_rate),
                                     0.0)
withdraw_tax_annual = gross_withdraw_annual * withdraw_tax_rate

income_taxes_annual = ss_tax_annual_active + withdraw_tax_annual
exp_m["Income Taxes"] = income_taxes_annual / 12.0
gross_withdraw_annual_series = pd.Series(gross_withdraw_annual, index=ages)

# ---------------------- Plot: MONTHLY stacked expenses + SS lines ----------------------
st.subheader("Monthly Stacked Expenses with SS Income Lines")
x = np.arange(len(ages))
fig1, ax1 = plt.subplots(figsize=(18, 9))

stack_order = [
    "Mortgage","Property Tax","Rent","Food","House Maintenance","Utilities","Wood",
    "Transportation","Insurance Other","Discretionary","Travel","Buffer",
    "Extra Net Portfolio Cash","HDHP (You)","HDHP (Spouse)","Medicare OOP (You)",
    "Medicare OOP (Spouse)","Income Taxes"
]
colors = [
    "#7f7f7f", "#c7c7c7", "#ffbb78", "#aec7e8", "#1f77b4", "#98df8a", "#2ca02c",
    "#ff9896", "#d62728", "#ffcc00", "#ff7f0e", "#c5b0d5",
    "#e377c2", "#9467bd", "#8c564b", "#17becf", "#9edae5", "#333333"
]

bottom = np.zeros(len(ages))
for cat, col in zip(stack_order, colors):
    vals = exp_m[cat].values
    ax1.bar(x, vals, bottom=bottom, color=col, label=cat, width=0.7, alpha=0.9, edgecolor="white", linewidth=0.5)
    bottom += vals

# SS monthly income lines (combined)
ax1.plot(x, ss62_m.values, linewidth=3, label="SS@62 net", linestyle="-", color="black", alpha=0.8)
ax1.plot(x, ss67_m.values, linewidth=3, label="SS@67 net", linestyle="--", color="black", alpha=0.8)

ax1.set_xticks(x); ax1.set_xticklabels(ages)
ax1.set_xlabel("Your Age", fontsize=12); ax1.set_ylabel("Monthly Amount ($/mo)", fontsize=12)
ax1.set_title("Monthly Expenses (stacked) vs. Net Social Security Income", fontsize=16)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", ncol=1, frameon=False)
st.pyplot(fig1)

# ---------------------- ANNUAL CHART (bars + layers) ----------------------
st.subheader("Annual Spending vs Income (Bars + Portfolio Layers + MC Median Lines)")

st.markdown(
    """
**How to read this annual chart:**  
- **Gray dashed line** connects your **annual spending** (stacked categories, taxes included).  
- **Blue/cyan stacked bars** show **Social Security (net of estimated tax)** for claiming at **62** vs **67**.  
- The colored blocks stacked **on top** of those SS bases represent a **target portfolio draw** at the chosen rate (e.g., **5%**) for three principal levels (e.g., **$500k → $750k → $1.75M**, shown incrementally).  
- The **dark gray lines** labeled “MC median realized 5% draw” show the **median dollar amount actually withdrawn** in the Monte Carlo simulations when you aim for that constant 5% draw each year.  
    - If a gray line drops **below** the top of the stacked bars, the median path can no longer fund the full target draw in that year.
    """
)

exp_y = exp_m * 12.0
spend_y = exp_y.sum(axis=1)

# SS annual
ss62_y = ss62_m * 12.0
ss67_y = ss67_m * 12.0

# Portfolio layers (annual amounts)
tiers = [tier1, tier2, tier3]
draws_y = np.array([t * withdraw_pct for t in tiers])
layer1_y, layer2_y, layer3_y = draws_y[0], max(draws_y[1] - draws_y[0], 0), max(draws_y[2] - draws_y[1], 0)

# Width and alignment
w = 0.25
fig2, ax2 = plt.subplots(figsize=(18, 9))

# 1. Spending bar (shifted left)
ax2.bar(x - w - 0.03, spend_y.values, w, color="#999999", alpha=0.7, label="Spending (annual total)", edgecolor="white")

def plot_group_annual(xpos, base_vals, label_base, base_color):
    # Base (SS)
    ax2.bar(xpos, base_vals.values, w, color=base_color, alpha=0.9, label=label_base)
    bt = base_vals.values.copy()
    # Layers
    ax2.bar(xpos, np.full_like(bt, layer1_y), w, bottom=bt, color="#ffaa55", label=f"Draw on ${tiers[0]/1e3:,.0f}k")
    bt = bt + layer1_y
    ax2.bar(xpos, np.full_like(bt, layer2_y), w, bottom=bt, color="#55aa55", label=f"+ to ${tiers[1]/1e3:,.0f}k")
    bt = bt + layer2_y
    ax2.bar(xpos, np.full_like(bt, layer3_y), w, bottom=bt, color="#cc4444", label=f"+ to ${tiers[2]/1e6:,.1f}M")

# 2. SS@62 + Layers (Center)
plot_group_annual(x, ss62_y, "SS@62 Net", "#4682b4")

# 3. SS@67 + Layers (Right)
plot_group_annual(x + w + 0.03, ss67_y, "SS@67 Net", "#003366")

# ----- Monte Carlo helpers (block bootstrap + draw tracking) -----
def generate_monthly_paths(years, sims, rng, hist_returns=None, mean_ann=0.07, vol_ann=0.15):
    total_months = years * 12
    if hist_returns is not None and len(hist_returns) > 0:
        returns = np.asarray(hist_returns, dtype=float)
        months_available = len(returns)
        block_size = min(12, months_available)
        n_blocks = math.ceil(total_months / block_size)
        paths = np.empty((sims, total_months), dtype=float)
        for s in range(sims):
            pieces = []
            for _ in range(n_blocks):
                if months_available > block_size:
                    start_idx = int(rng.integers(0, months_available - block_size + 1))
                else:
                    start_idx = 0
                pieces.append(returns[start_idx:start_idx + block_size])
            path = np.concatenate(pieces)
            if len(path) < total_months:
                extra_needed = total_months - len(path)
                extra_idx = rng.integers(0, months_available, size=extra_needed)
                path = np.concatenate([path, returns[extra_idx]])
            paths[s, :] = path[:total_months]
        return paths

    mu_m = (1 + mean_ann) ** (1/12) - 1
    vol_m = vol_ann / math.sqrt(12)
    return rng.normal(mu_m, vol_m, size=(sims, total_months))


def mc_median_available_draw_lines(principal, years, sims, rng, hist_returns, mean_ann, vol_ann, withdraw_pct):
    if principal <= 0 or withdraw_pct <= 0 or sims <= 0:
        return np.zeros(years)
    
    total_months = years * 12
    monthly_paths = generate_monthly_paths(years, sims, rng, hist_returns, mean_ann, vol_ann)
    balances = np.full(sims, principal, dtype=float)
    
    # We track annual realized draws to match the annual chart
    realized_draws_annual = np.zeros((sims, years), dtype=float)
    
    # Target is a fixed nominal percentage of initial principal, taken monthly?
    # Or is it 5% of *current*? The label says "5% draw on $500k", implying fixed nominal.
    target_draw_annual = principal * withdraw_pct
    target_draw_monthly = target_draw_annual / 12.0
    
    for m in range(total_months):
        # Grow (start of month balance grows? or end? standard is growth then draw or draw then growth)
        # We'll do: Growth -> Draw.
        balances *= (1 + monthly_paths[:, m])
        
        drawn = np.minimum(balances, target_draw_monthly)
        balances -= drawn
        
        # Accumulate to the correct annual bucket
        y = m // 12
        realized_draws_annual[:, y] += drawn

    median_draws = np.percentile(realized_draws_annual, 50, axis=0)
    return median_draws

years = int(horizon_age - retire_age + 1)
if (not use_hist):
    mean_default, vol_default = mean_ann, vol_ann
else:
    # when using uploaded series, the sliders are hidden; use nominal defaults for label only
    mean_default, vol_default = 0.07, 0.15
seed_sequence = np.random.SeedSequence() if mc_seed_input < 0 else np.random.SeedSequence(int(mc_seed_input))
children = seed_sequence.spawn(len(tiers) + 1)
capacity_rngs = [np.random.default_rng(child) for child in children[:-1]]
main_rng = np.random.default_rng(children[-1])

hist_series_for_mc = hist_monthly_returns if use_hist else None

for idx, P in enumerate(tiers):
    avail = mc_median_available_draw_lines(P, years, sims, capacity_rngs[idx], hist_series_for_mc, mean_default, vol_default, withdraw_pct)
    style = ["-", "--", ":"][idx % 3]
    color = ["#444444", "#666666", "#222222"][idx % 3]
    label = f"MC median realized 5% draw on ${P:,.0f}"
    ax2.plot(x, avail, linestyle=style, linewidth=2.0, color=color, label=label)

# Dashed line over spending peaks
ax2.plot(x - w, spend_y.values, linestyle="--", linewidth=2, color="gray")

ax2.set_xticks(x); ax2.set_xticklabels(ages)
ax2.set_xlabel("Your Age"); ax2.set_ylabel("Annual Amount ($/yr)")
ax2.set_title("Annual Spending (incl. taxes) vs Net Income — with Portfolio Draw Layers + MC Median Draw Lines")
ax2.grid(axis="y", linestyle=":", alpha=0.5)

hy, ly = ax2.get_legend_handles_labels()
seen = set(); H = []; L = []
for h, l in zip(hy, ly):
    if l not in seen:
        H.append(h); L.append(l); seen.add(l)
ax2.legend(H, L, bbox_to_anchor=(1.02, 1), loc="upper left")

st.pyplot(fig2)


st.markdown(
    """
🔎 **Walkthrough Example (ages 61–65):**  
- **Gray bar** = your total annual spending need (including taxes derived from the selected plan).  
- **Blue section** = Social Security (net of estimated taxes) if claimed at 62 (gray if not yet claimed).  
- **Orange, green, red stacks** = target 5% draws from portfolios of $500k → $750k → $1.75M (incremental).  
- **Dark lines across bars** = median dollars actually withdrawn in the simulations when you target that constant 5% draw.  

**How to read it:**  
- If a line sits **below** the orange layer, the median outcome couldn’t deliver the smallest draw that year.  
- If a line cuts through the green or red layers, the median outcome covers the smaller draws but not the larger ones.  
- When the line runs **along the top** of the stacks, the median path is still meeting the full draw target in those years.  

👉 In plain terms: **bars show what you want after taxes, lines show what the median market outcome actually delivers.**
    """
)


# ---------------------- Monte Carlo (original fan chart retained) ----------------------
st.header("Monte Carlo Withdrawal Simulation")

st.markdown(
    """
**About the Monte Carlo chart:**  
This simulation runs your portfolio through **many randomized return paths** (based on either **uploaded historical monthly returns** or a chosen mean/volatility).  
- Withdrawals follow the **age-by-age gross amounts** required to fund spending (including taxes) after the selected Social Security plan.  
- The shaded band shows the **middle 80% of outcomes (P10–P90)**.  
- The solid line shows the **median (50th percentile)** balance over time.  
- The title reports the **probability of depletion** (balance hitting zero) by your final age.  
It’s not a prediction, just a way to **see a range of plausible futures** given your assumptions.
    """
)
st.caption(f"Monte Carlo assumes {mc_ss_label} and the tax sliders above (withdrawal rate {withdraw_tax_rate:.1%}, SS taxable portion {ss_taxable_pct:.0%} × marginal {ss_marginal_rate:.1%}).")

def run_mc(principal0, gross_withdrawals_annual, years, sims, rng, hist_returns=None, mean_ann=0.07, vol_ann=0.15):
    if years <= 0 or sims <= 0:
        return None, None
    
    total_months = years * 12
    monthly_paths = generate_monthly_paths(years, sims, rng, hist_returns, mean_ann, vol_ann)
    
    balances = np.zeros((sims, total_months + 1), dtype=float)
    balances[:, 0] = principal0
    
    # Expand annual withdrawal requests to monthly
    gross = np.asarray(gross_withdrawals_annual, dtype=float)
    if gross.size < years:
        gross = np.pad(gross, (0, years - gross.size), constant_values=0.0)
    else:
        gross = gross[:years]
    
    monthly_gross_req = np.repeat(gross / 12.0, 12)
    monthly_gross_req = monthly_gross_req[:total_months]
    
    ruined = np.zeros(sims, dtype=bool)

    for m in range(total_months):
        growth = (1 + monthly_paths[:, m])
        balances[:, m] *= growth
        
        draw = np.minimum(balances[:, m], monthly_gross_req[m])
        balances[:, m+1] = balances[:, m] - draw
        
        # Check ruin (if balance drops to near zero)
        ruined |= (balances[:, m+1] <= 1.0)

    return balances, ruined

years = int(horizon_age - retire_age + 1)
if uploaded_returns is not None:
    mean_default, vol_default = 0.07, 0.15
else:
    mean_default, vol_default = mean_ann, vol_ann

balances, ruined = run_mc(
    start_principal,
    gross_withdraw_annual_series.values,
    years,
    sims,
    main_rng,
    hist_returns=hist_series_for_mc,
    mean_ann=mean_default,
    vol_ann=vol_default
)

if balances is not None:
    # Use monthly x-axis
    months_total = balances.shape[1]
    x_months = np.arange(months_total) / 12.0 + retire_age
    
    p50 = np.percentile(balances, 50, axis=0)
    p10 = np.percentile(balances, 10, axis=0)
    p90 = np.percentile(balances, 90, axis=0)
    prob_ruin = ruined.mean()

    fig3, ax3 = plt.subplots(figsize=(18, 6))
    ax3.fill_between(x_months, p10, p90, alpha=0.2, label="P10–P90", color="blue")
    ax3.plot(x_months, p50, lw=2, label="Median", color="blue")
    ax3.set_title(f"Portfolio Monte Carlo — Probability of depletion by age {horizon_age}: {prob_ruin:.1%}", fontsize=14)
    ax3.set_xlabel("Your Age", fontsize=12)
    ax3.set_ylabel("Portfolio Balance ($)", fontsize=12)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax3.legend()
    st.pyplot(fig3)

# ---------------------- Exports ----------------------
st.download_button("Download monthly expense table (CSV)", exp_m.to_csv().encode("utf-8"),
                   file_name="monthly_expenses.csv", mime="text/csv")
st.download_button("Download annual expense table (CSV)", (exp_m * 12.0).to_csv().encode("utf-8"),
                   file_name="annual_expenses.csv", mime="text/csv")
