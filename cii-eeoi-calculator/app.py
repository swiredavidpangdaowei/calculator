import os
import subprocess

import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(page_title="CII & EEOI Voyage Calculator", layout="wide")

# Bigger font for the speed/fuel, voyage-leg input, and leg-results tables.
# Streamlit's dataframe/data_editor grid is canvas-rendered, so a plain
# font-size CSS rule has no effect - `zoom` rescales the whole rendered
# grid (text, rows, headers) inside each named container.
st.markdown(
    """
    <style>
    .st-key-speed_fuel_container,
    .st-key-legs_container,
    .st-key-results_container {
        zoom: 1.3;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-left: 3rem !important;
        padding-right: 8rem !important;
    }
    .st-key-summary_container [data-testid="stHorizontalBlock"] {
        gap: 0.6rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

# IMO carbon factors, t CO2 / t fuel (MEPC.1/Circ.895)
CARBON_FACTORS = {"HFO": 3.114, "LFO": 3.151, "MGO": 3.206}

# CII reference-line parameters (CII_ref = a * DWT^-c) and rating-boundary
# multipliers d1-d4, per IMO MEPC.352(78)/MEPC.354(78).
# "Liner" is mapped to the container-ship reference line; general cargo uses
# the two DWT segments defined by IMO.
SHIP_TYPES = ["Bulk Carrier", "Liner (Container Ship)", "General Cargo"]

def get_reference_params(ship_type: str, dwt: float):
    if ship_type == "Bulk Carrier":
        a, c = 4745, 0.622
        d1, d2, d3, d4 = 0.86, 0.94, 1.06, 1.18
    elif ship_type == "Liner (Container Ship)":
        a, c = 1984, 0.489
        d1, d2, d3, d4 = 0.83, 0.94, 1.07, 1.19
    else:  # General Cargo
        if dwt < 20000:
            a, c = 31948, 0.792
        else:
            a, c = 588, 0.3885
        d1, d2, d3, d4 = 0.83, 0.94, 1.06, 1.18
    return a, c, d1, d2, d3, d4

# Reduction factor Z (%) relative to the 2019 baseline.
# 2023-2030 values are formally adopted
# 2031 values onwards are assumed based on a linear projection of reaching 0 in 2050
Z_FACTORS = {
    2023: 5, 2024: 7, 2025: 9, 2026: 11,
    2027: 13.625, 2028: 16.25, 2029: 18.875, 2030: 21.5,
    2031: 25.425, 2032: 29.35, 2033: 33.275, 2034: 37.2, 2035: 41.125,
}


def get_z_factor(year: int) -> float:
    # Reduction factors beyond the last defined year are not yet set; hold
    # the last known value flat as a placeholder.
    if year in Z_FACTORS:
        return Z_FACTORS[year]
    return Z_FACTORS[max(Z_FACTORS)]


FUEL_TYPES = ["HFO", "LFO", "MGO"]

def cii_grade(ratio: float, d1: float, d2: float, d3: float, d4: float) -> str:
    if ratio < d1:
        return "A"
    if ratio < d2:
        return "B"
    if ratio < d3:
        return "C"
    if ratio < d4:
        return "D"
    return "E"

GRADE_COLOR = {"A": "#2e7d32", "B": "#8bc34a", "C": "#fbc02d", "D": "#fb8c00", "E": "#d32f2f"}

# polynomial fit relationship
def polynomial_fit1(x, a, b, c):
    return a*x**3 + b*x**2 + c

# ---------------------------------------------------------------------------
# Sidebar - vessel & voyage-wide inputs
# ---------------------------------------------------------------------------

st.title("CII & EEOI Voyage Calculator")
st.caption(
    "Estimate the attained CII (AER), CII rating and EEOI of a voyage from "
    "vessel particulars, a speed/fuel-consumption curve and per-leg voyage data."
)

with st.sidebar:
    st.header("Vessel Details")
    vessel_name = st.text_input("Vessel name", value="MV Example")
    deadweight = st.number_input("Deadweight (DWT, tonnes)", min_value=1.0, value=50000.0, step=10.0)
    vessel_type = st.selectbox("Vessel type", SHIP_TYPES)

    st.header("CII Settings")
    cii_year = st.selectbox("Year of CII validation", list(range(2023, 2031)), index=1)
    weather_pct = st.selectbox("Weather effect (adds to sailing fuel consumption)", [0, 5, 10, 15, 20], index=0,
                                format_func=lambda x: f"{x}%")

    st.header("Auxiliary Consumption")
    aux_consumption = st.number_input("Auxiliary consumption (MT/day)", min_value=0.0, value=2.0, step=0.1)

# ---------------------------------------------------------------------------
# Speed / fuel consumption table
# ---------------------------------------------------------------------------

st.subheader("1. Speed / Fuel Consumption Table")
st.caption("Enter the main-engine fuel consumption (MT/day) at a range of speeds (knots). "
           "Leg sailing speeds are interpolated from this curve.")

default_speed_fuel = pd.DataFrame({
    "Speed (knots)": [10.0, 12.0, 14.0, 16.0, 18.0],
    "Fuel Consumption (MT/day)": [15.0, 22.0, 31.0, 42.0, 55.0],
})

with st.container(key="speed_fuel_container"):
    speed_fuel_df = st.data_editor(
        default_speed_fuel,
        num_rows="dynamic",
        width=480,
        key="speed_fuel_table",
        column_config={
            "Speed (knots)": st.column_config.NumberColumn(width=200, alignment="left"),
            "Fuel Consumption (MT/day)": st.column_config.NumberColumn(width=280, alignment="left"),
        },
    )

speed_fuel_df = speed_fuel_df.dropna().sort_values("Speed (knots)")

# ---------------------------------------------------------------------------
# Voyage legs table
# ---------------------------------------------------------------------------

st.subheader("2. Voyage Legs")
st.caption("Each row is one voyage leg (departure port -> arrival port).")

default_legs = pd.DataFrame({
    "Departure Port": ["Port A"],
    "Arrival Port": ["Port B"],
    "Distance (nm)": [1680.0],
    "Sailing Days": [5.0],
    "Fuel Type (Sailing)": ["HFO"],
    "Port Days": [2.0],
    "Fuel Type (Port)": ["MGO"],
    "Cargo Weight (%)": [80.0],
})

with st.container(key="legs_container"):
    legs_df = st.data_editor(
        default_legs,
        num_rows="dynamic",
        width=1200,
        key="legs_table",
        column_config={
            "Departure Port": st.column_config.TextColumn(width=150),
            "Arrival Port": st.column_config.TextColumn(width=150),
            "Distance (nm)": st.column_config.NumberColumn(min_value=0.0, width=150, alignment="left"),
            "Sailing Days": st.column_config.NumberColumn(min_value=0.0, width=150, alignment="left"),
            "Fuel Type (Sailing)": st.column_config.SelectboxColumn(options=FUEL_TYPES, required=True, width=150),
            "Port Days": st.column_config.NumberColumn(min_value=0.0, width=100, alignment="left"),
            "Fuel Type (Port)": st.column_config.SelectboxColumn(options=FUEL_TYPES, required=True, width=150),
            "Cargo Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, width=150, alignment="left"),
        },
    )

legs_df = legs_df.dropna(subset=["Sailing Days", "Distance (nm)", "Port Days", "Cargo Weight (%)"])

# ---------------------------------------------------------------------------
# Per-leg calculations
# ---------------------------------------------------------------------------

results = []
speeds = speed_fuel_df["Speed (knots)"].to_numpy()
rates = speed_fuel_df["Fuel Consumption (MT/day)"].to_numpy()

# Fit the speed/fuel curve to a cubic polynomial (a*x**3 + b*x**2 + c) via
# least squares. curve_fit needs at least as many points as parameters (3).
fit_params = None
if len(speeds) >= 3:
    try:
        fit_params, _ = curve_fit(polynomial_fit1, speeds, rates)
    except RuntimeError:
        fit_params = None

for _, row in legs_df.iterrows():
    sailing_days = float(row["Sailing Days"])
    port_days = float(row["Port Days"])
    distance_nm = float(row["Distance (nm)"])
    cargo_pct = float(row["Cargo Weight (%)"])
    sail_fuel_type = row["Fuel Type (Sailing)"]
    port_fuel_type = row["Fuel Type (Port)"]

    speed = distance_nm / (sailing_days * 24.0) if sailing_days > 0 else 0.0

    if fit_params is not None:
        me_rate = max(0.0, float(polynomial_fit1(speed, *fit_params)))
    elif len(speeds) == 2:
        me_rate = float(np.interp(speed, speeds, rates))
    elif len(speeds) == 1:
        me_rate = float(rates[0])
    else:
        me_rate = 0.0

    sailing_fuel = (me_rate + aux_consumption) * sailing_days * (1 + weather_pct / 100.0)
    port_fuel = aux_consumption * port_days
    total_fuel = sailing_fuel + port_fuel

    emissions = sailing_fuel * CARBON_FACTORS[sail_fuel_type] + port_fuel * CARBON_FACTORS[port_fuel_type]

    cargo_weight = cargo_pct / 100.0 * deadweight
    transport_work = cargo_weight * distance_nm

    results.append({
        "Departure Port": row["Departure Port"],
        "Arrival Port": row["Arrival Port"],
        "Sailing Days": sailing_days,
        "Speed (knots)": round(speed, 2),
        "Distance (nm)": round(distance_nm, 1),
        "Fuel Type (Sailing)": sail_fuel_type,
        "Sailing Fuel (MT)": round(sailing_fuel, 2),
        "Port Days": port_days,
        "Fuel Type (Port)": port_fuel_type,
        "Port Fuel (MT)": round(port_fuel, 2),
        "Total Fuel (MT)": round(total_fuel, 2),
        "CO2 Emissions (t)": round(emissions, 2),
        # "Cargo Weight (%)": cargo_pct,
        "Cargo Weight (t)": round(cargo_weight, 1),
    })

results_df = pd.DataFrame(results)

st.subheader("3. Leg Results")
if results_df.empty:
    st.info("Add at least one voyage leg above to see results.")
else:
    with st.container(key="results_container"):
        st.dataframe(
            results_df,
            width=1800,
            column_config={
                "Departure Port": st.column_config.TextColumn(width=150),
                "Arrival Port": st.column_config.TextColumn(width=150),
                "Sailing Days": st.column_config.NumberColumn(width=100, alignment="left"),
                "Speed (knots)": st.column_config.NumberColumn(width=100, alignment="left"),
                "Distance (nm)": st.column_config.NumberColumn(width=100, alignment="left"),
                "Fuel Type (Sailing)": st.column_config.TextColumn(width=150),
                "Sailing Fuel (MT)": st.column_config.NumberColumn(width=150, alignment="left"),
                "Port Days": st.column_config.NumberColumn(width=100, alignment="left"),
                "Fuel Type (Port)": st.column_config.TextColumn(width=150),
                "Port Fuel (MT)": st.column_config.NumberColumn(width=100, alignment="left"),
                "Total Fuel (MT)": st.column_config.NumberColumn(width=150, alignment="left"),
                "CO2 Emissions (t)": st.column_config.NumberColumn(width=150, alignment="left"),
                "Cargo Weight (t)": st.column_config.NumberColumn(width=150, alignment="left"),
            },
        )

# ---------------------------------------------------------------------------
# Voyage totals, CII and EEOI
# ---------------------------------------------------------------------------

st.subheader("4. Voyage Summary")

if results_df.empty:
    st.stop()

total_sailing_days = results_df["Sailing Days"].sum()
total_port_days = results_df["Port Days"].sum()
total_days = total_sailing_days + total_port_days
total_distance = results_df["Distance (nm)"].sum()
total_co2 = results_df["CO2 Emissions (t)"].sum()
total_transport_work = (results_df["Cargo Weight (t)"] * results_df["Distance (nm)"]).sum()

a, c, d1, d2, d3, d4 = get_reference_params(vessel_type, deadweight)
z = Z_FACTORS[cii_year]
cii_required = a * deadweight ** (-c) * (1 - z / 100.0)

cii_attained = (total_co2 * 1e6) / (deadweight * total_distance) if total_distance > 0 else 0.0
cii_ratio = cii_attained / cii_required if cii_required > 0 else 0.0
grade = cii_grade(cii_ratio, d1, d2, d3, d4)

eeoi = (total_co2 * 1e6) / total_transport_work if total_transport_work > 0 else 0.0

with st.container(key="summary_container"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sailing Days", f"{total_sailing_days:.1f}")
    col2.metric("Total Port Days", f"{total_port_days:.1f}")
    col3.metric("Total Voyage Days", f"{total_days:.1f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Distance (nm)", f"{total_distance:,.1f}")
    col5.metric("Total CO2 Emissions (t)", f"{total_co2:,.2f}")
    col6.metric("Required CII (gCO2/dwt·nm)", f"{cii_required:.3f}")

    col7, col8, col9 = st.columns(3)
    col7.metric("Attained CII - AER (gCO2/dwt·nm)", f"{cii_attained:.3f}")
    col8.markdown(
        f"""
        <div style="text-align:left">
            <div style="font-size:0.875rem;color:gray;">CII Grade</div>
            <div style="font-size:4.5rem;font-weight:700;line-height:1;color:{GRADE_COLOR[grade]};">{grade}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col9.metric("EEOI (gCO2/t·nm)", f"{eeoi:.3f}")

# ---------------------------------------------------------------------------
# CII forecast chart
# ---------------------------------------------------------------------------

st.subheader("5. CII Forecast")
st.caption(
    "Projected CII rating for the 5 years starting at the selected CII validation"
)

forecast_years = [cii_year + i for i in range(5)]
forecast_ratios = []
for yr in forecast_years:
    z_yr = get_z_factor(yr)
    req_yr = a * deadweight ** (-c) * (1 - z_yr / 100.0)
    forecast_ratios.append(cii_attained / req_yr if req_yr > 0 else 0.0)

upper_bound = max(max(forecast_ratios), d4) * 1.25
lower_bound = 0.75 * min(forecast_ratios)

# Grade bands as specified: A=green, B=teal, C=yellow, D=orange, E=red.
band_defs = [
    (0.0, d1, "#2e7d32", "A"),
    (d1, d2, "#009688", "B"),
    (d2, d3, "#fdd835", "C"),
    (d3, d4, "#fb8c00", "D"),
    (d4, upper_bound, "#d32f2f", "E"),
]

fig, ax = plt.subplots(figsize=(9, 4.5))

for lo, hi, color, _ in band_defs:
    ax.axhspan(lo, hi, color=color, alpha=0.18, zorder=0)

ax.step(forecast_years, forecast_ratios, where="mid", color="black", alpha=0.35,
        linestyle=":", linewidth=1.8, zorder=2)
ax.plot(forecast_years, forecast_ratios, linestyle="None", marker="o",
        color="black", markersize=8, zorder=3)

for yr, ratio in zip(forecast_years, forecast_ratios):
    yr_grade = cii_grade(ratio, d1, d2, d3, d4)
    ax.annotate(yr_grade, (yr, ratio), textcoords="offset points", xytext=(0, 10),
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="black", zorder=4)

ax.set_xticks(forecast_years)
ax.set_xlabel("Year")
ax.set_ylabel("CII Ratio (Attained / Required)")
ax.set_ylim(lower_bound, upper_bound)
ax.set_title(f"{vessel_name} - CII Forecast ({forecast_years[0]}-{forecast_years[-1]})")

legend_handles = [mpatches.Patch(color=color, alpha=0.35, label=f"Grade {label}")
                  for _, _, color, label in band_defs]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

fig.tight_layout()
st.pyplot(fig)

if forecast_years[-1] > max(Z_FACTORS):
    st.caption(
        f"Note: IMO reduction factors are only defined through {max(Z_FACTORS)}. "
        f"Years after {max(Z_FACTORS)} hold that value flat as a placeholder."
    )

with st.expander("Assumptions & methodology"):
    st.markdown(f"""
- **Main-engine sailing fuel** is estimated from a cubic polynomial (a·x³ + b·x² + c) fitted to the speed/fuel table via `scipy.optimize.curve_fit`, evaluated at each leg's sailing speed. Falls back to linear interpolation with only 2 points, or a constant with 1 point.
- **Sailing fuel (MT)** = (fitted ME rate + auxiliary consumption) × sailing days × (1 + weather effect %).
  Weather effect is applied to sailing fuel only, not to port fuel.
- **Port fuel (MT)** = auxiliary consumption × port days, using the port fuel type.
- **CO2 emissions** use IMO carbon factors: HFO = {CARBON_FACTORS['HFO']}, LFO = {CARBON_FACTORS['LFO']}, MGO = {CARBON_FACTORS['MGO']} (t CO2 / t fuel).
- **Speed per leg** = distance (nm) ÷ (sailing days × 24), derived from the entered distance.
- **Attained CII (AER)** = total CO2 (g) / (DWT × total distance sailed).
- **Required CII** = a × DWT⁻ᶜ × (1 − Z/100), reference parameters from IMO MEPC.352(78); Z (reduction factor) from MEPC.354(78) for 2023-2026. **Values for 2027-2030 are not yet formally adopted by IMO and are extrapolated (+2%/year) as a placeholder.**
- **CII rating boundaries** (A-E) use the IMO d1-d4 multipliers for the selected vessel type. "Liner" uses the container-ship reference line; General Cargo uses the DWT-segmented reference line (< 20,000 DWT vs ≥ 20,000 DWT).
- **EEOI** = total CO2 (g) / Σ(cargo weight per leg × distance per leg), i.e. grams CO2 per tonne-mile of cargo actually carried.
- This tool is for **estimation and planning purposes only** and is not a substitute for verified IMO DCS / SEEMP reporting.
""")

# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------

st.subheader("6. Export")
st.caption(
    "Renders this page to a PDF (A4 landscape) via a Node.js/Puppeteer script. "
    "Requires Node.js, and `npm install` to have been run once in the app folder."
)

if st.button("Create PDF Report"):
    app_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(app_dir, "generate_report.js")
    pdf_path = os.path.join(app_dir, "cii_eeoi_report.pdf")
    port = st.get_option("server.port") or 8501
    app_url = f"http://localhost:{port}"

    with st.spinner("Launching headless Chrome and rendering the PDF..."):
        try:
            result = subprocess.run(
                ["node", script_path, app_url, pdf_path],
                cwd=app_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            st.error(
                "Node.js was not found on this machine. Install Node.js, then run "
                "`npm install` inside the app folder to enable PDF export."
            )
        except subprocess.TimeoutExpired:
            st.error("PDF generation timed out after 120 seconds.")
        else:
            if result.returncode != 0:
                st.error(f"PDF generation failed:\n```\n{result.stderr}\n```")
            elif not os.path.exists(pdf_path):
                st.error("The script reported success but no PDF file was found.")
            else:
                st.success("PDF report generated.")
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        f,
                        file_name="cii_eeoi_report.pdf",
                        mime="application/pdf",
                    )
