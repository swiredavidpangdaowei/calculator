import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

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
Z_FACTORS = {
    2023: 5, 2024: 7, 2025: 9, 2026: 11,
    2027: 13.625, 2028: 16.25, 2029: 18.875, 2030: 21.5,
}

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
    deadweight = st.number_input("Deadweight (DWT, tonnes)", min_value=1.0, value=50000.0, step=100.0)
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
        width="stretch",
        key="speed_fuel_table",
        column_config={
            "Speed (knots)": st.column_config.NumberColumn(width=110),
            "Fuel Consumption (MT/day)": st.column_config.NumberColumn(width=180),
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
    "Sailing Days": [5.0],
    "Speed (knots)": [14.0],
    "Fuel Type (Sailing)": ["HFO"],
    "Port Days": [2.0],
    "Fuel Type (Port)": ["MGO"],
    "Cargo Weight (%)": [80.0],
})

with st.container(key="legs_container"):
    legs_df = st.data_editor(
        default_legs,
        num_rows="dynamic",
        width="stretch",
        key="legs_table",
        column_config={
            "Departure Port": st.column_config.TextColumn(width=120),
            "Arrival Port": st.column_config.TextColumn(width=120),
            "Sailing Days": st.column_config.NumberColumn(min_value=0.0, width=75),
            "Speed (knots)": st.column_config.NumberColumn(min_value=0.0, width=75),
            "Fuel Type (Sailing)": st.column_config.SelectboxColumn(options=FUEL_TYPES, required=True, width=75),
            "Port Days": st.column_config.NumberColumn(min_value=0.0, width=75),
            "Fuel Type (Port)": st.column_config.SelectboxColumn(options=FUEL_TYPES, required=True, width=75),
            "Cargo Weight (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, width=85),
        },
    )

legs_df = legs_df.dropna(subset=["Sailing Days", "Speed (knots)", "Port Days", "Cargo Weight (%)"])

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
    speed = float(row["Speed (knots)"])
    cargo_pct = float(row["Cargo Weight (%)"])
    sail_fuel_type = row["Fuel Type (Sailing)"]
    port_fuel_type = row["Fuel Type (Port)"]

    if fit_params is not None:
        me_rate = max(0.0, float(polynomial_fit1(speed, *fit_params)))
    elif len(speeds) == 2:
        me_rate = float(np.interp(speed, speeds, rates))
    elif len(speeds) == 1:
        me_rate = float(rates[0])
    else:
        me_rate = 0.0

    distance_nm = speed * sailing_days * 24.0

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
        "Speed (knots)": speed,
        "Distance (nm)": round(distance_nm, 1),
        "Fuel Type (Sailing)": sail_fuel_type,
        "Sailing Fuel (MT)": round(sailing_fuel, 2),
        "Port Days": port_days,
        "Fuel Type (Port)": port_fuel_type,
        "Port Fuel (MT)": round(port_fuel, 2),
        "Total Fuel (MT)": round(total_fuel, 2),
        "CO2 Emissions (t)": round(emissions, 2),
        "Cargo Weight (%)": cargo_pct,
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
            width="stretch",
            column_config={
                "Departure Port": st.column_config.TextColumn(width=95),
                "Arrival Port": st.column_config.TextColumn(width=95),
                "Sailing Days": st.column_config.NumberColumn(width=65),
                "Speed (knots)": st.column_config.NumberColumn(width=65),
                "Distance (nm)": st.column_config.NumberColumn(width=75),
                "Fuel Type (Sailing)": st.column_config.TextColumn(width=70),
                "Sailing Fuel (MT)": st.column_config.NumberColumn(width=80),
                "Port Days": st.column_config.NumberColumn(width=65),
                "Fuel Type (Port)": st.column_config.TextColumn(width=70),
                "Port Fuel (MT)": st.column_config.NumberColumn(width=75),
                "Total Fuel (MT)": st.column_config.NumberColumn(width=80),
                "CO2 Emissions (t)": st.column_config.NumberColumn(width=85),
                "Cargo Weight (%)": st.column_config.NumberColumn(width=75),
                "Cargo Weight (t)": st.column_config.NumberColumn(width=80),
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
    <div style="text-align:center">
        <div style="font-size:0.875rem;color:gray;">CII Grade</div>
        <div style="font-size:4.5rem;font-weight:700;line-height:1;color:{GRADE_COLOR[grade]};">{grade}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col9.metric("EEOI (gCO2/t·nm)", f"{eeoi:.3f}")

with st.expander("Assumptions & methodology"):
    st.markdown(f"""
- **Main-engine sailing fuel** is estimated from a cubic polynomial (a·x³ + b·x² + c) fitted to the speed/fuel table via `scipy.optimize.curve_fit`, evaluated at each leg's sailing speed. Falls back to linear interpolation with only 2 points, or a constant with 1 point.
- **Sailing fuel (MT)** = (fitted ME rate + auxiliary consumption) × sailing days × (1 + weather effect %).
  Weather effect is applied to sailing fuel only, not to port fuel.
- **Port fuel (MT)** = auxiliary consumption × port days, using the port fuel type.
- **CO2 emissions** use IMO carbon factors: HFO = {CARBON_FACTORS['HFO']}, LFO = {CARBON_FACTORS['LFO']}, MGO = {CARBON_FACTORS['MGO']} (t CO2 / t fuel).
- **Distance per leg** = speed (knots) × sailing days × 24.
- **Attained CII (AER)** = total CO2 (g) / (DWT × total distance sailed).
- **Required CII** = a × DWT⁻ᶜ × (1 − Z/100), reference parameters from IMO MEPC.352(78); Z (reduction factor) from MEPC.354(78) for 2023-2026. **Values for 2027-2030 are not yet formally adopted by IMO and are extrapolated (+2%/year) as a placeholder.**
- **CII rating boundaries** (A-E) use the IMO d1-d4 multipliers for the selected vessel type. "Liner" uses the container-ship reference line; General Cargo uses the DWT-segmented reference line (< 20,000 DWT vs ≥ 20,000 DWT).
- **EEOI** = total CO2 (g) / Σ(cargo weight per leg × distance per leg), i.e. grams CO2 per tonne-mile of cargo actually carried.
- This tool is for **estimation and planning purposes only** and is not a substitute for verified IMO DCS / SEEMP reporting.
""")
