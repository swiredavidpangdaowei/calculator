"""Voyage speed/port-time optimizer for the CII & EEOI Voyage Calculator.

Uses scipy.optimize.minimize (method="SLSQP") to find, per voyage leg, the
sailing speed - and, within a tolerance, the port time - that minimises
total main-engine (ME) fuel consumption across the whole voyage, while
holding total voyage duration fixed at the sum of the user's originally
entered sailing + port days. Fuel type per leg is not optimized - it stays
as entered.

This module mirrors app.py's own fuel-consumption formulas (Manual mode:
cubic polynomial curve fit; Actual mode: the log-linear predict_fuel model
with a cargo-weight-derived draft) so its results are directly comparable
to the calculator's own Leg Results / Voyage Summary / CII Forecast.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit, minimize

# ---------------------------------------------------------------------------
# Calculation constants/functions (mirrors app.py)
# ---------------------------------------------------------------------------

CARBON_FACTORS = {"HFO": 3.114, "LFO": 3.151, "MGO": 3.206}

Z_FACTORS = {
    2023: 5, 2024: 7, 2025: 9, 2026: 11,
    2027: 13.625, 2028: 16.25, 2029: 18.875, 2030: 21.5,
    2031: 25.425, 2032: 29.35, 2033: 33.275, 2034: 37.2, 2035: 41.125,
}

GRADE_COLOR = {"A": "#2e7d32", "B": "#8bc34a", "C": "#fbc02d", "D": "#fb8c00", "E": "#d32f2f"}


def get_z_factor(year: int) -> float:
    if year in Z_FACTORS:
        return Z_FACTORS[year]
    return Z_FACTORS[max(Z_FACTORS)]


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


def polynomial_fit1(x, a, b, c):
    return a * x ** 3 + b * x ** 2 + c


def predict_fuel(speed_knots: float, draft_m: float, wind_beaufort: float,
                  speed_percentage: float, alpha: float, delta: float) -> float:
    BETA_WIND = 0.0143316365
    BETA_DRAFT = 0.0392283333
    BETA_SPEED_PERCENT = -0.0263525409

    speed_knots = max(speed_knots, 1e-6)
    ln_fc = (
        alpha
        + BETA_WIND * wind_beaufort
        + BETA_DRAFT * draft_m
        + BETA_SPEED_PERCENT * speed_percentage
        + delta * math.log(speed_knots)
    )
    return math.exp(ln_fc)


@dataclass
class OptimizationResult:
    success: bool
    message: str
    results_df: pd.DataFrame = None
    summary: dict = None
    forecast_years: list = field(default_factory=list)
    forecast_ratios: list = field(default_factory=list)
    baseline_me_fuel: float = 0.0
    optimized_me_fuel: float = 0.0


def _build_me_rate_fn(speed_fuel_mode, speed_fuel_df, wind, speed_percentage,
                       alpha, delta, slope, intercept, cargo_pcts):
    """Returns me_rate(leg_index, speed) -> MT/day main-engine rate, matching
    app.py's Manual vs Actual mode logic exactly."""
    if speed_fuel_mode == "Manual":
        speeds_arr = speed_fuel_df["Speed (knots)"].to_numpy()
        rates_arr = speed_fuel_df["Fuel Consumption (MT/day)"].to_numpy()
        fit_params = None
        if len(speeds_arr) >= 3:
            try:
                fit_params, _ = curve_fit(polynomial_fit1, speeds_arr, rates_arr)
            except RuntimeError:
                fit_params = None

        def me_rate(_leg_idx, speed):
            if fit_params is not None:
                return max(0.0, float(polynomial_fit1(speed, *fit_params)))
            elif len(speeds_arr) == 2:
                return float(np.interp(speed, speeds_arr, rates_arr))
            elif len(speeds_arr) == 1:
                return float(rates_arr[0])
            return 0.0

        return me_rate

    def me_rate(leg_idx, speed):
        if speed <= 0:
            return 0.0
        intended_draft = slope * cargo_pcts[leg_idx] + intercept
        return max(0.0, predict_fuel(speed, intended_draft, wind, speed_percentage, alpha, delta))

    return me_rate


def optimize_voyage(
    legs_df: pd.DataFrame,
    speed_fuel_df: pd.DataFrame,
    aux_consumption: float,
    weather_pct: float,
    deadweight: float,
    vessel_type: str,
    cii_year: int,
    speed_fuel_mode: str,
    wind=None,
    speed_percentage=None,
    alpha=None,
    delta=None,
    slope=None,
    intercept=None,
    sailing_tolerance: float = 0.10,
    port_tolerance: float = 0.10,
    forecast_years_count: int = 5,
) -> OptimizationResult:
    """Minimise total ME fuel consumption over per-leg speed (and, within
    tolerance, port days), subject to:
      1. total voyage duration == sum(original sailing + port days)  [eq]
      2. per-leg sailing days within +-sailing_tolerance of original  [ineq]
      3. per-leg port days within +-port_tolerance of original        [ineq]
      4. speed within [min, max] of speed_fuel_df (no extrapolation)  [bounds]
      5. speed > 0                                                    [bounds]
    Fuel type per leg is fixed (not a decision variable).
    """
    n = len(legs_df)
    if n == 0:
        return OptimizationResult(success=False, message="No voyage legs to optimize.")

    distances = legs_df["Distance (nm)"].to_numpy(dtype=float)
    orig_sailing_days = legs_df["Sailing Days"].to_numpy(dtype=float)
    orig_port_days = legs_df["Port Days"].to_numpy(dtype=float)
    cargo_pcts = legs_df["Cargo Weight (%)"].to_numpy(dtype=float)
    sail_fuel_types = legs_df["Fuel Type (Sailing)"].tolist()
    port_fuel_types = legs_df["Fuel Type (Port)"].tolist()
    departure_ports = legs_df["Departure Port"].tolist()
    arrival_ports = legs_df["Arrival Port"].tolist()

    speed_min = float(speed_fuel_df["Speed (knots)"].min())
    speed_max = float(speed_fuel_df["Speed (knots)"].max())
    if speed_min <= 0:
        speed_min = 1e-3
    if speed_max < speed_min:
        speed_max = speed_min

    me_rate = _build_me_rate_fn(
        speed_fuel_mode, speed_fuel_df, wind, speed_percentage, alpha, delta, slope, intercept, cargo_pcts
    )

    total_original_duration = float(orig_sailing_days.sum() + orig_port_days.sum())

    # x = [speed_0..speed_{n-1}, port_days_0..port_days_{n-1}]
    x0_speed = np.where(
        orig_sailing_days > 0,
        distances / (24.0 * np.maximum(orig_sailing_days, 1e-9)),
        speed_min,
    )
    x0_speed = np.clip(x0_speed, speed_min, speed_max)
    x0 = np.concatenate([x0_speed, orig_port_days])

    def sailing_days_of(x, i):
        speed = max(x[i], 1e-9)
        return distances[i] / (24.0 * speed)

    def objective(x):
        total = 0.0
        for i in range(n):
            speed = x[i]
            sd = sailing_days_of(x, i)
            total += me_rate(i, speed) * sd * (1 + weather_pct / 100.0)
        return total

    constraints = []

    def duration_eq(x):
        port_days = x[n:]
        sailing_days = np.array([sailing_days_of(x, i) for i in range(n)])
        return sailing_days.sum() + port_days.sum() - total_original_duration

    constraints.append({"type": "eq", "fun": duration_eq})

    for i in range(n):
        lo_sail = orig_sailing_days[i] * (1 - sailing_tolerance)
        hi_sail = orig_sailing_days[i] * (1 + sailing_tolerance)
        constraints.append({"type": "ineq", "fun": (lambda x, i=i, lo=lo_sail: sailing_days_of(x, i) - lo)})
        constraints.append({"type": "ineq", "fun": (lambda x, i=i, hi=hi_sail: hi - sailing_days_of(x, i))})

        lo_port = orig_port_days[i] * (1 - port_tolerance)
        hi_port = orig_port_days[i] * (1 + port_tolerance)
        constraints.append({"type": "ineq", "fun": (lambda x, i=i, lo=lo_port: x[n + i] - lo)})
        constraints.append({"type": "ineq", "fun": (lambda x, i=i, hi=hi_port: hi - x[n + i])})

    bounds = [(speed_min, speed_max)] * n + [(0.0, None)] * n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    baseline_me_fuel = objective(x0)

    if not result.success:
        return OptimizationResult(success=False, message=result.message, baseline_me_fuel=baseline_me_fuel)

    opt_speeds = result.x[:n]
    opt_port_days = result.x[n:]
    optimized_me_fuel = objective(result.x)

    rows = []
    for i in range(n):
        speed = float(opt_speeds[i])
        sailing_days_i = sailing_days_of(result.x, i)
        port_days_i = float(opt_port_days[i])

        rate = me_rate(i, speed)
        sailing_fuel = (rate + aux_consumption) * sailing_days_i * (1 + weather_pct / 100.0)
        port_fuel = aux_consumption * port_days_i
        total_fuel = sailing_fuel + port_fuel
        emissions = sailing_fuel * CARBON_FACTORS[sail_fuel_types[i]] + port_fuel * CARBON_FACTORS[port_fuel_types[i]]
        cargo_weight = cargo_pcts[i] / 100.0 * deadweight

        rows.append({
            "Departure Port": departure_ports[i],
            "Arrival Port": arrival_ports[i],
            "Sailing Days": round(sailing_days_i, 2),
            "Speed (knots)": round(speed, 2),
            "Distance (nm)": round(distances[i], 1),
            "Fuel Type (Sailing)": sail_fuel_types[i],
            "Sailing Fuel (MT)": round(sailing_fuel, 2),
            "Port Days": round(port_days_i, 2),
            "Fuel Type (Port)": port_fuel_types[i],
            "Port Fuel (MT)": round(port_fuel, 2),
            "Total Fuel (MT)": round(total_fuel, 2),
            "CO2 Emissions (t)": round(emissions, 2),
            "Cargo Weight (t)": round(cargo_weight, 1),
        })

    results_df = pd.DataFrame(rows)

    total_sailing_days = results_df["Sailing Days"].sum()
    total_port_days = results_df["Port Days"].sum()
    total_distance = results_df["Distance (nm)"].sum()
    total_co2 = results_df["CO2 Emissions (t)"].sum()
    total_transport_work = (results_df["Cargo Weight (t)"] * results_df["Distance (nm)"]).sum()

    a, c, d1, d2, d3, d4 = get_reference_params(vessel_type, deadweight)
    z = get_z_factor(cii_year)
    cii_required = a * deadweight ** (-c) * (1 - z / 100.0)
    cii_attained = (total_co2 * 1e6) / (deadweight * total_distance) if total_distance > 0 else 0.0
    cii_ratio = cii_attained / cii_required if cii_required > 0 else 0.0
    grade = cii_grade(cii_ratio, d1, d2, d3, d4)
    eeoi = (total_co2 * 1e6) / total_transport_work if total_transport_work > 0 else 0.0

    summary = {
        "total_sailing_days": total_sailing_days,
        "total_port_days": total_port_days,
        "total_days": total_sailing_days + total_port_days,
        "total_distance": total_distance,
        "total_co2": total_co2,
        "cii_required": cii_required,
        "cii_attained": cii_attained,
        "grade": grade,
        "eeoi": eeoi,
        "d1": d1, "d2": d2, "d3": d3, "d4": d4,
    }

    forecast_years = [cii_year + i for i in range(forecast_years_count)]
    forecast_ratios = []
    for yr in forecast_years:
        z_yr = get_z_factor(yr)
        req_yr = a * deadweight ** (-c) * (1 - z_yr / 100.0)
        forecast_ratios.append(cii_attained / req_yr if req_yr > 0 else 0.0)

    return OptimizationResult(
        success=True,
        message=result.message,
        results_df=results_df,
        summary=summary,
        forecast_years=forecast_years,
        forecast_ratios=forecast_ratios,
        baseline_me_fuel=baseline_me_fuel,
        optimized_me_fuel=optimized_me_fuel,
    )


def render_forecast_figure(vessel_name, forecast_years, forecast_ratios, d1, d2, d3, d4):
    """Same visual style as app.py's own CII Forecast chart."""
    upper_bound = max(max(forecast_ratios), d4) * 1.25
    lower_bound = 0.75 * min(forecast_ratios)

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
    ax.set_title(f"{vessel_name} - Optimized CII Forecast ({forecast_years[0]}-{forecast_years[-1]})")

    legend_handles = [mpatches.Patch(color=color, alpha=0.35, label=f"Grade {label}")
                      for _, _, color, label in band_defs]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig.tight_layout()
    return fig
