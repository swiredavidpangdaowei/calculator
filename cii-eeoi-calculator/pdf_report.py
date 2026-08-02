"""Pure-Python PDF export for the CII & EEOI Voyage Calculator.

Replaces the previous Node.js/Puppeteer-based export (generate_report.js),
which launched a headless Chrome browser to screenshot the running app -
something Streamlit Community Cloud does not support (no Node.js, no
browser binaries, no way to launch external processes). This module
instead re-derives the same voyage calculation directly in Python and lays
the report out with reportlab + matplotlib, so it has no browser, no
Node.js, and no subprocess dependency, and works identically locally and on
Streamlit Cloud.

The calculation functions here mirror app.py's formulas exactly, so the
report matches what the calculator shows. Run this file directly to
generate a sample report from the same default data as the app:

    python pdf_report.py
"""

import io
import math
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Calculation constants/functions (mirrors app.py)
# ---------------------------------------------------------------------------

CARBON_FACTORS = {"HFO": 3.114, "LFO": 3.151, "MGO": 3.206}
SHIP_TYPES = ["Bulk Carrier", "Liner (Container Ship)", "General Cargo"]
FUEL_TYPES = ["HFO", "LFO", "MGO"]

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
    """Log-linear fuel consumption model used by app.py's Actual mode."""
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


def compute_leg_results(
    speed_fuel_df, legs_df, aux_consumption, weather_pct, deadweight,
    speed_fuel_mode="Manual", wind=None, speed_percentage=None,
    alpha=None, delta=None, slope=None, intercept=None,
):
    """Per-leg fuel/emissions table - mirrors app.py's calculation loop for
    both Manual mode (cubic curve fit) and Actual mode (log-linear model
    with a cargo-weight-derived draft per leg)."""
    speeds = speed_fuel_df["Speed (knots)"].to_numpy()
    rates = speed_fuel_df["Fuel Consumption (MT/day)"].to_numpy()

    fit_params = None
    if speed_fuel_mode == "Manual" and len(speeds) >= 3:
        try:
            fit_params, _ = curve_fit(polynomial_fit1, speeds, rates)
        except RuntimeError:
            fit_params = None

    results = []
    for _, row in legs_df.iterrows():
        sailing_days = float(row["Sailing Days"])
        port_days = float(row["Port Days"])
        distance_nm = float(row["Distance (nm)"])
        cargo_pct = float(row["Cargo Weight (%)"])
        sail_fuel_type = row["Fuel Type (Sailing)"]
        port_fuel_type = row["Fuel Type (Port)"]

        speed = distance_nm / (sailing_days * 24.0) if sailing_days > 0 else 0.0

        if speed_fuel_mode == "Manual":
            if fit_params is not None:
                me_rate = max(0.0, float(polynomial_fit1(speed, *fit_params)))
            elif len(speeds) == 2:
                me_rate = float(np.interp(speed, speeds, rates))
            elif len(speeds) == 1:
                me_rate = float(rates[0])
            else:
                me_rate = 0.0
        else:
            # Actual mode: draft for this leg is derived from its own cargo
            # weight (slope/intercept), then fed straight into predict_fuel.
            if speed > 0:
                intended_draft = slope * cargo_pct + intercept
                me_rate = max(0.0, predict_fuel(speed, intended_draft, wind, speed_percentage, alpha, delta))
            else:
                me_rate = 0.0

        sailing_fuel = (me_rate + aux_consumption) * sailing_days * (1 + weather_pct / 100.0)
        port_fuel = aux_consumption * port_days
        total_fuel = sailing_fuel + port_fuel
        emissions = sailing_fuel * CARBON_FACTORS[sail_fuel_type] + port_fuel * CARBON_FACTORS[port_fuel_type]
        cargo_weight = cargo_pct / 100.0 * deadweight

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
            "Cargo Weight (t)": round(cargo_weight, 1),
        })
    return pd.DataFrame(results)


def compute_voyage_summary(results_df, deadweight, vessel_type, cii_year):
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

    return {
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
        "a": a, "c": c,
    }


def compute_forecast(deadweight, vessel_type, cii_year, cii_attained, years=5):
    a, c, d1, d2, d3, d4 = get_reference_params(vessel_type, deadweight)
    forecast_years = [cii_year + i for i in range(years)]
    forecast_ratios = []
    for yr in forecast_years:
        z_yr = get_z_factor(yr)
        req_yr = a * deadweight ** (-c) * (1 - z_yr / 100.0)
        forecast_ratios.append(cii_attained / req_yr if req_yr > 0 else 0.0)
    return forecast_years, forecast_ratios, (d1, d2, d3, d4)


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------

def render_forecast_chart(vessel_name, forecast_years, forecast_ratios, d1, d2, d3, d4, title=None):
    upper_bound = max(max(forecast_ratios), d4) * 1.25
    lower_bound = 0.75 * min(forecast_ratios)

    # A=green, B=teal, C=yellow, D=orange, E=red.
    band_defs = [
        (0.0, d1, "#2e7d32", "A"),
        (d1, d2, "#009688", "B"),
        (d2, d3, "#fdd835", "C"),
        (d3, d4, "#fb8c00", "D"),
        (d4, upper_bound, "#d32f2f", "E"),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    for lo, hi, color, _ in band_defs:
        ax.axhspan(lo, hi, color=color, alpha=0.18, zorder=0)

    ax.step(forecast_years, forecast_ratios, where="mid", color="black", alpha=0.35,
            linestyle=":", linewidth=1.8, zorder=2)
    ax.plot(forecast_years, forecast_ratios, linestyle="None", marker="o",
            color="black", markersize=8, zorder=3)

    for yr, ratio in zip(forecast_years, forecast_ratios):
        g = cii_grade(ratio, d1, d2, d3, d4)
        ax.annotate(g, (yr, ratio), textcoords="offset points", xytext=(0, 10),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color="black", zorder=4)

    ax.set_xticks(forecast_years)
    ax.set_xlabel("Year")
    ax.set_ylabel("CII Ratio (Attained / Required)")
    ax.set_ylim(lower_bound, upper_bound)
    if title is None:
        title = f"{vessel_name} - CII Forecast ({forecast_years[0]}-{forecast_years[-1]})"
    ax.set_title(title)

    legend_handles = [mpatches.Patch(color=color, alpha=0.35, label=f"Grade {label}")
                      for _, _, color, label in band_defs]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# PDF layout
# ---------------------------------------------------------------------------

_styles = getSampleStyleSheet()
_title_style = ParagraphStyle("ReportTitle", parent=_styles["Title"], alignment=TA_LEFT, fontSize=20)
_subtitle_style = ParagraphStyle("ReportSubtitle", parent=_styles["Normal"], fontSize=9, textColor=colors.grey)
_heading_style = ParagraphStyle("SectionHeading", parent=_styles["Heading2"], spaceBefore=10, spaceAfter=4)
_caption_style = ParagraphStyle("Caption", parent=_styles["Normal"], fontSize=8, textColor=colors.grey, spaceAfter=6)
_cell_style = ParagraphStyle("Cell", parent=_styles["Normal"], fontSize=7, leading=9)
_header_style = ParagraphStyle("CellHeader", parent=_styles["Normal"], fontSize=7, leading=9, textColor=colors.whitesmoke, fontName="Helvetica-Bold")
_bullet_style = ParagraphStyle("Bullet", parent=_styles["Normal"], fontSize=5.5, leading=12, spaceAfter=1)
_metric_label_style = ParagraphStyle("MetricLabel", parent=_styles["Normal"], fontSize=8, textColor=colors.grey)
_metric_value_style = ParagraphStyle("MetricValue", parent=_styles["Normal"], fontSize=15, fontName="Helvetica-Bold", spaceBefore=1)

TABLE_HEADER_BG = colors.HexColor("#262730")
TABLE_GRID_COLOR = colors.HexColor("#e0e0e0")


def _p(text, style=_cell_style):
    return Paragraph(escape(str(text)), style)


def _dataframe_table(df: pd.DataFrame, col_widths_mm) -> Table:
    header_row = [_p(col, _header_style) for col in df.columns]
    data_rows = [[_p(v) for v in row] for row in df.itertuples(index=False)]
    table = Table([header_row] + data_rows, colWidths=[w * mm for w in col_widths_mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, TABLE_GRID_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _metric_cell(label, value, color=None):
    color_attr = f' color="{color}"' if color else ""
    html = f'<font size=8 color="grey">{escape(label)}</font><br/><font size=15{color_attr}><b>{escape(value)}</b></font>'
    return Paragraph(html, _styles["Normal"])


def _metrics_table(rows):
    table = Table(rows, colWidths=[90 * mm, 90 * mm, 90 * mm])
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def build_pdf_report(
    vessel_name: str,
    deadweight: float,
    vessel_type: str,
    cii_year: int,
    weather_pct: float,
    aux_consumption: float,
    speed_fuel_df: pd.DataFrame,
    legs_df: pd.DataFrame,
    speed_fuel_mode: str = "Manual",
    wind=None,
    speed_percentage=None,
    alpha=None,
    delta=None,
    slope=None,
    intercept=None,
    optimization=None,
) -> bytes:
    """Build the full CII & EEOI voyage report PDF and return it as bytes.

    speed_fuel_mode: "Manual" or "Actual", mirrors app.py's Section 1 toggle.
    wind/speed_percentage/alpha/delta/slope/intercept: only used when
    speed_fuel_mode is "Actual".
    optimization: optional voyage_optimizer.OptimizationResult (or any
    object exposing the same attributes) from app.py's Voyage Optimization
    section. When given and .success is True, an extra "Voyage Optimization"
    section is added on its own page, followed by Assumptions & methodology
    on a further new page.
    """

    results_df = compute_leg_results(
        speed_fuel_df, legs_df, aux_consumption, weather_pct, deadweight,
        speed_fuel_mode=speed_fuel_mode, wind=wind, speed_percentage=speed_percentage,
        alpha=alpha, delta=delta, slope=slope, intercept=intercept,
    )
    summary = compute_voyage_summary(results_df, deadweight, vessel_type, cii_year)
    forecast_years, forecast_ratios, (d1, d2, d3, d4) = compute_forecast(
        deadweight, vessel_type, cii_year, summary["cii_attained"]
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=12 * mm, rightMargin=12 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title=f"{vessel_name} - CII & EEOI Voyage Report",
    )

    story = []

    # --- Title & vessel/voyage parameters -------------------------------
    story.append(Paragraph("CII & EEOI Voyage Calculator", _title_style))
    story.append(Paragraph(
        "Estimate the attained CII (AER), CII rating and EEOI of a voyage from vessel "
        "particulars, a speed/fuel-consumption curve and per-leg voyage data.",
        _subtitle_style,
    ))
    story.append(Spacer(1, 6))

    param_rows = [
        [_metric_cell("Vessel name", str(vessel_name)),
         _metric_cell("Deadweight (DWT, t)", f"{deadweight:,.0f}"),
         _metric_cell("Vessel type", str(vessel_type))],
        [_metric_cell("Year of CII validation", str(cii_year)),
         _metric_cell("Weather effect", f"{weather_pct:g}%"),
         _metric_cell("Auxiliary consumption (MT/day)", f"{aux_consumption:g}")],
    ]
    story.append(KeepTogether([_metrics_table(param_rows), Spacer(1, 4)]))

    # --- 1. Speed / Fuel Consumption Table -------------------------------
    story.append(Paragraph("1. Speed / Fuel Consumption Table", _heading_style))
    story.append(Paragraph(
        "ME + AE fuel consumption (MT/day) at a range of speeds (knots). Leg sailing "
        "speeds are fitted from this curve.", _caption_style,
    ))
    if speed_fuel_mode != "Manual":
        story.append(Paragraph(
            f"This table is based on a Beaufort wind value of {wind} and a speed "
            f"percentage value of {speed_percentage}%, at an average mean draft of "
            f"7 metres, for illustrative purposes.",
            _caption_style,
        ))
    story.append(_dataframe_table(speed_fuel_df, [35, 55]))

    # --- 2. Voyage Legs ----------------------------------------------------
    story.append(Paragraph("2. Voyage Legs", _heading_style))
    story.append(Paragraph("Each row is one voyage leg (departure port &rarr; arrival port).", _caption_style))
    story.append(_dataframe_table(legs_df, [26, 26, 22, 20, 26, 20, 26, 24]))

    # --- 3. Leg Results ------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("3. Leg Results", _heading_style))
    story.append(_dataframe_table(
        results_df,
        [22, 22, 16, 16, 18, 18, 19, 16, 18, 18, 19, 20, 19],
    ))
    story.append(Spacer(1, 15))

    # --- 4. Voyage Summary -------------------------------------------------
    summary_rows = [
        [_metric_cell("Total Sailing Days", f"{summary['total_sailing_days']:.1f}"),
         _metric_cell("Total Port Days", f"{summary['total_port_days']:.1f}"),
         _metric_cell("Total Voyage Days", f"{summary['total_days']:.1f}")],
        [_metric_cell("Total Distance (nm)", f"{summary['total_distance']:,.1f}"),
         _metric_cell("Total CO2 Emissions (t)", f"{summary['total_co2']:,.2f}"),
         _metric_cell("Required CII (gCO2/dwt·nm)", f"{summary['cii_required']:.3f}")],
        [_metric_cell("Attained CII - AER (gCO2/dwt·nm)", f"{summary['cii_attained']:.3f}"),
         _metric_cell("CII Grade", summary["grade"], color=GRADE_COLOR[summary["grade"]]),
         _metric_cell("EEOI (gCO2/t·nm)", f"{summary['eeoi']:.3f}")],
    ]
    story.append(KeepTogether([
        Paragraph("4. Voyage Summary", _heading_style),
        _metrics_table(summary_rows),
    ]))

    # --- 5. CII Forecast -----------------------------------------------
    story.append(PageBreak())
    chart_buf = render_forecast_chart(vessel_name, forecast_years, forecast_ratios, d1, d2, d3, d4)
    story.append(KeepTogether([
        Paragraph("5. CII Forecast", _heading_style),
        Paragraph(
            "Projected CII rating for the 5 years starting at the selected CII validation year.",
            _caption_style,
        ),
        Image(chart_buf, width=230 * mm, height=107 * mm),
    ]))
    if forecast_years[-1] > max(Z_FACTORS):
        story.append(Paragraph(
            f"Note: IMO reduction factors are only defined through {max(Z_FACTORS)}. "
            f"Years after {max(Z_FACTORS)} hold that value flat as a placeholder.",
            _caption_style,
        ))

    # --- 6. Voyage Optimization (optional) --------------------------------
    if optimization is not None and getattr(optimization, "success", False):
        story.append(PageBreak())
        story.append(Paragraph("6. Voyage Optimization", _heading_style))

        reduction_pct = (
            (1 - optimization.optimized_me_fuel / optimization.baseline_me_fuel) * 100
            if optimization.baseline_me_fuel > 0 else 0.0
        )
        story.append(Paragraph(
            f"Optimized total ME fuel: {optimization.optimized_me_fuel:,.2f} MT "
            f"(baseline: {optimization.baseline_me_fuel:,.2f} MT, {reduction_pct:.1f}% reduction). "
            f"Total voyage duration is unchanged; each leg's sailing and port days moved by at "
            f"most &plusmn;10%.",
            _caption_style,
        ))

        story.append(Paragraph("Optimized Leg Results", _heading_style))
        story.append(_dataframe_table(
            optimization.results_df,
            [22, 22, 16, 16, 18, 18, 19, 16, 18, 18, 19, 20, 19],
        ))
        story.append(Spacer(1, 15))

        opt_summary = optimization.summary
        opt_summary_rows = [
            [_metric_cell("Total Sailing Days", f"{opt_summary['total_sailing_days']:.1f}"),
             _metric_cell("Total Port Days", f"{opt_summary['total_port_days']:.1f}"),
             _metric_cell("Total Voyage Days", f"{opt_summary['total_days']:.1f}")],
            [_metric_cell("Total Distance (nm)", f"{opt_summary['total_distance']:,.1f}"),
             _metric_cell("Total CO2 Emissions (t)", f"{opt_summary['total_co2']:,.2f}"),
             _metric_cell("Required CII (gCO2/dwt·nm)", f"{opt_summary['cii_required']:.3f}")],
            [_metric_cell("Attained CII - AER (gCO2/dwt·nm)", f"{opt_summary['cii_attained']:.3f}"),
             _metric_cell("CII Grade", opt_summary["grade"], color=GRADE_COLOR[opt_summary["grade"]]),
             _metric_cell("EEOI (gCO2/t·nm)", f"{opt_summary['eeoi']:.3f}")],
        ]
        story.append(KeepTogether([
            Paragraph("Optimized Voyage Summary", _heading_style),
            _metrics_table(opt_summary_rows),
        ]))

        opt_chart_buf = render_forecast_chart(
            vessel_name, optimization.forecast_years, optimization.forecast_ratios,
            opt_summary["d1"], opt_summary["d2"], opt_summary["d3"], opt_summary["d4"],
            title=(
                f"{vessel_name} - Optimized CII Forecast "
                f"({optimization.forecast_years[0]}-{optimization.forecast_years[-1]})"
            ),
        )
        story.append(KeepTogether([
            Paragraph("Optimized CII Forecast", _heading_style),
            Image(opt_chart_buf, width=230 * mm, height=107 * mm),
        ]))

        story.append(PageBreak())

    # --- Assumptions & methodology --------------------------------------
    story.append(Paragraph("Assumptions &amp; methodology", _heading_style))
    assumptions = [
        "<b>Speed/fuel table</b> is either entered manually, or shown as <b>Actual</b> mode's "
        "illustrative table: fuel at 10-15 knots from a log-linear model at a fixed 7m draft. "
        "In Actual mode, the Weather effect input is disabled since wind is already a model input.",
        "<b>Main-engine sailing fuel in Manual mode</b> is estimated from a cubic polynomial "
        "(a·x³ + b·x² + c) fitted to the speed/fuel table via "
        "scipy.optimize.curve_fit, evaluated at each leg's sailing speed. Falls back to "
        "linear interpolation with only 2 points, or a constant with 1 point.",
        "<b>Main-engine sailing fuel in Actual mode</b> is instead computed per leg directly "
        "from the log-linear model, using that leg's own sailing speed and a leg-specific "
        "draft derived from its cargo weight: draft = slope &times; cargo % + intercept.",
        "<b>Sailing fuel (MT)</b> = (main-engine rate + auxiliary consumption) &times; sailing "
        "days &times; (1 + weather effect %). Weather effect is applied to sailing fuel "
        "only, not to port fuel, and is always 0% in Actual mode.",
        "<b>Port fuel (MT)</b> = auxiliary consumption &times; port days, using the port fuel type.",
        f"<b>CO2 emissions</b> use IMO carbon factors: HFO = {CARBON_FACTORS['HFO']}, "
        f"LFO = {CARBON_FACTORS['LFO']}, MGO = {CARBON_FACTORS['MGO']} (t CO2 / t fuel).",
        "<b>Speed per leg</b> = distance (nm) &divide; (sailing days &times; 24), derived "
        "from the entered distance.",
        "<b>Attained CII (AER)</b> = total CO2 (g) / (DWT &times; total distance sailed).",
        "<b>Required CII</b> = a &times; DWT<super>-c</super> &times; (1 &minus; Z/100), "
        "reference parameters from IMO MEPC.352(78); Z (reduction factor) from MEPC.354(78).",
        "<b>CII rating boundaries</b> (A-E) use the IMO d1-d4 multipliers for the selected "
        "vessel type. \"Liner\" uses the container-ship reference line; General Cargo uses "
        "the DWT-segmented reference line (&lt; 20,000 DWT vs &ge; 20,000 DWT).",
        "<b>EEOI</b> = total CO2 (g) / &Sigma;(cargo weight per leg &times; distance per leg), "
        "i.e. grams CO2 per tonne-mile of cargo actually carried.",
        "<b>Voyage Optimization</b> (optional) uses scipy.optimize.minimize (SLSQP) to "
        "re-solve per-leg sailing speed - and, within &plusmn;10%, per-leg port time - that "
        "minimises total main-engine fuel, holding total voyage duration fixed and keeping "
        "speed within the speed/fuel table's range. Fuel type per leg is not optimized.",
        "This tool is for <b>estimation and planning purposes only</b> and is not a "
        "substitute for verified IMO DCS / SEEMP reporting.",
    ]
    for item in assumptions:
        story.append(Paragraph(f"&bull; {item}", _bullet_style))

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Standalone entry point - generates a sample report using the same default
# data as app.py's own defaults, with no Streamlit server required.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_speed_fuel = pd.DataFrame({
        "Speed (knots)": [10.0, 12.0, 14.0, 16.0, 18.0],
        "Fuel Consumption (MT/day)": [15.0, 22.0, 31.0, 42.0, 55.0],
    })
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

    pdf_bytes = build_pdf_report(
        vessel_name="MV Example",
        deadweight=50000.0,
        vessel_type="Bulk Carrier",
        cii_year=2024,
        weather_pct=0,
        aux_consumption=2.0,
        speed_fuel_df=default_speed_fuel,
        legs_df=default_legs,
    )

    with open("cii_eeoi_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Saved report to cii_eeoi_report.pdf")
