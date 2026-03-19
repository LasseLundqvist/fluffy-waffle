"""
HTML dashboard generator using Plotly.

Produces an interactive single-file HTML report with:
1. Nowcast vs realised CPI over time (fan chart)
2. Model comparison (RMSE bar chart)
3. Variable contribution decomposition
4. Data availability calendar
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from cpi_nowcast.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def _ensure_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("plotly is required for the dashboard. Run: pip install plotly")


def make_fan_chart(
    forecasts_df: pd.DataFrame,
    title: str = "CPI Nowcast — Denmark",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    Create a fan chart: actual CPI + ensemble forecast + uncertainty bands.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Must contain columns: actual, ensemble_nowcast, lower_68, upper_68,
        lower_95, upper_95.  Index is DatetimeIndex.
    title : str
        Chart title.
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # 95% confidence band
    if "lower_95" in forecasts_df and "upper_95" in forecasts_df:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index.tolist() + forecasts_df.index[::-1].tolist(),
            y=forecasts_df["upper_95"].tolist() + forecasts_df["lower_95"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(99,110,250,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±95% CI",
            showlegend=True,
        ))

    # 68% confidence band
    if "lower_68" in forecasts_df and "upper_68" in forecasts_df:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index.tolist() + forecasts_df.index[::-1].tolist(),
            y=forecasts_df["upper_68"].tolist() + forecasts_df["lower_68"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(99,110,250,0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±68% CI",
            showlegend=True,
        ))

    # Individual model forecasts
    for col, color, dash in [
        ("bridge", "#ef553b", "dot"),
        ("midas", "#00cc96", "dot"),
    ]:
        if col in forecasts_df:
            fig.add_trace(go.Scatter(
                x=forecasts_df.index,
                y=forecasts_df[col],
                mode="lines",
                name=col.title(),
                line=dict(color=color, dash=dash, width=1.5),
            ))

    # Ensemble forecast
    ens_col = next(
        (c for c in ["ensemble_nowcast", "ensemble"] if c in forecasts_df.columns), None
    )
    if ens_col:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index,
            y=forecasts_df[ens_col],
            mode="lines",
            name="Ensemble",
            line=dict(color="#636efa", width=2.5),
        ))

    # Realised CPI
    if "actual" in forecasts_df:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index,
            y=forecasts_df["actual"],
            mode="lines+markers",
            name="Realised CPI YoY",
            line=dict(color="#000000", width=2),
            marker=dict(size=4),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Date",
        yaxis_title="Year-over-year growth rate",
        yaxis_tickformat=".1%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=500,
    )
    return fig


def make_metrics_chart(metrics_df: pd.DataFrame) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Bar chart comparing RMSE and MAE across models."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics_df.index.tolist(),
        y=metrics_df["rmse"].tolist(),
        name="RMSE",
        marker_color="#636efa",
    ))
    fig.add_trace(go.Bar(
        x=metrics_df.index.tolist(),
        y=metrics_df["mae"].tolist(),
        name="MAE",
        marker_color="#ef553b",
    ))
    fig.update_layout(
        title="Model Comparison — Backtest Metrics",
        xaxis_title="Model",
        yaxis_title="Error (pp)",
        barmode="group",
        template="plotly_white",
        height=400,
    )
    return fig


def make_contribution_chart(
    contributions_df: pd.DataFrame,
    title: str = "Nowcast Decomposition",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Stacked bar chart of per-feature contributions to the nowcast."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in contributions_df.columns:
        fig.add_trace(go.Bar(
            x=contributions_df.index.tolist(),
            y=contributions_df[col].tolist(),
            name=col,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Contribution (pp)",
        barmode="relative",
        template="plotly_white",
        height=450,
    )
    return fig


def make_data_availability_chart(
    raw_df: pd.DataFrame,
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    Heatmap showing last update date and coverage for each data series.
    """
    import plotly.graph_objects as go

    last_obs = raw_df.apply(lambda s: s.last_valid_index())
    coverage = raw_df.notna().mean()

    series_names = raw_df.columns.tolist()
    # Days since last observation
    today = pd.Timestamp.now()
    days_stale = [(today - lo).days if lo is not None else 9999 for lo in last_obs]

    fig = go.Figure(go.Bar(
        x=series_names,
        y=days_stale,
        marker_color=[
            "#2ca02c" if d <= 7 else "#ff7f0e" if d <= 30 else "#d62728"
            for d in days_stale
        ],
        text=[
            f"{lo.strftime('%Y-%m-%d') if lo is not None else 'N/A'} ({d}d)"
            for lo, d in zip(last_obs, days_stale)
        ],
        textposition="outside",
    ))
    fig.update_layout(
        title="Data Availability (days since last observation)",
        xaxis_tickangle=-45,
        yaxis_title="Days stale",
        template="plotly_white",
        height=450,
    )
    return fig


def build_dashboard(
    forecasts_df: pd.DataFrame,
    metrics_df: Optional[pd.DataFrame] = None,
    contributions_df: Optional[pd.DataFrame] = None,
    raw_df: Optional[pd.DataFrame] = None,
    latest_nowcast: Optional[float] = None,
    previous_nowcast: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build and save the full HTML dashboard.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Backtest forecasts with actual and model columns.
    metrics_df : pd.DataFrame, optional
        Backtest metrics table.
    contributions_df : pd.DataFrame, optional
        Feature contributions for the bridge model.
    raw_df : pd.DataFrame, optional
        Raw data for the availability chart.
    latest_nowcast : float, optional
        Most recent nowcast value.
    previous_nowcast : float, optional
        Previous period nowcast for directional indicator.
    output_path : Path, optional
        Where to save the HTML file.

    Returns
    -------
    Path
        Path to the saved HTML file.
    """
    _ensure_plotly()
    import plotly.io as pio
    from plotly.subplots import make_subplots

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"dashboard_{ts}.html"

    # Direction indicator
    if latest_nowcast is not None and previous_nowcast is not None:
        delta = latest_nowcast - previous_nowcast
        arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
    else:
        arrow = "→"
    nowcast_str = f"{latest_nowcast:.2%}" if latest_nowcast is not None else "N/A"

    # Figures
    fig_fan = make_fan_chart(forecasts_df)
    figs_html = [pio.to_html(fig_fan, full_html=False, include_plotlyjs="cdn")]

    if metrics_df is not None and not metrics_df.empty:
        fig_metrics = make_metrics_chart(metrics_df)
        figs_html.append(pio.to_html(fig_metrics, full_html=False, include_plotlyjs=False))

    if contributions_df is not None and not contributions_df.empty:
        fig_contrib = make_contribution_chart(contributions_df)
        figs_html.append(pio.to_html(fig_contrib, full_html=False, include_plotlyjs=False))

    if raw_df is not None and not raw_df.empty:
        fig_avail = make_data_availability_chart(raw_df)
        figs_html.append(pio.to_html(fig_avail, full_html=False, include_plotlyjs=False))

    # Metrics table HTML
    metrics_table = ""
    if metrics_df is not None and not metrics_df.empty:
        metrics_table = metrics_df.round(4).to_html(classes="metrics-table", border=0)

    html = f"""<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CPI Nowcast — Danmark</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; padding: 20px; background: #f5f7fa; color: #2d3748; }}
  .header {{ background: linear-gradient(135deg, #1a365d, #2c5282); color: white;
             padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
  .header p {{ margin: 4px 0; opacity: 0.85; font-size: 0.95rem; }}
  .nowcast-card {{ background: white; border-radius: 12px; padding: 24px;
                   margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                   display: inline-block; min-width: 220px; }}
  .nowcast-value {{ font-size: 2.8rem; font-weight: 700; color: #2c5282; }}
  .nowcast-arrow {{ font-size: 2rem; margin-left: 8px; }}
  .nowcast-label {{ color: #718096; font-size: 0.9rem; margin-top: 4px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 20px;
                 margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  .metrics-table th, .metrics-table td {{ padding: 8px 14px; text-align: right;
                                           border-bottom: 1px solid #e2e8f0; }}
  .metrics-table th {{ background: #edf2f7; text-align: center; }}
  .metrics-table tr:first-child th {{ border-radius: 8px 8px 0 0; }}
  .footer {{ text-align: center; color: #a0aec0; font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🇩🇰 CPI Nowcasting — Danmark</h1>
  <p>Automatisk nowcastingmodel baseret på oliepris, valutakurser, elpriser, fragtrater og Google Trends</p>
  <p>Genereret: {datetime.now().strftime('%d. %B %Y kl. %H:%M')}</p>
</div>

<div class="nowcast-card">
  <div class="nowcast-label">Seneste nowcast (KPIY/Y)</div>
  <div>
    <span class="nowcast-value">{nowcast_str}</span>
    <span class="nowcast-arrow">{arrow}</span>
  </div>
  <div class="nowcast-label">Ensemble model</div>
</div>

<div class="chart-card">
  {"".join(figs_html[:1])}
</div>

{"".join(f'<div class="chart-card">{h}</div>' for h in figs_html[1:])}

{f'<div class="chart-card"><h3>Backtest Metrics</h3>{metrics_table}</div>' if metrics_table else ''}

<div class="footer">
  CPI Nowcast Pipeline &nbsp;|&nbsp; Kilde: FRED, ECB, DST, Eurostat, Nord Pool, Google Trends
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard saved to %s", output_path)
    return output_path
