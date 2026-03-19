"""
HTML dashboard generator for UK CPI nowcasting.

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
from typing import Optional

import numpy as np
import pandas as pd

from uk_cpi_nowcast.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def _ensure_plotly() -> None:
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("plotly is required for the dashboard. Run: pip install plotly")


def make_fan_chart(
    forecasts_df: pd.DataFrame,
    title: str = "UK CPI Nowcast (HICP YoY)",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Fan chart: actual CPI + ensemble forecast + uncertainty bands."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if "lower_95" in forecasts_df and "upper_95" in forecasts_df:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index.tolist() + forecasts_df.index[::-1].tolist(),
            y=forecasts_df["upper_95"].tolist() + forecasts_df["lower_95"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,94,184,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±95% CI",
            showlegend=True,
        ))

    if "lower_68" in forecasts_df and "upper_68" in forecasts_df:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index.tolist() + forecasts_df.index[::-1].tolist(),
            y=forecasts_df["upper_68"].tolist() + forecasts_df["lower_68"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,94,184,0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±68% CI",
            showlegend=True,
        ))

    for col, color, dash in [
        ("bridge", "#ef553b", "dot"),
        ("midas",  "#00cc96", "dot"),
    ]:
        if col in forecasts_df:
            fig.add_trace(go.Scatter(
                x=forecasts_df.index,
                y=forecasts_df[col],
                mode="lines",
                name=col.title(),
                line=dict(color=color, dash=dash, width=1.5),
            ))

    ens_col = next(
        (c for c in ["ensemble_nowcast", "ensemble"] if c in forecasts_df.columns), None
    )
    if ens_col:
        fig.add_trace(go.Scatter(
            x=forecasts_df.index,
            y=forecasts_df[ens_col],
            mode="lines",
            name="Ensemble",
            line=dict(color="#003087", width=2.5),
        ))

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


def make_forecast_path_chart(
    forecast_path: pd.DataFrame,
    y_actuals: pd.Series,
    title: str = "6-Month CPI MoM Forecast Path",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    Combined chart: last 6 months of realised CPI MoM (solid) +
    6-month direct forecast path per model (dashed), with a vertical
    divider at the forecast origin.

    Parameters
    ----------
    forecast_path : pd.DataFrame
        Indexed by horizon (1..6), columns: bridge, midas, ensemble.
    y_actuals : pd.Series
        Full realised MoM series (monthly DatetimeIndex).
    """
    import plotly.graph_objects as go

    last_6 = y_actuals.dropna().iloc[-6:]
    origin = y_actuals.dropna().index[-1]
    future_dates = pd.date_range(
        start=origin + pd.DateOffset(months=1),
        periods=len(forecast_path),
        freq="MS",
    ).tolist()

    fig = go.Figure()

    # Realised last 6 months
    fig.add_trace(go.Scatter(
        x=last_6.index.tolist(),
        y=last_6.values.tolist(),
        mode="lines+markers",
        name="Realised (last 6m)",
        line=dict(color="#000000", width=2.5),
        marker=dict(size=7),
    ))

    # Forecast paths
    for col, color, dash in [
        ("bridge",   "#ef553b", "dot"),
        ("midas",    "#00cc96", "dot"),
        ("ensemble", "#003087", "dash"),
    ]:
        if col not in forecast_path.columns:
            continue
        vals = forecast_path[col].tolist()
        # Join origin to first forecast point for a continuous line
        fig.add_trace(go.Scatter(
            x=[origin] + future_dates,
            y=[float(last_6.iloc[-1])] + vals,
            mode="lines+markers",
            name=col.title(),
            line=dict(color=color, dash=dash, width=2 if col == "ensemble" else 1.5),
            marker=dict(size=6),
        ))

    origin_str = origin.strftime("%Y-%m-%d")
    end_str = future_dates[-1].strftime("%Y-%m-%d")
    fig.update_layout(
        shapes=[
            dict(
                type="line",
                x0=origin_str, x1=origin_str, y0=0, y1=1, yref="paper",
                line=dict(color="#718096", dash="dash", width=1.5),
            ),
            dict(
                type="rect",
                x0=origin_str, x1=end_str, y0=0, y1=1, yref="paper",
                fillcolor="rgba(0,48,135,0.04)", line_width=0, layer="below",
            ),
        ],
        annotations=[dict(
            x=origin_str, y=1.0, yref="paper",
            text="Forecast origin", showarrow=False,
            xanchor="left", font=dict(color="#718096", size=11),
        )],
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Month",
        yaxis_title="CPI MoM",
        yaxis_tickformat=".2%",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
    )
    return fig


def make_multihorizon_metrics_chart(
    metrics_df: pd.DataFrame,
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    Line chart showing RMSE per model across forecast horizons h=1..6.
    metrics_df is indexed by (horizon, model).
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = {"bridge": "#ef553b", "midas": "#00cc96", "rw": "#aaaaaa", "ar1": "#888888"}
    dashes = {"bridge": "dot", "midas": "dot", "rw": "dash", "ar1": "dash"}

    if "horizon" not in metrics_df.index.names:
        return fig

    horizons = sorted(metrics_df.index.get_level_values("horizon").unique())
    models = metrics_df.index.get_level_values("model").unique()

    for model in models:
        rmse_vals = []
        for h in horizons:
            try:
                rmse_vals.append(float(metrics_df.loc[(h, model), "rmse"]))
            except KeyError:
                rmse_vals.append(np.nan)
        fig.add_trace(go.Scatter(
            x=horizons,
            y=rmse_vals,
            mode="lines+markers",
            name=model.title(),
            line=dict(color=colors.get(model, "#999"), dash=dashes.get(model, "solid"), width=2),
            marker=dict(size=7),
        ))

    fig.update_layout(
        title="RMSE by Forecast Horizon (Direct Multi-Step Backtest)",
        xaxis_title="Horizon (months ahead)",
        yaxis_title="RMSE (pp MoM)",
        xaxis=dict(dtick=1),
        hovermode="x unified",
        template="plotly_white",
        height=420,
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
        marker_color="#003087",
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
    title: str = "Nowcast Decomposition — Variable Contributions",
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Stacked bar chart of per-feature contributions."""
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


def make_lag_chart(lag_dict: dict) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """
    Bar chart showing the selected lag (months) for each predictor.

    Visualises the automatic lag selection from the Bridge Equation model,
    making it easy to see which indicators lead UK CPI and by how much.
    """
    import plotly.graph_objects as go

    variables = list(lag_dict.keys())
    lags = list(lag_dict.values())

    fig = go.Figure(go.Bar(
        x=variables,
        y=lags,
        marker_color="#003087",
        text=[f"L{l}" for l in lags],
        textposition="outside",
    ))
    fig.update_layout(
        title="Selected Lags per Predictor (Bridge Equation CV)",
        xaxis_title="Predictor",
        yaxis_title="Lag (months)",
        yaxis=dict(dtick=1, range=[-0.5, max(lags) + 1.5]),
        xaxis_tickangle=-45,
        template="plotly_white",
        height=420,
    )
    return fig


def make_data_availability_chart(
    raw_df: pd.DataFrame,
) -> "plotly.graph_objects.Figure":  # type: ignore[name-defined]
    """Bar chart showing days since last observation per series."""
    import plotly.graph_objects as go

    last_obs = raw_df.apply(lambda s: s.last_valid_index())
    series_names = raw_df.columns.tolist()
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
    lag_dict: Optional[dict] = None,
    latest_nowcast: Optional[float] = None,
    previous_nowcast: Optional[float] = None,
    forecast_path: Optional[pd.DataFrame] = None,
    y_actuals: Optional[pd.Series] = None,
    multihorizon_metrics: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build and save the full UK CPI nowcast HTML dashboard.

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
    lag_dict : dict, optional
        Predictor → selected lag mapping from Bridge Equation.
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

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"uk_cpi_dashboard_{ts}.html"

    if latest_nowcast is not None and previous_nowcast is not None:
        delta = latest_nowcast - previous_nowcast
        arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
    else:
        arrow = "→"
    nowcast_str = f"{latest_nowcast:.2%}" if latest_nowcast is not None else "N/A"

    figs_html = []

    # 6-month forecast path (shown first, most actionable)
    if forecast_path is not None and y_actuals is not None and not forecast_path.empty:
        fig_path = make_forecast_path_chart(forecast_path, y_actuals)
        figs_html.append(pio.to_html(fig_path, full_html=False, include_plotlyjs="cdn"))

    fig_fan = make_fan_chart(forecasts_df)
    figs_html.append(pio.to_html(
        fig_fan, full_html=False,
        include_plotlyjs="cdn" if not figs_html else False,
    ))

    # Multi-horizon RMSE chart
    if multihorizon_metrics is not None and not multihorizon_metrics.empty:
        fig_mh = make_multihorizon_metrics_chart(multihorizon_metrics)
        figs_html.append(pio.to_html(fig_mh, full_html=False, include_plotlyjs=False))

    if metrics_df is not None and not metrics_df.empty:
        fig_metrics = make_metrics_chart(metrics_df)
        figs_html.append(pio.to_html(fig_metrics, full_html=False, include_plotlyjs=False))

    if lag_dict:
        fig_lags = make_lag_chart(lag_dict)
        figs_html.append(pio.to_html(fig_lags, full_html=False, include_plotlyjs=False))

    if contributions_df is not None and not contributions_df.empty:
        fig_contrib = make_contribution_chart(contributions_df)
        figs_html.append(pio.to_html(fig_contrib, full_html=False, include_plotlyjs=False))

    if raw_df is not None and not raw_df.empty:
        fig_avail = make_data_availability_chart(raw_df)
        figs_html.append(pio.to_html(fig_avail, full_html=False, include_plotlyjs=False))

    metrics_table = ""
    if metrics_df is not None and not metrics_df.empty:
        metrics_table = metrics_df.round(4).to_html(classes="metrics-table", border=0)

    sources_list = (
        "FRED (Brent oil, TTF gas, USD/GBP, Fertilizer) · "
        "ECB SDMX 2.1 (EUR/GBP, OIS 2Y/5Y) · "
        "Eurostat SDMX (UK HICP flash) · "
        "ONS (UK CPI fallback) · "
        "Stooq (BDI) · FAO (Food Price Index) · "
        "Google Trends (GB)"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UK CPI Nowcast</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 0; padding: 20px; background: #f5f7fa; color: #2d3748; }}
  .header {{ background: linear-gradient(135deg, #003087, #0050c8); color: white;
             padding: 24px 32px; border-radius: 12px; margin-bottom: 24px; }}
  .header h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
  .header p {{ margin: 4px 0; opacity: 0.85; font-size: 0.95rem; }}
  .nowcast-card {{ background: white; border-radius: 12px; padding: 24px;
                   margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                   display: inline-block; min-width: 220px; }}
  .nowcast-value {{ font-size: 2.8rem; font-weight: 700; color: #003087; }}
  .nowcast-arrow {{ font-size: 2rem; margin-left: 8px; }}
  .nowcast-label {{ color: #718096; font-size: 0.9rem; margin-top: 4px; }}
  .chart-card {{ background: white; border-radius: 12px; padding: 20px;
                 margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .metrics-table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  .metrics-table th, .metrics-table td {{ padding: 8px 14px; text-align: right;
                                           border-bottom: 1px solid #e2e8f0; }}
  .metrics-table th {{ background: #edf2f7; text-align: center; }}
  .footer {{ text-align: center; color: #a0aec0; font-size: 0.8rem; margin-top: 32px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🇬🇧 UK CPI Nowcasting Dashboard</h1>
  <p>Ensemble nowcast for UK HICP (YoY) — Bridge Equation + MIDAS with automatic lag selection</p>
  <p>Sources: {sources_list}</p>
  <p>Generated: {datetime.now().strftime('%d %B %Y %H:%M')}</p>
</div>

<div class="nowcast-card">
  <div class="nowcast-label">Latest Nowcast (CPI YoY)</div>
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
  UK CPI Nowcast Pipeline &nbsp;|&nbsp; {sources_list}
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard saved to %s", output_path)
    return output_path
