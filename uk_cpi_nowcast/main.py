"""
UK CPI Nowcasting CLI.

Usage:
    python -m uk_cpi_nowcast.main --nowcast          # Nowcast for current month
    python -m uk_cpi_nowcast.main --backtest         # Expanding-window backtest
    python -m uk_cpi_nowcast.main --update-data      # Refresh all data sources
    python -m uk_cpi_nowcast.main --dashboard        # Generate HTML dashboard
    python -m uk_cpi_nowcast.main --lag-report       # Print selected lags per variable
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

import uk_cpi_nowcast._logging  # noqa: F401  (configures root logger)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--nowcast",      is_flag=True, help="Nowcast for current month")
@click.option("--backtest",     is_flag=True, help="Run full expanding-window backtest")
@click.option("--update-data",  is_flag=True, help="Force-refresh all data sources")
@click.option("--dashboard",    is_flag=True, help="Generate HTML dashboard")
@click.option("--lag-report",   is_flag=True, help="Print selected predictor lags")
@click.option("--max-lags",     default=None, type=int,
              help="Override max lags tested (default from config)")
@click.option("--start-date",   default=None, help="Override history start date (YYYY-MM-DD)")
@click.option("--output",       default=None, help="Output path for dashboard HTML")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def cli(
    ctx: click.Context,
    nowcast: bool,
    backtest: bool,
    update_data: bool,
    dashboard: bool,
    lag_report: bool,
    max_lags: Optional[int],
    start_date: Optional[str],
    output: Optional[str],
    verbose: bool,
) -> None:
    """UK CPI Nowcasting Pipeline — Bridge Equation + MIDAS with lag selection."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if ctx.invoked_subcommand is not None:
        return

    if not any([nowcast, backtest, update_data, dashboard, lag_report]):
        click.echo(ctx.get_help())
        return

    from uk_cpi_nowcast.config import HISTORY_START
    _start = start_date or HISTORY_START

    if update_data:
        _run_update_data(_start)

    if any([nowcast, backtest, dashboard, lag_report]):
        X, y = _load_data(_start, force_refresh=update_data)
        if X.empty or y.empty:
            logger.error("No data available. Run with --update-data first.")
            sys.exit(1)

        bridge_kwargs = {"max_lags": max_lags} if max_lags is not None else {}

        if backtest or dashboard:
            forecasts_df, metrics_df = _run_backtest(X, y, bridge_kwargs=bridge_kwargs)
        else:
            forecasts_df, metrics_df = pd.DataFrame(), pd.DataFrame()

        if nowcast or lag_report:
            _run_nowcast(X, y, forecasts_df, bridge_kwargs=bridge_kwargs,
                         print_lags=lag_report)

        if dashboard:
            _run_dashboard(
                forecasts_df=forecasts_df,
                metrics_df=metrics_df,
                X=X,
                y=y,
                output_path=Path(output) if output else None,
            )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def _run_update_data(start_date: str) -> None:
    click.echo("Updating data from all sources…")
    try:
        from uk_cpi_nowcast.data.pipeline import fetch_all_raw
        raw = fetch_all_raw(start_date=start_date, force_refresh=True)
        click.echo(
            f"  OK Fetched {len(raw)} daily observations × {raw.shape[1]} series"
        )
    except Exception as exc:
        logger.error("Data update failed: %s", exc)
        click.echo(f"  FAIL Data update failed: {exc}", err=True)


def _load_data(
    start_date: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    click.echo("Loading processed data…")
    try:
        from uk_cpi_nowcast.data.pipeline import load_processed
        X, y = load_processed(start_date=start_date, force_refresh=force_refresh)
        click.echo(f"  OK {len(y)} monthly observations | {X.shape[1]} features")
        return X, y
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        click.echo(f"  FAIL Data loading failed: {exc}", err=True)
        return pd.DataFrame(), pd.Series()


def _run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    bridge_kwargs: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    click.echo("\nRunning expanding-window backtest…")
    try:
        from uk_cpi_nowcast.evaluation.backtest import run_backtest
        forecasts_df, metrics_df = run_backtest(
            X, y, bridge_kwargs=bridge_kwargs or {}
        )
        if not metrics_df.empty:
            click.echo("\nBacktest results:")
            click.echo(metrics_df[["rmse", "mae", "mfe"]].round(4).to_string())
        return forecasts_df, metrics_df
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        click.echo(f"  FAIL Backtest failed: {exc}", err=True)
        return pd.DataFrame(), pd.DataFrame()


def _run_nowcast(
    X: pd.DataFrame,
    y: pd.Series,
    forecasts_df: pd.DataFrame,
    bridge_kwargs: Optional[dict] = None,
    print_lags: bool = False,
) -> None:
    click.echo("\nGenerating nowcast for current period…")
    try:
        from uk_cpi_nowcast.models.bridge import BridgeEquationModel
        from uk_cpi_nowcast.models.midas import MIDASModel
        from uk_cpi_nowcast.models.ensemble import EnsembleModel

        bridge = BridgeEquationModel(**(bridge_kwargs or {}))
        bridge.fit(X, y)

        midas = MIDASModel()
        midas.fit(X, y)

        latest_X = X.iloc[[-1]]
        t = latest_X.index[0]

        # Bridge needs full X history so lag shifts have prior-row context.
        # A single-row DataFrame shifted by lag >= 1 always produces NaN.
        bridge_pred_full = bridge.predict(X)
        valid_bridge = bridge_pred_full.dropna()
        bridge_scalar = float(valid_bridge.iloc[-1]) if not valid_bridge.empty else np.nan
        bridge_pred = pd.Series([bridge_scalar], index=[t])

        # MIDAS: same context approach — cpi_mom_lag1 may be NaN for the
        # current month if last month's release isn't out yet.
        midas_pred_full = midas.predict(X)
        valid_midas = midas_pred_full.dropna()
        midas_scalar = float(valid_midas.iloc[-1]) if not valid_midas.empty else np.nan
        midas_pred = pd.Series([midas_scalar], index=[t])

        preds_df = pd.concat(
            [bridge_pred.rename("bridge"), midas_pred.rename("midas")], axis=1
        )

        ensemble = EnsembleModel()
        if not forecasts_df.empty:
            cols = [c for c in ["bridge", "midas"] if c in forecasts_df.columns]
            ensemble.fit(forecasts_df[cols].dropna(), forecasts_df["actual"].dropna())

        ensemble_pred = ensemble.predict(preds_df)
        ensemble_with_ci = ensemble.predict_with_intervals(preds_df)

        t = latest_X.index[0]
        click.echo(f"\n{'='*52}")
        click.echo(f"  UK CPI Nowcast for: {t.strftime('%B %Y')}")
        click.echo(f"{'='*52}")
        click.echo(f"  Bridge equation : {bridge_pred.iloc[0]:+.3%} MoM")
        click.echo(f"  MIDAS           : {midas_pred.iloc[0]:+.3%} MoM")
        click.echo(f"  Ensemble        : {ensemble_pred.iloc[0]:+.3%} MoM")

        if not ensemble_with_ci.empty:
            lo95 = ensemble_with_ci["lower_95"].iloc[0]
            hi95 = ensemble_with_ci["upper_95"].iloc[0]
            click.echo(f"  95% interval    : [{lo95:+.3%}, {hi95:+.3%}] MoM")

        if len(y) > 0:
            last_mom = y.iloc[-1]
            last_date = y.index[-1]
            click.echo(
                f"\n  Last observed CPI MoM ({last_date.strftime('%b %Y')}): {last_mom:+.3%}"
            )
            delta = ensemble_pred.iloc[0] - last_mom
            arrow = "^" if delta > 0.0001 else "v" if delta < -0.0001 else "->"
            click.echo(f"  Direction vs last: {arrow} ({delta:+.3%})")

        click.echo(f"{'='*52}\n")

        # Lag report
        if print_lags and hasattr(bridge, "best_lags_"):
            click.echo("\nSelected lags per predictor (Bridge Equation CV):")
            click.echo(f"  {'Variable':<35} {'Lag (months)':>12}")
            click.echo(f"  {'-'*35} {'-'*12}")
            for var, lag in sorted(bridge.best_lags_.items(), key=lambda x: -x[1]):
                click.echo(f"  {var:<35} {lag:>12}")
            click.echo()

    except Exception as exc:
        logger.error("Nowcast failed: %s", exc)
        click.echo(f"  FAIL Nowcast failed: {exc}", err=True)


def _run_dashboard(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Optional[Path] = None,
) -> None:
    click.echo("\nGenerating HTML dashboard…")
    try:
        from uk_cpi_nowcast.output.dashboard import build_dashboard
        from uk_cpi_nowcast.data.pipeline import _load_parquet, CACHE_RAW
        from uk_cpi_nowcast.models.bridge import BridgeEquationModel

        raw_df = _load_parquet(CACHE_RAW)

        # Fit bridge on full history to get contributions and lags
        bridge = BridgeEquationModel()
        bridge.fit(X, y)
        contributions_df = bridge.feature_contributions(X)
        lag_dict = bridge.best_lags_

        latest_nowcast = None
        previous_nowcast = None
        if "ensemble" in forecasts_df.columns and not forecasts_df["ensemble"].dropna().empty:
            vals = forecasts_df["ensemble"].dropna()
            latest_nowcast = float(vals.iloc[-1])
            if len(vals) > 1:
                previous_nowcast = float(vals.iloc[-2])

        saved_path = build_dashboard(
            forecasts_df=forecasts_df,
            metrics_df=metrics_df if not metrics_df.empty else None,
            contributions_df=contributions_df,
            raw_df=raw_df,
            lag_dict=lag_dict,
            latest_nowcast=latest_nowcast,
            previous_nowcast=previous_nowcast,
            output_path=output_path,
        )
        click.echo(f"  OK Dashboard saved to: {saved_path}")
        click.echo(f"    Open in browser: file://{saved_path.resolve()}")

    except Exception as exc:
        logger.error("Dashboard generation failed: %s", exc)
        click.echo(f"  FAIL Dashboard failed: {exc}", err=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
