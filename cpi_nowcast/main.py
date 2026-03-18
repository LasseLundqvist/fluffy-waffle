"""
CPI Nowcasting CLI — Denmark.

Usage:
    python main.py --nowcast          # Run nowcast for current month
    python main.py --backtest         # Run full backtest
    python main.py --update-data      # Refresh all data sources
    python main.py --dashboard        # Generate HTML dashboard
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import pandas as pd

# ── Bootstrap logging before any other imports ───────────────────────────
import cpi_nowcast._logging  # noqa: F401  (side-effect: configures root logger)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--nowcast",      is_flag=True, help="Run nowcast for current month")
@click.option("--backtest",     is_flag=True, help="Run full expanding-window backtest")
@click.option("--update-data",  is_flag=True, help="Force-refresh all data sources")
@click.option("--dashboard",    is_flag=True, help="Generate HTML dashboard")
@click.option("--start-date",   default=None, help="Override history start date (YYYY-MM-DD)")
@click.option("--output",       default=None, help="Output path for dashboard HTML")
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging")
def cli(
    ctx: click.Context,
    nowcast: bool,
    backtest: bool,
    update_data: bool,
    dashboard: bool,
    start_date: Optional[str],
    output: Optional[str],
    verbose: bool,
) -> None:
    """CPI Nowcasting Pipeline for Denmark."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if ctx.invoked_subcommand is not None:
        return

    # If no flag given, show help
    if not any([nowcast, backtest, update_data, dashboard]):
        click.echo(ctx.get_help())
        return

    from cpi_nowcast.config import HISTORY_START
    _start = start_date or HISTORY_START

    if update_data:
        _run_update_data(_start)

    if nowcast or backtest or dashboard:
        X, y = _load_data(_start, force_refresh=update_data)
        if X.empty or y.empty:
            logger.error("No data available. Run with --update-data first.")
            sys.exit(1)

        if backtest or dashboard:
            forecasts_df, metrics_df = _run_backtest(X, y)
        else:
            forecasts_df, metrics_df = pd.DataFrame(), pd.DataFrame()

        if nowcast:
            _run_nowcast(X, y, forecasts_df)

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
    """Force-refresh all data from upstream APIs."""
    click.echo("Updating data from all sources…")
    try:
        from cpi_nowcast.data.pipeline import fetch_all_raw
        raw = fetch_all_raw(start_date=start_date, force_refresh=True)
        click.echo(
            f"  ✓ Fetched {len(raw)} daily observations × {raw.shape[1]} series"
        )
    except Exception as exc:
        logger.error("Data update failed: %s", exc)
        click.echo(f"  ✗ Data update failed: {exc}", err=True)


def _load_data(
    start_date: str,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load processed (monthly, YoY) feature matrix and target."""
    click.echo("Loading processed data…")
    try:
        from cpi_nowcast.data.pipeline import load_processed
        X, y = load_processed(start_date=start_date, force_refresh=force_refresh)
        click.echo(f"  ✓ {len(y)} monthly observations | {X.shape[1]} features")
        return X, y
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        click.echo(f"  ✗ Data loading failed: {exc}", err=True)
        return pd.DataFrame(), pd.Series()


def _run_backtest(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window backtest and print summary."""
    click.echo("\nRunning expanding-window backtest…")
    try:
        from cpi_nowcast.evaluation.backtest import run_backtest
        forecasts_df, metrics_df = run_backtest(X, y)

        if not metrics_df.empty:
            click.echo("\nBacktest results:")
            click.echo(metrics_df[["rmse", "mae", "mfe"]].round(4).to_string())

        return forecasts_df, metrics_df
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        click.echo(f"  ✗ Backtest failed: {exc}", err=True)
        return pd.DataFrame(), pd.DataFrame()


def _run_nowcast(
    X: pd.DataFrame,
    y: pd.Series,
    forecasts_df: pd.DataFrame,
) -> None:
    """Fit on full history and generate nowcast for the current period."""
    click.echo("\nGenerating nowcast for current period…")
    try:
        from cpi_nowcast.models.bridge import BridgeEquationModel
        from cpi_nowcast.models.midas import MIDASModel
        from cpi_nowcast.models.ensemble import EnsembleModel

        # Fit on all available history
        bridge = BridgeEquationModel()
        bridge.fit(X, y)

        midas = MIDASModel()
        midas.fit(X, y)

        # Predict for the most recent X row (current month stub)
        latest_X = X.iloc[[-1]]
        bridge_pred = bridge.predict(latest_X)
        midas_pred = midas.predict(latest_X)

        preds_df = pd.concat([bridge_pred.rename("bridge"), midas_pred.rename("midas")], axis=1)

        # Ensemble weights from backtest history (if available)
        ensemble = EnsembleModel()
        if not forecasts_df.empty:
            cols = [c for c in ["bridge", "midas"] if c in forecasts_df.columns]
            ensemble.fit(forecasts_df[cols].dropna(), forecasts_df["actual"].dropna())

        ensemble_pred = ensemble.predict(preds_df)
        ensemble_with_ci = ensemble.predict_with_intervals(preds_df)

        t = latest_X.index[0]
        click.echo(f"\n{'='*50}")
        click.echo(f"  Nowcast for: {t.strftime('%B %Y')}")
        click.echo(f"{'='*50}")
        click.echo(f"  Bridge equation : {bridge_pred.iloc[0]:+.2%}")
        click.echo(f"  MIDAS           : {midas_pred.iloc[0]:+.2%}")
        click.echo(f"  Ensemble        : {ensemble_pred.iloc[0]:+.2%}")

        if not ensemble_with_ci.empty:
            lo95 = ensemble_with_ci["lower_95"].iloc[0]
            hi95 = ensemble_with_ci["upper_95"].iloc[0]
            click.echo(f"  95% interval    : [{lo95:+.2%}, {hi95:+.2%}]")

        # Compare to last observed CPI
        if len(y) > 0:
            last_cpi = y.iloc[-1]
            last_date = y.index[-1]
            click.echo(f"\n  Last observed CPI YoY ({last_date.strftime('%b %Y')}): {last_cpi:+.2%}")
            delta = ensemble_pred.iloc[0] - last_cpi
            arrow = "↑" if delta > 0.001 else "↓" if delta < -0.001 else "→"
            click.echo(f"  Direction vs last: {arrow} ({delta:+.2%})")

        click.echo(f"{'='*50}\n")

    except Exception as exc:
        logger.error("Nowcast failed: %s", exc)
        click.echo(f"  ✗ Nowcast failed: {exc}", err=True)


def _run_dashboard(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Optional[Path] = None,
) -> None:
    """Generate and save the HTML dashboard."""
    click.echo("\nGenerating HTML dashboard…")
    try:
        from cpi_nowcast.output.dashboard import build_dashboard
        from cpi_nowcast.data.pipeline import _load_parquet, CACHE_RAW

        raw_df = _load_parquet(CACHE_RAW)

        # Get latest nowcast for header card
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
            raw_df=raw_df,
            latest_nowcast=latest_nowcast,
            previous_nowcast=previous_nowcast,
            output_path=output_path,
        )
        click.echo(f"  ✓ Dashboard saved to: {saved_path}")
        click.echo(f"    Open in browser: file://{saved_path.resolve()}")

    except Exception as exc:
        logger.error("Dashboard generation failed: %s", exc)
        click.echo(f"  ✗ Dashboard failed: {exc}", err=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point used by the package."""
    cli()


if __name__ == "__main__":
    main()
