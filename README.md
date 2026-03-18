# CPI Nowcasting Pipeline — Danmark 🇩🇰

Et Python-baseret CLI-tool til månedlige nowcast-estimater af dansk inflation (CPI/HICP).

## Arkitektur

```
cpi_nowcast/
├── config.py              # API keys, parametre, variable-definitioner
├── data/
│   ├── fetchers/
│   │   ├── fred.py        # Oliepriser, råvarer via FRED API
│   │   ├── ecb.py         # Valutakurser, swap-renter via ECB SDMX API
│   │   ├── nordpool.py    # Elpriser (ENTSO-E / Nord Pool)
│   │   ├── dst.py         # Danmarks Statistik API (CPI, forbrugertillid)
│   │   ├── eurostat.py    # HICP flash estimate
│   │   ├── gtrends.py     # Google Trends via pytrends + PCA
│   │   └── freight.py     # Baltic Dry Index / FAO Food Price Index
│   ├── pipeline.py        # Orchestrerer alle fetchers, merger data
│   └── cache/             # Lokal cache (parquet)
├── models/
│   ├── bridge.py          # Bridge equation (Ridge/ElasticNet + lag-selektion)
│   ├── midas.py           # MIDAS regression (Almon polynomial)
│   └── ensemble.py        # Simpel ensemble (equal / inv-RMSE vægtet)
├── evaluation/
│   ├── backtest.py        # Expanding-window out-of-sample backtest
│   └── metrics.py         # RMSE, MAE, MFE, Diebold-Mariano test
├── output/
│   ├── dashboard.py       # Interaktivt HTML-dashboard (Plotly)
│   └── reports/           # Gemte nowcast-rapporter
├── main.py                # CLI entry point (package-level)
├── _logging.py            # Logging-konfiguration
requirements.txt
main.py                    # Top-level entry point
README.md
```

## Datakilder

| Kilde | Variable | Frekvens |
|-------|----------|----------|
| FRED | Brent olie, naturgas | Daglig/månedlig |
| ECB SDMX | USD/DKK, EUR/DKK, OIS 2Y/5Y | Daglig |
| Danmarks Statistik | CPI (PRIS111), forbrugertillid (FORV1) | Månedlig |
| Eurostat | HICP flash (EA og DK) | Månedlig |
| ENTSO-E / Nord Pool | Elspot DK1/DK2 | Daglig → månedlig |
| Stooq / FAO | Baltic Dry Index, FAO Food Price Index | Månedlig |
| Google Trends | Inflationsrelaterede søgetermer + PCA | Ugentlig → månedlig |

## Installation

```bash
pip install -r requirements.txt
```

## Konfiguration

Sæt API-nøgler som miljøvariabler (eller i en `.env`-fil):

```bash
export FRED_API_KEY="din_fred_nøgle"        # https://fredaccount.stlouisfed.org/
export ENTSO_E_TOKEN="din_entsoe_token"     # https://transparency.entsoe.eu/
export LOG_LEVEL="INFO"                     # DEBUG for mere output
```

FRED API-nøglen er gratis og anbefalet (øger rate limits markant).
Uden ENTSO-E-token bruges Nord Pools offentlige data som fallback.

## Brug

```bash
# Opdatér alle datakilder (sæt cache)
python main.py --update-data

# Generér nowcast for indeværende måned
python main.py --nowcast

# Kør fuld backtest (expanding window, 2015→nu)
python main.py --backtest

# Generér interaktivt HTML-dashboard
python main.py --dashboard

# Kombiner flag frit:
python main.py --update-data --nowcast --backtest --dashboard

# Åbn dashboard i browser:
open cpi_nowcast/output/reports/dashboard_*.html
```

## Modelspecifikation

### Bridge Equation

```
CPI_yoy(t) = α + Σᵢ βᵢ(Lᵢ)·Xᵢ_yoy(t) + ε(t)
```

- Alle variable i year-over-year vækstrater (ikke niveauer)
- Automatisk lag-selektion (0–3 lags) via time-series CV
- Ridge eller ElasticNet regularisering

### MIDAS Regression

- Almon polynomial lag weights
- Faldback til Ridge hvis daglige data ikke er tilgængelige
- Direkte brug af højfrekvente variable

### Ensemble

- Simpelt gennemsnit eller inverse-RMSE-vægtet kombination
- Vægte beregnes ud fra rolling backtest-performance

## Evaluering

| Metric | Forklaring |
|--------|------------|
| RMSE | Root Mean Squared Error (procentpoint) |
| MAE | Mean Absolute Error |
| MFE | Mean Forecast Error (bias) |
| Skill score | 1 − RMSE(model)/RMSE(benchmark) |
| Diebold-Mariano | Test for signifikant forskel vs. benchmark |

Benchmarks: Random walk + AR(1)

## Output eksempel

```
==================================================
  Nowcast for: March 2026
==================================================
  Bridge equation :  +2.31%
  MIDAS           :  +2.18%
  Ensemble        :  +2.25%
  95% interval    : [+1.45%, +3.05%]

  Last observed CPI YoY (Feb 2026):  +2.10%
  Direction vs last: ↑ (+0.15%)
==================================================
```

## Principper

1. **Vækstrater, ikke niveauer** — undgår spuriøse korrelationer
2. **Real-time information timing** — bruger kun data tilgængeligt på forecast-tidspunktet
3. **Aggressiv caching** — parquet-cache i `cpi_nowcast/data/cache/`
4. **Fail gracefully** — manglende datakilder stopper ikke pipelinen
5. **Reproducerbarhed** — fast random seed (42), timestamps på alle fetches

## Licens

MIT
