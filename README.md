# Financial TDA

Explore the changing relationships between financial assets with Topological Data Analysis (TDA). This Python project combines hierarchical clustering, return normality tests, PCA visualization, and persistent homology in a pipeline that downloads prices and exports charts and CSV files.

The project supports exploratory research. It does not include strategy backtesting or an evaluation of market crisis prediction performance.

## Analysis workflow

1. **Download prices:** Retrieve the current S&P 500 constituent list from Wikipedia, take its first 80 symbols by default, and add GLD, TLT, VIXY, and BTC-USD. Download adjusted closing prices through yfinance.
2. **Prepare returns:** Keep weekday prices, forward-fill missing prices, and calculate percentage log returns. Remove assets with less than 95% return coverage, then remove incomplete observations. Returns are not backward-filled.
3. **Select representative assets:** Apply Ward hierarchical clustering to correlation distances and select the asset with the smallest mean within-cluster distance from each cluster. The default target is at most 20 representatives.
4. **Explore distributions and structure:** Calculate skewness, kurtosis, Shapiro-Wilk and Jarque-Bera tests; plot return distributions, Q-Q plots, PCA projections, and correlation structure.
5. **Compute persistent homology:** Build Vietoris-Rips filtrations over rolling windows of 60 observations by default. Calculate separate H0 and H1 persistence landscape L1 amplitudes, export their time series, and generate a persistence diagram for the target date.

Correlation distance is defined as:

$$d_{ij}=\sqrt{2(1-\rho_{ij})}$$

Here, $\rho_{ij}$ is the Pearson correlation between asset returns within a window. H0 describes connected components, while H1 describes loops. Their amplitudes summarize topology and do not directly measure financial risk. The network plot shows edges with distances below `epsilon`, representing the complex's one-dimensional skeleton. PCA is used only for visualization and does not enter the persistent homology calculation.

## Repository structure

```text
TDA/
├── main_tda_pipeline.py    # Main pipeline and command-line interface
├── Test_main.py            # Legacy entry point; delegates to the main pipeline
├── Tool/
│   ├── helper.py           # Symbols, normality tests, and representative selection
│   └── tda_utils.py        # Data preparation, topology, plotting, and exports
├── poster/
│   └── TDA.pdf             # Project poster
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

See the [project poster](poster/TDA.pdf). Despite its filename, `Test_main.py` is a compatibility entry point, not a test suite.

## Installation and quick start

Create an isolated environment with Python 3.10. Run the following commands from the project root.

**Windows PowerShell:**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main_tda_pipeline.py --no-show
```

**macOS / Linux:**

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main_tda_pipeline.py --no-show
```

Dependency installation and price downloads require internet access. Omit `--no-show` to open chart windows during execution; close each window to continue. Charts are saved in either mode.

To use a custom asset pool and skip the Wikipedia lookup:

```powershell
.\.venv\Scripts\python.exe main_tda_pipeline.py --symbols SPY QQQ IWM GLD TLT --anchors 5 --window 60 --start 2019-01-01 --end 2021-12-31 --target-date 2020-03-16 --no-show
```

Run `python main_tda_pipeline.py --help` with the environment active to list all options.

| Option | Default | Description |
| --- | --- | --- |
| `--start` / `--end` | `2019-01-01` / `2021-12-31` | Download date range; the end date is exclusive |
| `--target-date` | `2020-03-16` | Snapshot date; uses the latest available observation on or before this date |
| `--pool-size` | `80` | First N symbols in the current constituent list, not a market-cap ranking |
| `--anchors` | `20` | Maximum number of representative assets; must be at least 2 |
| `--window` | `60` | Observations per rolling window; must be at least 3 |
| `--epsilon` | `0.8` | Network edge threshold; does not limit the persistent homology filtration |
| `--symbols` | Unset | Space-separated symbols that replace the default asset pool |
| `--output-dir` | `tda_outputs` | Output directory |
| `--no-show` | Disabled | Save charts without opening windows |

## Outputs

Results are written to `tda_outputs/` by default. Repeated runs overwrite files with the same names; use different `--output-dir` values to preserve separate experiments.

| File | Contents |
| --- | --- |
| `returns.csv` | Percentage log returns for selected assets |
| `normality.csv` | Skewness, Pearson kurtosis, and normality test p-values |
| `topology_timeseries.csv` | `H0_Norm` and `H1_Norm`, indexed by window end date |
| `empirical_distribution.png` | Return distribution and Q-Q plot for the first selected asset |
| `asset_cloud.png` | Three-dimensional PCA projection; skipped with fewer than 3 assets |
| `correlation_matrix.png` | Pearson correlation matrix |
| `distance_matrix.png` | Correlation distance matrix |
| `market_graph.png` | Asset network at the specified distance threshold |
| `homology_timeseries.png` | H0 and H1 amplitude time series with applicable event annotations |
| `persistence_diagram_YYYYMMDD.pdf` | Birth-death persistence diagram for the target window, rather than a barcode plot |

The `Is Normal?` column uses a 0.05 threshold for the Jarque-Bera test. `Yes` means the test did not reject normality; it does not establish that returns are normally distributed. Assets with fewer than 30 valid observations are omitted from this table.

## Method limitations and reproducibility

- The default asset pool uses **current constituents at execution time**. Applying it to historical periods introduces survivorship bias. List ordering, provider revisions, and download availability can also change results.
- Representatives are selected using the entire analysis period, making this an exploratory, retrospective analysis. Prediction or backtesting would require selection using training data alone.
- Stocks and cryptocurrencies follow different trading calendars. The pipeline removes weekends but does not fully align exchange calendars. Forward-filling holiday prices can introduce zero returns and affect correlations.
- Missing-data handling can remove assets or observations. Windows count retained rows and do not necessarily represent consecutive exchange trading days.
- Constant asset returns within a window, insufficient observations, or download failures can stop analysis. Check the asset pool and date range. The target date must have enough preceding observations.
- Landscape amplitudes use giotto-tda's default discretization and number of landscape layers, fitted separately for each window. They are not exact integrals on a shared fixed grid. Keep parameters and asset pools consistent when comparing experiments.
- `requirements.txt` pins giotto-tda and scikit-learn, while other dependencies use version ranges. It is not a complete dependency lock file.

## Future work

Potential extensions include historical constituent data, exchange calendar alignment, a shared landscape sampling grid, price caching, and out-of-sample event detection and backtesting.
