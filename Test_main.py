import pandas as pd
import numpy as np
from Tool.helper import select_topological_anchors, get_sp500_symbols, batch_normality_test
from Tool.tda_utils import TDAFinancialEngine

if __name__ == "__main__":
    # --- Step 1: Define an expansive pool with exogenous assets ---
    print("Step 1: Initializing Broad Market Pool...")
    base_pool = get_sp500_symbols(80)  # Take the first 80 stocks as the pool
    # Incorporate safe-haven and non-related assets to expand the characteristic space
    exogenous_assets = ['GLD', 'TLT', 'VIXY', 'BTC-USD']
    full_pool = list(set(base_pool + exogenous_assets))

    START_DATE = '2019-01-01'
    END_DATE = '2021-12-31'

    engine = TDAFinancialEngine(window_size=60)


    print("Fetching raw data...")
    raw_returns = engine.prepare_returns(full_pool, START_DATE, END_DATE)

    valid_thresh = int(raw_returns.shape[0] * 0.95)
    raw_returns = raw_returns.dropna(axis=1, thresh=valid_thresh)

    raw_returns = raw_returns.ffill().bfill()

    raw_returns = raw_returns.dropna(axis=0)

    print(f"Data cleaned. Remaining shape: {raw_returns.shape} (Days, Assets)")

    # --- Step 2: Manifold Learning / Topological Anchor Selection ---
    print("\nStep 2: Selecting 20 Topological Anchors via Hierarchical Clustering...")
    optimal_symbols = select_topological_anchors(raw_returns, num_anchors=20)

    returns = raw_returns[optimal_symbols]

    # --- Step 3: Statistical Rigor (Normality Testing) ---
    print("\nStep 3: Executing Normality Tests on selected Anchors...")
    normality_table = batch_normality_test(returns)
    print(normality_table)

    engine.plot_empirical_distribution(returns, asset_name=optimal_symbols[0])

    # --- Step 4: Topological Data Analysis Visualization ---
    crash_date = '2020-03-16'  # 核心考察日：COVID 熔断日
    print(f"\nStep 4: Executing TDA Visualizations for crash date ({crash_date})...")

    engine.plot_asset_cloud_3D(returns, crash_date)
    engine.plot_market_topology_separated(returns, crash_date, epsilon=0.80)

    # --- Step 4.5: Generate Persistence Barcode for the crash date ---
    print(f"\nStep 4.5: Generating Persistence Barcode for {crash_date}...")

    target_dt = pd.to_datetime(crash_date)
    if target_dt not in returns.index:
        closest_idx = returns.index.get_indexer([target_dt], method='nearest')[0]
        actual_date = returns.index[closest_idx].strftime('%Y-%m-%d')
        print(f"   ⚠️ {crash_date} 非交易日，已自动对齐到 {actual_date}")
        crash_date = actual_date
        target_idx = closest_idx
    else:
        target_idx = returns.index.get_loc(target_dt)

    start_idx = max(0, target_idx - engine.window_size + 1)
    window_data = returns.iloc[start_idx: target_idx + 1]

    corr_matrix = window_data.corr().fillna(0)
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))

    np.fill_diagonal(dist_matrix.values, 0)

    diagrams = engine.generate_persistence_barcode(dist_matrix, crash_date)
    print("   ✅ Barcode generation successful!")

    # --- Step 5: Timeseries Computation ---
    print("\nStep 5: Computing the full L1 Norm time series with Macro Events...")
    topo_ts = engine.compute_topology_timeseries(returns)

    macro_shocks = {
        '2020-02-20': 'COVID-19 Initial Selloff',
        '2020-03-16': 'Black Monday (VIX Record High)',
        '2020-11-09': 'Pfizer Vaccine Efficacy Announced',
        '2021-01-27': 'GameStop Short Squeeze',
        '2021-05-19': 'Crypto Crash (China Ban/Tesla)'
    }

    engine.plot_homology_timeseries(topo_ts, events=macro_shocks)

    print("\nPipeline execution complete! All structural analyses are finished.")