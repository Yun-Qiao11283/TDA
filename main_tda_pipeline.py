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

    # --- Step 2: Manifold Learning / Topological Anchor Selection ---
    print("\nStep 2: Selecting 20 Topological Anchors via Hierarchical Clustering...")
    optimal_symbols = select_topological_anchors(raw_returns, num_anchors=20)

    # Our current core dataset is clean and highly representative
    returns = raw_returns[optimal_symbols].dropna()

    # --- Step 3: Statistical Rigor (Normality Testing) ---
    print("\nStep 3: Executing Normality Tests on selected Anchors...")
    normality_table = batch_normality_test(returns)
    print(normality_table)
    engine.plot_empirical_distribution(returns, asset_name=None)

    # --- Step 4: Topological Data Analysis Visualization ---
    print("\nStep 4: Executing TDA Visualizations for crash date...")
    crash_date = '2020-03-16'  # COVID circuit breaker day, '2021-05-18'

    engine.plot_asset_cloud_3D(returns, crash_date)
    engine.plot_market_topology_separated(returns, crash_date, epsilon=0.80)

    # --- Step 5: Timeseries Computation ---
    print("\nStep 5: Computing the full L1 Norm time series with Macro Events...")
    topo_ts = engine.compute_topology_timeseries(returns)

    # Define the macro/financial major events marked on the timeline
    macro_shocks = {
        '2020-02-20': 'COVID-19 Initial Selloff',  # The first wave of selling during the epidemic has begun
        '2020-03-16': 'Black Monday (VIX Record High)',  # On Black Monday, the US stock market experienced a circuit breaker
        '2020-11-09': 'Pfizer Vaccine Efficacy Announced',  # Pfizer's vaccine is good news, and the market style has switched
        '2021-01-27': 'GameStop Short Squeeze',  # GME retail investor short squeeze incident (Local liquidity shock)
        '2021-05-19': 'Crypto Crash (China Ban/Tesla)'  # 5.19 The cryptocurrency market plunged and technology stocks pulled back
    }

    engine.plot_homology_timeseries(topo_ts, events=macro_shocks)

    print("\nPipeline execution complete! All structural analyses are finished.")
