import matplotlib.pyplot as plt
import pandas as pd
from Tool.tda_utils import TDAFinancialEngine

# =============================================================================
# 1. Configuration & Initialization
# =============================================================================
COINS = ['BTC', 'ETH', 'USDT', 'XRP', 'DOGE', 'LTC', 'XLM', 'XMR', 'DASH', 'XEM']
START_DATE = '2016-01-01'
END_DATE = '2021-12-31'

engine = TDAFinancialEngine(window_size=60)

# =============================================================================
# 2. Data Preparation
# =============================================================================
print("Fetching historical data and calculating log-returns...")
returns = engine.prepare_returns(COINS, START_DATE, END_DATE)

# =============================================================================
# 3. TDA Execution
# =============================================================================
print("Method: Computing TDA based on Correlation Distance (Ultrametric)...")
#  D = sqrt(2(1-C))
results_corr = engine.compute_correlation_tda(returns)

engine.plot_market_topology(returns, '2021-05-18', epsilon=0.85)
# =============================================================================
# 4. Visualization (Replicating Paper Figure 3 Logic)
# =============================================================================
plt.figure(figsize=(14, 7))

# L1 Norm curve
plt.plot(results_corr.index, results_corr['L1_Corr_Distance'],
         color='tab:green', linewidth=1.5, label='L1 Norm (Corr-Based)')


plt.title('Method: Ultrametric Distance Topology (Asset Correlation Scaffold)', fontsize=14)
plt.ylabel('L1 Norm Value', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# =============================================================================
# 5. Mark key exogenous shocks
# =============================================================================
shocks = {
    '2020-03-12': 'Covid-19 Shock',
    '2021-05-18': 'Elon Musk / Tesla Tweet'
}

for date, label in shocks.items():
    event_date = pd.to_datetime(date)
    # Check data
    if event_date in results_corr.index or (event_date >= results_corr.index.min() and event_date <= results_corr.index.max()):
        plt.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
        plt.text(event_date, plt.gca().get_ylim()[1] * 0.95, label,
                 color='red', rotation=90, verticalalignment='top',
                 horizontalalignment='right', fontweight='bold', fontsize=10)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print("Replication analysis complete. Single graph generated.")


table_5_replica = engine.calculate_summary_statistics(results_corr)
print("\n--- Table 5 Replication: Summary Statistics ---")
print(table_5_replica.to_string(index=False))