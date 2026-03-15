import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats  # <-- 新增：用于正态性检验
import requests
import io
from Tool.tda_utils import TDAFinancialEngine


# =============================================================================
# Helper Functions
# =============================================================================
def get_sp500_symbols(limit=20):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        table = pd.read_html(io.StringIO(response.text))
        df = table[0]
        symbols = df['Symbol'].head(limit).str.replace('.', '-', regex=False).tolist()
        return symbols
    else:
        print(f"Failed to fetch data, Status Code: {response.status_code}")
        return []


def batch_normality_test(returns_df):
    """批量检验所有资产的正态性并返回汇总表格"""
    results = []
    for column in returns_df.columns:
        data = returns_df[column].dropna().values
        if len(data) < 30:
            continue

        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=False)
        stat_sw, p_sw = stats.shapiro(data)
        stat_jb, p_jb = stats.jarque_bera(data)

        is_normal = "Yes" if p_jb > 0.05 else "No (Reject)"
        results.append({
            'Asset': column,
            'Skewness': round(skew, 2),
            'Kurtosis': round(kurt, 2),
            'J-B p-value': f"{p_jb:.2e}",
            'S-W p-value': f"{p_sw:.2e}",
            'Is Normal?': is_normal
        })

    results_df = pd.DataFrame(results)
    results_df.set_index('Asset', inplace=True)
    return results_df


def test_normality_visual(returns_series, asset_name="Asset"):
    """绘制单只股票的直方图和 Q-Q 图"""
    data = returns_series.dropna().values
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=False)
    stat_jb, p_jb = stats.jarque_bera(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Histogram
    sns.histplot(data, bins=50, kde=False, stat="density", ax=ax1, color='lightskyblue')
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(data), np.std(data))
    ax1.plot(x, p, 'k', linewidth=2, label='Normal Fit')
    ax1.set_title(f"A. Histogram of {asset_name} Returns")
    ax1.legend()

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title(f"B. Q-Q Plot of {asset_name} Returns")
    ax2.get_lines()[0].set_markerfacecolor('coral')
    ax2.get_lines()[0].set_markeredgecolor('k')
    ax2.get_lines()[0].set_alpha(0.6)

    results_text = (
        f"--- Statistical Tests ---\n"
        f"Skewness: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f} (Normal=3)\n\n"
        f"Jarque-Bera p-value: {p_jb:.2e}\n"
        f"(p < 0.05 rejects Normality)"
    )
    ax2.text(0.05, 0.95, results_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


# =============================================================================
# 1. Configuration & Initialization
# =============================================================================
stock_symbols = get_sp500_symbols(30)
START_DATE = '2016-01-01'
END_DATE = '2021-12-31'

engine = TDAFinancialEngine(window_size=60)

# =============================================================================
# 2. Data Preparation
# =============================================================================
print("Fetching historical data and calculating log-returns...")
returns = engine.prepare_returns(stock_symbols, START_DATE, END_DATE)

# =============================================================================
# 2.5 Data Distribution & Normality Testing (For Academic Justification)
# =============================================================================
print("\nRunning Normality Tests on Asset Returns...")
normality_table = batch_normality_test(returns)
print("\n--- Normality Test Results (Highlighting Fat Tails) ---")
print(normality_table.to_string())

# 提取第一只股票（例如 AAPL）画出 Q-Q 图供报告使用
first_asset = returns.columns[0]
print(f"\nGenerating Normality visual report for representative asset: {first_asset}")
test_normality_visual(returns[first_asset], asset_name=first_asset)

# =============================================================================
# 3. TDA Execution
# =============================================================================
print("\nMethod: Computing TDA based on Correlation Distance (Ultrametric)...")
results_corr = engine.compute_correlation_tda(returns)

engine.plot_asset_cloud_3D(returns, '2021-05-18')
engine.plot_market_topology(returns, '2021-05-18', epsilon=0.85)
engine.plot_persistence_diagram(returns, '2021-05-18')

topo_timeseries = engine.compute_topology_timeseries(returns)
engine.plot_homology_timeseries(topo_timeseries)

# =============================================================================
# 4. Visualization (Replicating Paper Figure 3 Logic)
# =============================================================================
plt.figure(figsize=(14, 7))

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
    if event_date in results_corr.index or (
            event_date >= results_corr.index.min() and event_date <= results_corr.index.max()):
        plt.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
        plt.text(event_date, plt.gca().get_ylim()[1] * 0.95, label,
                 color='red', rotation=90, verticalalignment='top',
                 horizontalalignment='right', fontweight='bold', fontsize=10)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print("\nReplication analysis complete. Single graph generated.")

# =============================================================================
# 6. Summary Statistics
# =============================================================================
table_5_replica = engine.calculate_summary_statistics(results_corr)
print("\n--- Table 5 Replication: Summary Statistics ---")
print(table_5_replica.to_string(index=False))