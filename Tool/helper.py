import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import requests
import io

def get_sp500_symbols(limit=50):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    if response.status_code == 200:
        table = pd.read_html(io.StringIO(response.text))
        df = table[0]
        symbols = df['Symbol'].head(limit).str.replace('.', '-', regex=False).tolist()
        return symbols
    return []


def batch_normality_test(returns_df):
    """Batch verify the normality of assets and return the summary table"""
    results = []
    for column in returns_df.columns:
        data = returns_df[column].dropna().values
        if len(data) < 30: continue
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
    df = pd.DataFrame(results, columns=['Asset', 'Skewness', 'Kurtosis', 'J-B p-value', 'S-W p-value', 'Is Normal?']).set_index('Asset')
    return df


def select_topological_anchors(returns_df, num_anchors=20):
    """Select one medoid per Ward cluster (at most num_anchors assets)."""
    if num_anchors < 1:
        raise ValueError('num_anchors must be positive')
    clean_returns = returns_df.dropna(axis=0, how='any')
    clean_returns = clean_returns.loc[:, clean_returns.std() > 0]
    if len(clean_returns) < 3 or clean_returns.shape[1] < 2:
        raise ValueError('Anchor selection requires at least 3 complete rows and 2 nonconstant assets')
    if num_anchors >= clean_returns.shape[1]:
        return clean_returns.columns.tolist()
    corr_matrix = clean_returns.corr()
    dist_matrix = np.sqrt(2 * (1 - np.clip(corr_matrix, -1.0, 1.0)))

    condensed_dist = squareform(dist_matrix, checks=False)
    Z = linkage(condensed_dist, method='ward')
    cluster_labels = fcluster(Z, t=num_anchors, criterion='maxclust')

    selected_assets = []
    asset_names = clean_returns.columns
    for i in np.unique(cluster_labels):
        cluster_assets = asset_names[cluster_labels == i]
        if len(cluster_assets) == 1:
            selected_assets.append(cluster_assets[0])
            continue
        sub_dist = dist_matrix.loc[cluster_assets, cluster_assets]
        medoid = sub_dist.mean(axis=1).idxmin()
        selected_assets.append(medoid)
    return selected_assets
