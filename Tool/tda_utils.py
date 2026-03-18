import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.decomposition import PCA
from ripser import ripser
import persim
import os
# Giotto-TDA import
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
from gtda.time_series import SlidingWindow

class TDAFinancialEngine:
    def __init__(self, window_size=60, max_homology_dim=1, output_dir='./tda_outputs'):
        self.window_size = window_size
        self.sw = SlidingWindow(size=self.window_size, stride=1)

        self.maxdim = max_homology_dim
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def prepare_returns(self, symbols, start_date, end_date):
        # Download data
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Close']

        # The yield rate calculated on Monday is based on Monday's price - Friday's price
        # From the close of trading on Friday to the close of trading on Monday, over these three full days, your paper profit or loss is a whole
        data = data.ffill()

        # When calculating the rate of return, only remove the blank row at the very beginning and retain the subsequent data
        returns = (np.log(data) - np.log(data.shift(1))).dropna(how='all') * 100
        return returns

    def plot_asset_cloud_3D(self, returns_df, target_date):
        try:
            target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
            window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]
            valid_window = window_data.dropna(axis=1, how='any')
            X = valid_window.T.values

            if X.shape[1] == 0 or X.shape[0] < 3:
                print("Skipping 3D Plot: Not enough valid data.")
                return


            pca = PCA(n_components=3)
            assets_3d = pca.fit_transform(X)

            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(assets_3d[:, 0], assets_3d[:, 1], assets_3d[:, 2], s=100, color='coral', edgecolors='k')

            for i, name in enumerate(valid_window.columns):
                ax.text(assets_3d[i, 0], assets_3d[i, 1], assets_3d[i, 2], name, fontsize=8)

            ax.set_title(f"3D Market Manifold (PCA) - {target_date}")

            plt.show()
        except KeyError:
            print(f"Date {target_date} not found.")

    def plot_market_topology_separated(self, returns_df, target_date, epsilon=0.85):
        target_dt = pd.to_datetime(target_date)

        # Fault-tolerance mechanism: Search for the most recent trading day
        if target_dt not in returns_df.index:
            closest_idx = returns_df.index.get_indexer([target_dt], method='nearest')[0]
            actual_date = returns_df.index[closest_idx]
            print(f"⚠️ Warning: Not found {target_date}，automatically align to the most recent trading day: {actual_date.strftime('%Y-%m-%d')}")
            target_idx = closest_idx
            target_date = actual_date.strftime('%Y-%m-%d')
        else:
            target_idx = returns_df.index.get_loc(target_dt)

        window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]

        # Calculate the correlation matrix and distance matrix
        corr_matrix = window_data.corr()
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.fillna(0)))

        # --- Figure 1：Correlation Matrix ---
        plt.figure(figsize=(8, 8))
        # Use the RdBu_r color system: positive correlation is red, negative correlation is blue, and no correlation is white
        sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={'label': 'Pearson Correlation $C_{ij}$', 'shrink': 0.8})
        plt.title(f"Correlation Matrix (Statistical Space) - {target_date}")
        plt.show()

        # --- Figure 2：Distance Matrix ---
        plt.figure(figsize=(8, 8))
        # Using the viridis_r color system: deep purple for close distances (strong correlation) and light yellow for far distances
        sns.heatmap(dist_matrix, annot=False, cmap='viridis_r', square=True,
                    cbar_kws={'label': 'Ultrametric Distance $D_{ij}$', 'shrink': 0.8})
        plt.title(f"Ultrametric Distance Matrix (Topological Space) - {target_date}")
        plt.show()

        # --- Figure 3：Simplicial Complex ---
        plt.figure(figsize=(8, 8))
        G = nx.from_pandas_adjacency((dist_matrix < epsilon).astype(int))
        G.remove_edges_from(nx.selfloop_edges(G))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=500, font_size=8, alpha=0.8)
        plt.title(f"Simplicial Complex ($\epsilon$ = {epsilon}) - {target_date}")
        plt.show()

    def compute_topology_timeseries(self, returns_df):
        print("Computing Persistent Homology over sliding windows (This may take a moment)...")
        # Ensure no NaN columns disrupt the sliding window
        clean_returns = returns_df.dropna(axis=1, how='any')
        X_windows = self.sw.fit_transform(clean_returns.values)

        l1_h0, l1_h1 = [], []
        VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1])
        l1_amp = Amplitude(metric='landscape', metric_params={'p': 1}, order=1.0)

        for win in X_windows:
            corr = pd.DataFrame(win).corr().fillna(0).values
            dist = np.sqrt(2 * (1 - np.clip(corr, -1, 1)))
            diagrams = VR.fit_transform([dist])
            scores = l1_amp.fit_transform(diagrams)[0]
            l1_h0.append(scores[0])
            l1_h1.append(scores[1] if len(scores) > 1 else 0.0)

        dates = returns_df.index[self.window_size - 1:]
        return pd.DataFrame({'H0_Norm': l1_h0, 'H1_Norm': l1_h1}, index=dates)

    def plot_homology_timeseries(self, topo_df, events=None):
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # Draw H0 (red line)
        ax1.plot(topo_df.index, topo_df['H0_Norm'], color='tab:red', alpha=0.8, label='$H_0$ Norm (Clustering)')
        ax1.set_ylabel('$H_0$ L1 Norm', color='tab:red')

        # Draw H1 (blue line)
        ax2 = ax1.twinx()
        ax2.plot(topo_df.index, topo_df['H1_Norm'], color='tab:blue', alpha=0.8, label='$H_1$ Norm (Holes)')
        ax2.set_ylabel('$H_1$ L1 Norm', color='tab:blue')

        plt.title("Topological Dynamics ($L_1$ Norm) vs. Macro Shocks")

        # --- Core: Dynamically add vertical lines and labels for major events ---
        if events:
            for date_str, label in events.items():
                event_date = pd.to_datetime(date_str)
                # Ensure that the event is within the range of the data timeline
                if event_date >= topo_df.index.min() and event_date <= topo_df.index.max():
                    # Draw a grey dotted line running from top to bottom
                    ax1.axvline(x=event_date, color='dimgrey', linestyle='--', linewidth=1.5, alpha=0.7)

                    # Add text descriptions at the top of the dotted line
                    # Dynamically obtain the upper limit of the Y-axis to prevent text from flying out of the chart
                    y_pos = ax1.get_ylim()[1] * 0.95
                    ax1.text(event_date, y_pos, f' {label}',
                             color='black', rotation=90, verticalalignment='top',
                             horizontalalignment='right', fontweight='bold', fontsize=10,
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

        # Merge the dual-axis legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        ax1.grid(True, alpha=0.3)
        plt.show()

    def plot_empirical_distribution(self, returns_df, asset_name=None):
        """
        Draw the empirical distribution histogram and Q-Q plot of the specified asset.
        """
        import scipy.stats as stats
        import matplotlib.gridspec as gridspec

        # If no asset is specified, the first asset will be drawn by default
        if asset_name is None:
            asset_name = returns_df.columns[0]

        if asset_name not in returns_df.columns:
            print(f"⚠️ 找不到资产 {asset_name}")
            return

        data_1d = returns_df[asset_name].dropna()

        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

        # --- Left figure: Histogram and fitted normal curve---
        ax0 = plt.subplot(gs[0])
        sns.histplot(data_1d, bins=50, kde=True, stat='density', ax=ax0, color='teal', alpha=0.6)
        xmin, xmax = ax0.get_xlim()
        x_pdf = np.linspace(xmin, xmax, 100)
        y_pdf = stats.norm.pdf(x_pdf, np.mean(data_1d), np.std(data_1d))
        ax0.plot(x_pdf, y_pdf, 'r--', lw=2, label='Normal Dist Fit')
        ax0.set_title(f"Empirical Distribution of {asset_name}")
        ax0.legend()

        # --- Right picture: Q-Q chart ---
        ax1 = plt.subplot(gs[1])
        stats.probplot(data_1d, dist="norm", plot=ax1)
        ax1.get_lines()[0].set_markerfacecolor('teal')
        ax1.get_lines()[0].set_markeredgecolor('teal')
        ax1.get_lines()[1].set_color('red')
        ax1.set_title("Q-Q Plot (Fat Tails Observation)")

        plt.show()

    def generate_persistence_barcode(self, dist_matrix, date_str):
        dist_array = np.array(dist_matrix)

        result = ripser(dist_array, distance_matrix=True, maxdim=self.maxdim)
        diagrams = result['dgms']

        plt.figure(figsize=(6, 6))
        title = f"Persistence Diagram - {date_str}"
        persim.plot_diagrams(diagrams, title=title)
        plt.tight_layout()

        safe_date = date_str.replace('-', '')
        filename = f'barcode_{safe_date}.pdf'
        save_path = os.path.join(self.output_dir, filename)

        # plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.close()

        print(f"[{date_str}] Barcode saved to: {save_path}")
        return diagrams