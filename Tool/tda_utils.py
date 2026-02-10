import pandas as pd
import numpy as np
import yfinance as yf
from gtda.time_series import SlidingWindow
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
import scipy.stats as stats
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class TDAFinancialEngine:
    """
    Enhanced TDA Tool Engine for Financial Time Series.
    Supports Log-returns and Landscape Norms.
    """

    def __init__(self, window_size=60, stride=1):
        self.window_size = window_size
        self.stride = stride
        # Initialize sliding window transformer
        self.sw = SlidingWindow(size=window_size, stride=stride)

    def prepare_returns(self, symbols, start, end):
        """Step 1: Fetch data and calculate Log-Returns as per image_366cc3.png"""
        data = yf.download([f"{s}-USD" for s in symbols], start=start, end=end)['Close']
        data = data.ffill().dropna()
        # Logarithmic Returns calculation
        returns = (np.log(data) - np.log(data.shift(1))).dropna() * 100
        return returns

    def compute_correlation_tda(self, returns_df):
        """
        Step-by-step implementation of Correlation-based TDA:
        1. Calculate Correlation Matrix (C)
        2. Transform to Distance Matrix (D)
        3. Compute Persistence on Distance Matrices
        """
        # Create sliding windows of return data
        X_windows = self.sw.fit_transform(returns_df.values)

        # 1. Compute Pearson Correlation Matrix for each window
        # X_windows shape is (n_windows, 60, n_assets)
        corr_matrices = []
        for win in X_windows:
            c = pd.DataFrame(win).corr().fillna(0).values
            np.fill_diagonal(c, 1.0)
            corr_matrices.append(c)

        corr_matrices = np.array(corr_matrices)

        # 2. Map Correlation to Distance: D = sqrt(2 * (1 - C))
        # This ensures D=0 for perfect correlation and D=2 for anti-correlation
        dist_matrices = np.sqrt(2 * (1 - np.clip(corr_matrices, -1, 1)))

        # 3. Compute Persistence using 'precomputed' metric because input is a distance matrix
        VR = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=[0, 1],
            n_jobs=-1
        )
        diagrams = VR.fit_transform(dist_matrices)

        # 4. Extract Landscape Norms
        l1_amp = Amplitude(metric='landscape', metric_params={'p': 1}, order=1.0)
        l1_scores = l1_amp.fit_transform(diagrams)

        # Robust indexing for H1 (Dimension 1)
        h1_col = 1 if l1_scores.shape[1] > 1 else 0

        dates = returns_df.index[self.window_size - 1:]
        return pd.DataFrame({'L1_Corr_Distance': l1_scores[:, h1_col]}, index=dates)

    def calculate_summary_statistics(self, results_df):
        """
        Replicate Table 5: Summary statistics for persistence norms.
        Calculates Mean, SD, Skewness, and Kurtosis.
        """
        stats_list = []

        for col in results_df.columns:
            series = results_df[col].dropna()

            # Calculate moments
            res = {
                'Metric': col,
                'Mean': np.mean(series),
                'Std. Dev': np.std(series),
                'Skewness': stats.skew(series),
                'Kurtosis': stats.kurtosis(series),  # Excess Kurtosis (Fisher’s definition)
                'Min': np.min(series),
                'Max': np.max(series)
            }
            stats_list.append(res)

        return pd.DataFrame(stats_list)


    def plot_market_topology(self, returns_df, target_date, epsilon=0.8):
        """
        Visualizes the step-by-step transformation for a specific date.
        """
        # 1. Data
        target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
        window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]

        # 2. Calculate correlation matrix and distance matrix
        corr_matrix = window_data.corr()
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.fillna(0)))

        # --- Set figure ---
        fig = plt.figure(figsize=(20, 6))

        # Figure A: Correlation and Its Implications for Risk
        ax1 = fig.add_subplot(131)
        sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0, ax=ax1,
                    cbar_kws={'label': 'Correlation Strength\n(Red=High Risk/Convergence)'})
        ax1.set_title(r"A. Correlation Matrix $C_{ij}^\tau(t)$" + f"\nDate: {target_date}")

        # Figure B: Distance and Its Topological Meaning
        ax2 = fig.add_subplot(132)
        sns.heatmap(dist_matrix, annot=False, cmap='viridis_r', ax=ax2,
                    cbar_kws={'label': 'Metric Distance $D_{ij}$\n(Yellow=Close/Simplex Connects)'})
        ax2.set_title(r"B. Ultrametric Distance $D = \sqrt{2(1-C)}$")

        # Figure C: Geometric Connection of Simple Complexes
        ax3 = fig.add_subplot(133)
        G = nx.from_pandas_adjacency((dist_matrix < epsilon).astype(int))
        G.remove_edges_from(nx.selfloop_edges(G))

        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightcoral',
                node_size=600, font_size=9, ax=ax3, edge_color='gray', alpha=0.5)

        # Add text explanation
        explanation = (
            f"Topological Meaning:\n"
            f"1. Edges exist if $D_{{ij}} < {epsilon}$\n"
            f"2. Complex density relates to $L_1$ Norm\n"
            f"3. High cluster density = Market Instability"
        )
        ax3.text(1.05, 0.1, explanation, transform=ax3.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                 fontsize=10, verticalalignment='bottom')
        ax3.set_title(f"C. Simplicial Complex ($\epsilon$ = {epsilon})")

        plt.tight_layout()
        plt.show()

    def plot_asset_cloud_2D(self, returns_df, target_date):
        """
        Visualizes the 10 assets as points in 60-dimensional space via PCA.
        This illustrates the 'Asset as a Point' duality.
        """
        try:
            target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
            # Transpose to get (Assets, Time) -> (10, 60)
            window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1].T
        except KeyError:
            print(f"Error: Date {target_date} not found.")
            return

        # Project 60D assets to 2D for visualization
        pca = PCA(n_components=2)
        assets_2d = pca.fit_transform(window_data.values)

        plt.figure(figsize=(10, 8))
        plt.scatter(assets_2d[:, 0], assets_2d[:, 1], s=100, color='coral', edgecolors='black')

        # Add labels for each asset
        for i, asset in enumerate(window_data.index):
            plt.annotate(asset, (assets_2d[i, 0], assets_2d[i, 1]), xytext=(5, 5), textcoords='offset points')

        plt.title(f"10 Assets in 60D Feature Space (PCA)\nTarget Date: {target_date}")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    def plot_asset_cloud_3D(self, returns_df, target_date):
        """
        Visualizes the 10 assets as points in 60-dimensional space via 3D PCA.
        Consistent with the '10 points in R^60' duality.
        """
        try:
            target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
            # Transpose: (Time, Assets) -> (Assets, Time) to treat assets as 60D points
            window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1].T
        except KeyError:
            print(f"Error: Date {target_date} not found.")
            return

        # 1. Project 60D assets to 3D space
        pca = PCA(n_components=3)
        assets_3d = pca.fit_transform(window_data.values)
        explained_var = np.sum(pca.explained_variance_ratio_) * 100

        # 2. Setup 3D Plot
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # 3. Scatter Plot
        # Using 'coral' for nodes as per your previous preference
        scatter = ax.scatter(assets_3d[:, 0], assets_3d[:, 1], assets_3d[:, 2],
                             s=150, color='coral', edgecolors='black', alpha=0.8)

        # 4. Label each asset point (the '10 names')
        for i, asset in enumerate(window_data.index):
            ax.text(assets_3d[i, 0], assets_3d[i, 1], assets_3d[i, 2], asset,
                    fontsize=10, fontweight='bold')

        # 5. Formatting with Mathematical Context
        ax.set_title(
            f"3D Asset Cloud in Feature Space (PCA)\nTarget Date: {target_date}\nTotal Variance Explained: {explained_var:.2f}%")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")

        # Use standard white grid for clarity
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

