import pandas as pd
import numpy as np
import yfinance as yf
from gtda.time_series import SlidingWindow
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import Amplitude
import scipy.stats as stats


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

