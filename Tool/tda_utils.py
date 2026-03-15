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
import ccxt
import time


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

    def prepare_returns(self, symbols, start, end, is_crypto=False):
        """Step 1: Fetch data and calculate Log-Returns as per image_366cc3.png"""
        if is_crypto:
            tickers = [f"{s}-USD" for s in symbols]
        else:
            tickers = symbols
        data = yf.download(tickers, start=start, end=end)['Close']
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
        Visualizes the step-by-step transformation for a specific date in separate plots.
        """
        # 1. Data preparation
        target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
        window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]

        # 2. Calculate correlation matrix and distance matrix
        corr_matrix = window_data.corr()
        dist_matrix = np.sqrt(2 * (1 - corr_matrix.fillna(0)))

        # ==========================================
        # Figure A: Correlation Matrix
        # ==========================================
        plt.figure(figsize=(8, 8))  # 独立的方形画布
        sns.heatmap(corr_matrix, annot=False, cmap='RdBu_r', center=0,
                    square=True, xticklabels=True, yticklabels=True,
                    # shrink 参数让右侧的颜色条高度与矩阵对齐，看起来更专业
                    cbar_kws={'label': 'Correlation Strength\n(Red=High Risk/Convergence)', 'shrink': 0.8})
        plt.title(r"A. Correlation Matrix $C_{ij}^\tau(t)$" + f"\nDate: {target_date}", pad=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # ==========================================
        # Figure B: Distance Matrix
        # ==========================================
        plt.figure(figsize=(8, 8))  # 独立的方形画布
        sns.heatmap(dist_matrix, annot=False, cmap='viridis_r',
                    square=True, xticklabels=True, yticklabels=True,
                    cbar_kws={'label': 'Metric Distance $D_{ij}$\n(Yellow=Close/Simplex Connects)', 'shrink': 0.8})
        plt.title(r"B. Ultrametric Distance $D = \sqrt{2(1-C)}$", pad=20)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # ==========================================
        # Figure C: Simplicial Complex
        # ==========================================
        plt.figure(figsize=(8, 8))  # 独立的方形画布
        G = nx.from_pandas_adjacency((dist_matrix < epsilon).astype(int))
        G.remove_edges_from(nx.selfloop_edges(G))

        # spring_layout 在方形画布中表现最好，能均匀展开节点
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightcoral',
                node_size=600, font_size=9, edge_color='gray', alpha=0.8)

        # 调整文本框位置，防止与节点重叠
        explanation = (
            f"Topological Meaning:\n"
            f"1. Edges exist if $D_{{ij}} < {epsilon}$\n"
            f"2. Complex density relates to $L_1$ Norm\n"
            f"3. High cluster density = Market Instability"
        )
        plt.text(1.05, 0.05, explanation, transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                 fontsize=10, verticalalignment='bottom')

        plt.title(f"C. Simplicial Complex ($\epsilon$ = {epsilon})", pad=20)
        # 使用 tight_layout 确保边缘不会被裁剪
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
            # 截取窗口：行是时间，列是资产
            window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]

            # --- 关键修复步骤 ---
            # 1. 剔除在这个窗口内包含任何 NaN 的资产（列）
            valid_window = window_data.dropna(axis=1, how='any')

            # 2. 转置，让 PCA 把“资产”当成要降维的对象 (n_samples = 资产数, n_features = 时间维度)
            X = valid_window.T.values

            if X.shape[1] == 0:
                print(f"Error: No valid features found for {target_date}. All data might be NaN.")
                return
            if X.shape[0] < 3:
                print(f"Error: Need at least 3 valid assets to plot 3D, found {X.shape[0]}.")
                return

            # 3. 执行 PCA
            pca = PCA(n_components=3)
            assets_3d = pca.fit_transform(X)

            # 4. 绘图 (保持不变...)
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(assets_3d[:, 0], assets_3d[:, 1], assets_3d[:, 2], s=100, color='coral', edgecolors='k')

            asset_names = valid_window.columns
            for i, name in enumerate(asset_names):
                ax.text(assets_3d[i, 0], assets_3d[i, 1], assets_3d[i, 2], name)

            ax.set_title(f"3D Asset Geometry ({len(asset_names)} Assets) - {target_date}")
            plt.show()

        except KeyError:
            print(f"Date {target_date} not found in index.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def plot_homology_timeseries(self, topo_df, title="Topological Dynamics (L1 Norm) Over Time"):
        """
        Plots the H0 and H1 L1 Norms over time using dual y-axes.
        """
        fig, ax1 = plt.subplots(figsize=(15, 6))

        # --- 绘制 H0 (红色, 左轴) ---
        color_h0 = 'tab:red'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('$H_0$ L1 Norm (Market Clustering)', color=color_h0)
        ax1.plot(topo_df.index, topo_df['H0_Norm'], color=color_h0, alpha=0.8, linewidth=1.5, label='$H_0$ Norm')
        ax1.tick_params(axis='y', labelcolor=color_h0)

        # --- 绘制 H1 (蓝色, 右轴) ---
        ax2 = ax1.twinx()
        color_h1 = 'tab:blue'
        ax2.set_ylabel('$H_1$ L1 Norm (Topological Divergence)', color=color_h1)
        ax2.plot(topo_df.index, topo_df['H1_Norm'], color=color_h1, alpha=0.8, linewidth=1.5, label='$H_1$ Norm')
        ax2.tick_params(axis='y', labelcolor=color_h1)

        # 如果你的数据跨越了 2020 年 3 月，可以加上这行高亮危机时刻
        # plt.axvspan(pd.to_datetime('2020-02-20'), pd.to_datetime('2020-03-25'), color='gray', alpha=0.2, label='COVID-19 Crash')

        # 图例排版
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.title(title)
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def compute_topology_timeseries(self, returns_df):
        """
        Computes both H0 and H1 L1 Norms over time using sliding windows.
        """
        X_windows = self.sw.fit_transform(returns_df.values)

        # 计算相关性矩阵和距离矩阵
        corr_matrices = np.array([pd.DataFrame(win).corr().fillna(0).values for win in X_windows])
        dist_matrices = np.sqrt(2 * (1 - np.clip(corr_matrices, -1, 1)))

        # 提取 0 维 (H0) 和 1 维 (H1) 拓扑特征
        VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1])
        diagrams = VR.fit_transform(dist_matrices)

        # 计算 Landscape 的 L1 范数
        l1_amp = Amplitude(metric='landscape', metric_params={'p': 1}, order=1.0)
        l1_scores = l1_amp.fit_transform(diagrams)

        # giotto-tda 默认按输入的 homology_dimensions 顺序返回列
        h0_norms = l1_scores[:, 0]
        h1_norms = l1_scores[:, 1] if l1_scores.shape[1] > 1 else np.zeros(len(l1_scores))

        # 对齐时间轴 (滑动窗口会吃掉前面的 window_size - 1 天)
        dates = returns_df.index[self.window_size - 1:]

        return pd.DataFrame({'H0_Norm': h0_norms, 'H1_Norm': h1_norms}, index=dates)

    def plot_persistence_diagram(self, returns_df, target_date):
        """
        Computes and plots the Persistence Diagram (Birth vs Death) for a specific date.
        Shows H0 (Connected Components) and H1 (1D Loops).
        """
        # 1. 提取特定窗口的数据并计算距离矩阵 D
        try:
            target_idx = returns_df.index.get_loc(pd.to_datetime(target_date))
            window_data = returns_df.iloc[target_idx - self.window_size + 1: target_idx + 1]
        except KeyError:
            print(f"Error: Date {target_date} not found.")
            return

        # 数据清洗与距离矩阵计算
        clean_df = window_data.dropna(axis=1, how='any')
        corr_matrix = clean_df.corr().fillna(0).values
        dist_matrix = np.sqrt(2 * (1 - np.clip(corr_matrix, -1, 1)))

        # 2. 计算持久同调 (Vietoris-Rips 过滤)
        # 注意：fit_transform 需要 3D 数组输入，所以我们用 [dist_matrix] 包裹一下
        VR = VietorisRipsPersistence(metric='precomputed', homology_dimensions=[0, 1])
        # 提取第一张（也是唯一一张）图表数据
        diagram = VR.fit_transform([dist_matrix])[0]

        # 3. 解析 H0 和 H1 特征
        # diagram 的每一行是 [Birth, Death, Dimension]
        h0_features = diagram[diagram[:, 2] == 0]
        h1_features = diagram[diagram[:, 2] == 1]

        # 4. 使用 Matplotlib 绘图
        plt.figure(figsize=(8, 8))

        # 画出 H0 (红点：连通分量)
        plt.scatter(h0_features[:, 0], h0_features[:, 1],
                    color='red', label='$H_0$ (Components)', alpha=0.7, s=60, edgecolors='k')

        # 画出 H1 (蓝三角：环路/空洞)
        if len(h1_features) > 0:
            plt.scatter(h1_features[:, 0], h1_features[:, 1],
                        color='blue', marker='^', label='$H_1$ (Loops)', alpha=0.7, s=80, edgecolors='k')

        # 绘制对角线 y = x (Birth = Death)
        # 距离的最大值在相关性网络中通常是 sqrt(4) = 2
        max_val = 2.05
        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Diagonal ($y=x$)')

        # 填充靠近对角线的“拓扑噪声”区域 (可选，为了数学美观)
        plt.fill_between([0, max_val], [0, max_val], [0.1, max_val + 0.1], color='gray', alpha=0.1)

        # 格式化
        plt.title(f"Persistence Diagram\nDate: {target_date} ({clean_df.shape[1]} Assets)", pad=20)
        plt.xlabel(r"Birth ($\epsilon$)")
        plt.ylabel(r"Death ($\epsilon$)")
        plt.xlim(-0.05, max_val)
        plt.ylim(-0.05, max_val)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


def batch_normality_test(returns_df):
    """
    对多只股票的收益率进行批量的正态性检验，生成学术报告表格。
    """
    results = []

    # 遍历 DataFrame 中的每一列（每一只股票）
    for column in returns_df.columns:
        data = returns_df[column].dropna().values

        # 如果有效数据太少，跳过
        if len(data) < 30:
            continue

        # 1. 计算偏度 (Skewness) 和 峰度 (Kurtosis)
        # 正态分布的 Skewness=0, Kurtosis=3
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=False)

        # 2. Shapiro-Wilk 检验 (适用于小样本 < 5000)
        stat_sw, p_sw = stats.shapiro(data)

        # 3. Jarque-Bera 检验 (金融最常用，专门测偏度和峰度)
        stat_jb, p_jb = stats.jarque_bera(data)

        # 4. 判断是否拒绝正态假设 (显著性水平 alpha = 0.05)
        # p < 0.05 意味着我们有 95% 的把握说它【不是】正态分布
        is_normal = "Yes" if p_jb > 0.05 else "No (Reject)"

        results.append({
            'Asset': column,
            'Skewness': round(skew, 2),
            'Kurtosis': round(kurt, 2),
            'J-B p-value': f"{p_jb:.2e}",
            'S-W p-value': f"{p_sw:.2e}",
            'Is Normal?': is_normal
        })

    # 转换为 DataFrame 方便展示
    results_df = pd.DataFrame(results)
    results_df.set_index('Asset', inplace=True)

    return results_df
