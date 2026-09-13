"""Command-line entry point for the financial TDA research pipeline."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

from Tool.helper import batch_normality_test, get_sp500_symbols, select_topological_anchors
from Tool.tda_utils import TDAFinancialEngine

MACRO_EVENTS = {
    '2020-02-20': 'COVID-19 Selloff',
    '2020-03-16': 'March 2020 Selloff',
    '2020-11-09': 'Vaccine Announcement',
    '2021-01-27': 'GameStop Short Squeeze',
    '2021-05-19': 'Crypto Selloff',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end', default='2021-12-31')
    parser.add_argument('--target-date', default='2020-03-16')
    parser.add_argument('--pool-size', type=int, default=80)
    parser.add_argument('--anchors', type=int, default=20)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--epsilon', type=float, default=0.8)
    parser.add_argument('--symbols', nargs='+', help='Override the default asset pool')
    parser.add_argument('--output-dir', default='tda_outputs')
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening windows')
    args = parser.parse_args()
    if args.window < 3 or args.anchors < 2 or args.pool_size < 1:
        parser.error('window >= 3, anchors >= 2 and pool-size >= 1 are required')
    if args.no_show:
        matplotlib.use('Agg')

    symbols = args.symbols or get_sp500_symbols(args.pool_size) + ['GLD', 'TLT', 'VIXY', 'BTC-USD']
    symbols = list(dict.fromkeys(symbols))
    engine = TDAFinancialEngine(window_size=args.window, output_dir=args.output_dir, show=not args.no_show)
    output = Path(args.output_dir)
    print(f'Downloading {len(symbols)} assets...')
    raw_returns = engine.prepare_returns(symbols, args.start, args.end)
    # Keep assets with at least 95% coverage, then use complete observations.
    raw_returns = raw_returns.replace([np.inf, -np.inf], np.nan)
    raw_returns = raw_returns.dropna(axis=1, thresh=int(np.ceil(len(raw_returns) * 0.95)))
    raw_returns = raw_returns.dropna(axis=0, how='any')
    anchors = select_topological_anchors(raw_returns, args.anchors)
    returns = raw_returns[anchors]
    if len(returns) < args.window:
        raise ValueError('Not enough complete observations for the requested window')
    print(f'Selected {len(anchors)} anchors: {", ".join(anchors)}')
    returns.to_csv(output / 'returns.csv', index_label='Date')
    normality = batch_normality_test(returns)
    normality.to_csv(output / 'normality.csv')
    print(normality)

    window, actual_date = engine.get_window(returns, args.target_date)
    engine.plot_empirical_distribution(returns)
    engine.plot_asset_cloud_3D(returns, actual_date)
    engine.plot_market_topology_separated(returns, actual_date, epsilon=args.epsilon)
    distance = engine.correlation_distance(window)
    engine.generate_persistence_barcode(distance, actual_date)
    topology = engine.compute_topology_timeseries(returns)
    topology.to_csv(output / 'topology_timeseries.csv', index_label='Date')
    engine.plot_homology_timeseries(topology, events=MACRO_EVENTS)
    print(f'Analysis complete. Results: {output.resolve()}')


if __name__ == '__main__':
    main()
