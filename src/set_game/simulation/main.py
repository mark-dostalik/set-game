from pathlib import Path

import click

from set_game.simulation.plotting import SETStatsPlotter
from set_game.simulation.simulator import SETSimulator


@click.command()
@click.argument('n', type=int)
@click.option('--stats-folder-path', type=Path, default=None, help='Output folder for calculated statistics.')
@click.option('--plots-folder-path', type=Path, default=None, help='Output folder for statistics plots.')
def main(n: int, stats_folder_path: Path | None, plots_folder_path: Path | None) -> None:
    """Runs SET simulator for given number of games."""
    stats_folder_path = stats_folder_path or Path(f'output/statistics/{n:_}')
    plots_folder_path = plots_folder_path or Path(f'output/plots/{n:_}')

    simulator = SETSimulator()
    simulator.run(n)
    simulator.compute_statistics()
    simulator.save_statistics(stats_folder_path)

    plotter = SETStatsPlotter(stats_folder_path, plots_folder_path)
    plotter.plot()


if __name__ == '__main__':
    main()
