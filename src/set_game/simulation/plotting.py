import logging
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    datefmt='%H:%M:%S',
    format="",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')


class SETStatsPlotter:
    """Card game SET statistics plotter."""

    def __init__(self, statistics_folder_path: Path, output_folder_path: Path) -> None:
        self.statistics_path: Path = statistics_folder_path
        self.output_folder_path: Path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

    def _plot_state_stats(self) -> None:
        """Plots game state statistics for states where at least 12 cards are dealt."""
        df = pl.read_csv(self.statistics_path / 'state_stats.csv')
        for kind, ylabel, title in zip(
                ['num_sets_mean', 'prob_no_set'],
                ['mean of # SETs in dealt', 'probability of no SET in dealt'],
                ['Average number of SETs', 'Cap SET probability']
        ):
            hover_data: dict[str, bool | str] = {'num_occurrences': ':,'}
            if kind == 'num_sets_mean':
                hover_data = {'num_sets_var': True, 'num_occurrences': ':,'}

            fig = px.line(
                df.filter(pl.col('dealt') >= 12),
                x='in_deck',
                y=kind,
                color='dealt',
                hover_data=hover_data,
                markers=True,
                labels={
                    'in_deck': '# cards in deck',
                    kind: ylabel,
                    'dealt': '# dealt cards',
                    'num_occurrences': '# occurrences',
                    'num_sets_var': 'variance of # SETs in dealt',
                },
                title=title,
            )
            fig.update_xaxes(autorange='reversed')
            fig.update_yaxes(minor=dict(tickmode='auto', nticks=5, showgrid=True))
            fig.update_layout(
                font=dict(
                    family="Lato, sans-serif",
                    weight=100,
                    color='black',
                ),
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(69, -3, -3)),
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.01,
                    xanchor='right',
                    x=1,
                ),
                margin=dict(l=0, r=0, b=40, t=60),
            )
            fig.update_traces(
                marker=dict(
                    size=8,
                    line=dict(
                        width=1,
                    )
                )
            )

            fig.write_html(self.output_folder_path / f'{kind}.html', full_html=False)

    def _plot_prob_num_remain_cards(self) -> None:
        """
        Plots probability distribution of number of cards remaining at the end of the game.

        The same distribution corresponds to the total number of SETs found during the game.
        """
        df = pl.read_csv(self.statistics_path / 'prob_num_remain_cards.csv')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(27, 20, -1)),
            y=df['prob'],
            showlegend=False,
            hovertemplate='probability: %{y:.5f}<extra></extra>',
        ))
        fig.add_trace(go.Bar(
            x=list(range(18, -3, -3)),
            y=7 * [0],
            showlegend=False,
            xaxis="x2",
        ))
        fig.update_layout(
            font=dict(
                family="Lato, sans-serif",
                weight=100,
                color='black',
            ),
            margin=dict(l=0, r=0, t=50, b=50),
            hovermode='closest',
            yaxis=dict(
                domain=[0.15, 1.0],
                title=dict(
                    text='probability',
                )
            ),
            xaxis=dict(
                title=dict(
                    text='total # collected SETs during the game',
                ),
            ),
            xaxis2=dict(
                title=dict(
                    text='# remaining cards at the end of the game',
                ),
                anchor="free",
                overlaying="x",
                side="bottom",
                position=0.,
                tickmode='array',
                tickvals=list(range(18, -3, -3)),
                autorange='reversed',
            ),
            title='Distribution of total number of SETs collected',
        )
        fig.write_html(self.output_folder_path / 'num_found_sets.html', full_html=False)

    def _plot_prob_max_dealt(self) -> None:
        """Plots probability distribution of maximum number of cards dealt during the game."""
        df = pl.read_csv(self.statistics_path / 'prob_max_dealt.csv')
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(12, 24, 3)),
            y=df['prob'],
            showlegend=False,
            hovertemplate='probability: %{y:.5f}<extra></extra>',
        ))
        fig.update_layout(
            font=dict(
                family="Lato, sans-serif",
                weight=100,
                color='black',
            ),
            margin=dict(l=0, r=0, t=50, b=60),
            hovermode='closest',
            yaxis=dict(
                title=dict(
                    text='probability',
                )
            ),
            xaxis=dict(
                title=dict(
                    text='maximum # dealt cards during the game',
                ),
                tickmode='array',
                tickvals=list(range(12, 24, 3)),
            ),
            title='Distribution of maximum number of dealt cards',
        )
        fig.write_html(self.output_folder_path / 'max_num_dealt.html', full_html=False)

    def plot(self) -> None:
        """Plots all SET game statistics."""
        logger.info('Plotting statistics.')
        self._plot_state_stats()
        self._plot_prob_num_remain_cards()
        self._plot_prob_max_dealt()
