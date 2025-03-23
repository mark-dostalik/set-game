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
        for kind, ylabel in zip(
                ['avg_num_sets', 'prob_no_set'],
                ['average # of sets in dealt', 'probability of no set in dealt'],
        ):
            fig = px.line(
                df.filter(pl.col('dealt') >= 12),
                x='in_deck',
                y=kind,
                color='dealt',
                hover_data=['num_occurrences'],
                markers=True,
                labels={
                    'in_deck': '# cards in deck',
                    kind: ylabel,
                    'dealt': '# cards in dealt',
                    'num_occurrences': '# occurrences',
                },
            )
            fig.update_xaxes(autorange='reversed')
            fig.update_yaxes(minor=dict(tickmode='auto', nticks=5, showgrid=True))
            fig.update_layout(
                font=dict(
                    family="Lato, sans-serif",
                    weight=100,
                ),
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(69, -3, -3)),
                ),
                legend=dict(
                    yanchor='top',
                    y=0.99,
                    xanchor='left',
                    x=0.01,
                ),
                margin=dict(l=20, r=20, t=20, b=20),
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

        The same distribution corresponds to the total number of sets found during the game.
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
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode='closest',
            yaxis=dict(
                domain=[0.1, 1.0],
                title=dict(
                    text='probability',
                )
            ),
            xaxis=dict(
                title=dict(
                    text='total # of collected sets during a game',
                ),
            ),
            xaxis2=dict(
                title=dict(
                    text='# of remaining cards at the end of the game',
                ),
                anchor="free",
                overlaying="x",
                side="bottom",
                position=0.,
                tickmode='array',
                tickvals=list(range(18, -3, -3)),
                autorange='reversed',
            ),
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
            ),
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode='closest',
            yaxis=dict(
                title=dict(
                    text='probability',
                )
            ),
            xaxis=dict(
                title=dict(
                    text='maximum # of dealt cards during a game',
                ),
                tickmode='array',
                tickvals=list(range(12, 24, 3)),
            ),
        )
        fig.write_html(self.output_folder_path / 'max_num_dealt.html', full_html=False)

    def plot(self) -> None:
        """Plots all SET game statistics."""
        logger.info('Plotting statistics.')
        self._plot_state_stats()
        self._plot_prob_num_remain_cards()
        self._plot_prob_max_dealt()
