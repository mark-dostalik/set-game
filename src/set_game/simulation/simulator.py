import itertools
import logging
import multiprocessing
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    datefmt='%H:%M:%S',
    format="",
    handlers=[RichHandler()],
)
logger = logging.getLogger('rich')


@dataclass
class StateRunningTotal:
    """Running totals for individual game states."""
    num_sets: int = 0
    no_set: int = 0
    occurrence: int = 0

    def __add__(self, other):
        if isinstance(other, StateRunningTotal):
            return StateRunningTotal(
                self.num_sets + other.num_sets,
                self.no_set + other.no_set,
                self.occurrence + other.occurrence
            )
        return NotImplemented


@dataclass
class StateStatistics:
    """Statistics for individual game states."""
    avg_num_sets: float = 0
    prob_no_set: float = 0
    occurrence: int = 0


@dataclass
class Statistics:
    """Complete game state statistics."""
    state_stats: defaultdict[int, dict[int, StateStatistics]] | None = None
    prob_num_remain_cards: dict[int, float] | None = None
    prob_max_dealt: dict[int, float] | None = None


RunningTotals = dict[int, dict[int, StateRunningTotal]]
Sets = dict[tuple[int, int, int], int]
Card = tuple[int, int, int, int]


class SETSimulator:
    """Card game SET Monte Carlo simulator."""

    def __init__(self) -> None:
        self.sets: Sets = dict()
        self._find_all_sets()
        self.running_totals: RunningTotals = self._init_running_totals()
        self.num_remain_cards: Counter = Counter()
        self.max_dealt: Counter = Counter()
        self.stats: Statistics = Statistics()

    @staticmethod
    def _encode(card: Card) -> int:
        """Encodes given card as a base 10 number."""
        return int(''.join(str(c) for c in card), 3)

    @staticmethod
    def _decode(n: int) -> Card:
        """Decodes given base 10 number to its base 3 representation in the form of a 4-tuple."""
        if n not in range(81):
            raise ValueError(f'Cannot decode {n} into a 4-feature SET card.')
        if n == 0:
            return 0, 0, 0, 0
        nums: list = []
        while n:
            n, r = divmod(n, 3)
            nums.append(r)
        nums.extend([0] * (4 - len(nums)))
        return tuple(reversed(nums))  # type: ignore

    @staticmethod
    def _init_running_totals() -> RunningTotals:
        """Initializes running total statistics for all possible game states."""
        running_totals: RunningTotals = {
            dealt: {
                in_deck: StateRunningTotal() for in_deck in range(81 - dealt, -3, -3)
            }
            for dealt in range(12, 24, 3)
        }
        for dealt in [3, 6, 9]:
            running_totals[dealt] = {0: StateRunningTotal()}

        return running_totals

    def _find_all_sets(self) -> None:
        """
        Finds all possible sets and assigns each a positive "similarity" score.

        If 0, 1, 2, or 3 feature(s) is (are) the same among the given triple of cards that constitute a set, the
        "similarity" score is 1, 2, 3, or 4, respectively. The "similarity" scores are later used as weights for
        randomly choosing sets from dealt cards (hence the positivity of the scores).
        """
        deck = itertools.product(range(3), repeat=4)
        for triple in itertools.combinations(deck, 3):
            weight: int = 1
            for i in range(4):
                if triple[0][i] == triple[1][i] == triple[2][i]:
                    weight += 1
                elif (triple[0][i] + triple[1][i] + triple[2][i]) % 3 != 0:
                    break
            else:
                self.sets[tuple(sorted(self._encode(card) for card in triple))] = weight  # type: ignore

    def _find_sets_in_dealt(self, dealt: set[int]) -> Sets:
        """Finds all sets in dealt cards along with their "similarity" scores."""
        sets_in_dealt: Sets = dict()
        for triple in itertools.combinations(sorted(dealt), 3):
            if triple in self.sets:
                sets_in_dealt[triple] = self.sets[triple]
        return sets_in_dealt

    def _simulate_game(
            self,
            running_totals: RunningTotals,
            num_remain_cards: Counter,
            max_dealt: Counter,
            seed: int | None = None,
    ) -> None:
        """
        Simulates a single game run.

        Probability of choosing a set from dealt cards is weighted by its "similarity" score.
        """
        random.seed(seed)
        deck: list = list(range(81))
        random.shuffle(deck)
        dealt: set = set(deck[:12])
        deck = deck[12:]
        curr_max_dealt: int = 12

        while dealt:
            sets_in_dealt = self._find_sets_in_dealt(dealt)
            curr_max_dealt = max(curr_max_dealt, len(dealt))

            running_totals[len(dealt)][len(deck)].num_sets += len(sets_in_dealt)
            running_totals[len(dealt)][len(deck)].occurrence += 1

            if not sets_in_dealt:
                running_totals[len(dealt)][len(deck)].no_set += 1
                if not deck:
                    break
                new_triple, deck = deck[:3], deck[3:]
                dealt |= set(new_triple)
            else:
                triple = random.choices(list(sets_in_dealt.keys()), weights=list(sets_in_dealt.values()), k=1)[0]
                dealt -= set(triple)
                if len(dealt) >= 12:
                    continue
                new_triple, deck = deck[:3], deck[3:]
                dealt |= set(new_triple)

        num_remain_cards.update([len(dealt)])
        max_dealt.update([curr_max_dealt])

    def _simulate_games(self, n: int) -> tuple[RunningTotals, Counter, Counter]:
        """Simulates given number of games sequentially."""
        running_totals = self._init_running_totals()
        num_remain_cards: Counter = Counter()
        max_dealt: Counter = Counter()
        for _ in range(n):
            self._simulate_game(running_totals, num_remain_cards, max_dealt)
        return running_totals, num_remain_cards, max_dealt

    def run(self, n: int):
        """Simulates given number of games in parallel."""
        num_proc = multiprocessing.cpu_count()
        m, r = divmod(n, num_proc)
        logger.info(f'Running {n:,d} game simulations in parallel on {num_proc} cores.')
        with multiprocessing.Pool(num_proc) as pool:
            ret = pool.map(self._simulate_games, (num_proc - 1) * [m] + [m + r])

        logger.info('Collecting statistics.')
        for running_totals, num_remain_cards, max_dealt in ret:
            self.num_remain_cards += num_remain_cards
            self.max_dealt += max_dealt
            for dealt in running_totals:
                for in_deck in running_totals[dealt]:
                    self.running_totals[dealt][in_deck] += running_totals[dealt][in_deck]

    def compute_statistics(self):
        """Computes statistics from running totals."""
        logger.info('Computing statistics.')
        self.stats.prob_num_remain_cards = {
            i: self.num_remain_cards.get(i, 0) / self.num_remain_cards.total() for i in range(0, 21, 3)
        }
        self.stats.prob_max_dealt = {i: self.max_dealt.get(i, 0) / self.max_dealt.total() for i in range(12, 24, 3)}
        self.stats.state_stats = defaultdict(dict)
        for dealt in self.running_totals.keys():
            for in_deck, running_total in self.running_totals[dealt].items():
                if running_total.occurrence > 0:
                    self.stats.state_stats[dealt][in_deck] = StateStatistics(
                        avg_num_sets=running_total.num_sets / running_total.occurrence,
                        prob_no_set=running_total.no_set / running_total.occurrence,
                        occurrence=running_total.occurrence,
                    )

    def save_statistics(self, output_folder_path: Path) -> None:
        """Dumps statistics to CSV files."""
        if (
                self.stats.state_stats is None
                or self.stats.prob_max_dealt is None
                or self.stats.prob_num_remain_cards is None
        ):
            raise RuntimeError('Statistics not yet computed.')

        logger.info('Saving statistics.')
        output_folder_path.mkdir(parents=True, exist_ok=True)

        pl.DataFrame(
            data=[
                [dealt, in_deck, case_stat.avg_num_sets, case_stat.prob_no_set, case_stat.occurrence]
                for dealt, values in self.stats.state_stats.items()
                for in_deck, case_stat in values.items()
            ],
            schema=[
                ('dealt', pl.UInt8),
                ('in_deck', pl.UInt8),
                ('avg_num_sets', pl.Float32),
                ('prob_no_set', pl.Float32),
                ('num_occurrences', pl.UInt64),
            ],
            orient='row',
        ).sort(['dealt', 'in_deck']).write_csv(output_folder_path / 'state_stats.csv')

        pl.DataFrame(
            data=[[k, v] for k, v in self.stats.prob_num_remain_cards.items()],
            schema=[('num_remain_cards', pl.UInt8), ('prob', pl.Float32)],
            orient='row',
        ).sort(['num_remain_cards']).write_csv(output_folder_path / 'prob_num_remain_cards.csv')

        pl.DataFrame(
            data=[[k, v] for k, v in self.stats.prob_max_dealt.items()],
            schema=[('max_dealt', pl.UInt8), ('prob', pl.Float32)],
            orient='row',
        ).sort(['max_dealt']).write_csv(output_folder_path / 'prob_max_dealt.csv')
