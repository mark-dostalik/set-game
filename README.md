Monte Carlo simulator of the card game [SET](https://en.wikipedia.org/wiki/Set_(card_game)). For more details check out
TODO.

To run the simulator on your own machine, clone this repository, `cd` into it, and run[^1]
```bash
uv run src/set_game/simulation/main.py <num-games>
```
where `<num-games>` is the number of games to be simulated. Statistics and interactive plots will then be available in
the `output` folder.

[^1]: Make sure you have `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/).
