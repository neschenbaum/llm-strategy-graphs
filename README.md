# Strategy graphs of LLM agents

Pilot that maps the full strategy graph of LLM pricing agents in a repeated pricing game.

The game is the logit pricing game of Calvano et al. (2020), restricted to three prices (the Bertrand price, the monopoly price, and their midpoint), as in the three-price game of Eschenbaum and Meylahn (2026).


## Files

* `game.py`: three-price game, strategy graphs, graph metrics, and the deviation test
* `pilot.py`: queries the agents through the Anthropic API and stores the answers in `samples.jsonl`
* `analyze.py`: builds the strategy graphs from the answers and prints the results

