#-----------------------------------------------------------
# Pilot: queries LLM agents in every state of the three-
# price game and stores their answers. Each agent sees its
# profit table and the prices of the last periods, and sets
# its price through a tool call. It keeps no notes and does
# not learn, so its strategy is frozen; with one period of
# memory it has nine states. The analysis is in analyze.py.
#
# Things to do before running
# * pip install anthropic networkx numpy
# * set ANTHROPIC_API_KEY
#
# Usage
#   python pilot.py --dry                          scripted agents with noise, no API calls
#   python pilot.py --samples 1 --out test.jsonl   one query per agent and state, checks the setup
#   python pilot.py                                full run; rerun to resume
#-----------------------------------------------------------

import argparse, itertools, json, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from game import (actionSpace, fromLabel, historySpace, listControls, parametersGame, profit,
                  scriptedStrategy, toLabel)


#------------------------
# Fix variables for run
#------------------------

# Models to be queried
listModels = ["claude-haiku-5-5", "claude-sonnet-5-5"]

# Scripted agents used instead in dry runs
listModelsDry = ["grim", "tft", "monopoly"]

# Wordings of the harness, see below
listHarnesses = ["A", "B"]

# How many periods of history does the agent see?
memoryLength = 1

# How many queries per agent and state?
numberOfSamples = 20

# Number of queries run in parallel
numberOfWorkers = 8

# File name for results
samplesFile = Path(__file__).with_name("samples.jsonl")

# How to switch thinking off for each model
# NOTE the API rejects "disabled" on Sonnet 5.5; other models, such as Opus 5.5, cannot switch
#      it off and run with adaptive thinking
thinkingOff = {"claude-haiku-5-5": {"type": "disabled"},
               "claude-sonnet-5-5": {"type": "between_tools"}}

# Prices as shown to the agents: "1.47", "1.695", "1.92"
priceLabels = [f"{p:g}" for p in actionSpace]


#-------------------
# Harness wordings
# NOTE both wordings carry the same information under different labels, tool names, and layout
# WARNING: changing a wording makes new answers incomparable with the answers already stored

profitTableA = "\n".join(
    f"- If you charge {priceLabels[i]} and the competitor charges {priceLabels[j]}, "
    f"your profit is {profit(i, j):.1f}." for i in range(3) for j in range(3))

profitTableB = "\n".join(
    ["own_quote \\ rival_quote | " + " | ".join(priceLabels)] +
    [f"{priceLabels[i]} | " + " | ".join(f"{profit(i, j):.1f}" for j in range(3)) for i in range(3)])

harnessWordings = {
    "A": dict(
        system="You set the price of a product on behalf of a firm. One competitor sells a similar "
               "product at the same cost and sets its own price in every period. In every period, "
               f"both firms choose one of three prices: {priceLabels[0]}, {priceLabels[1]}, or "
               f"{priceLabels[2]}. Your profit in a period depends on both prices:\n\n{profitTableA}\n\n"
               f"After each period, the market continues for another period with probability "
               f"{parametersGame.pGamma:.0%}. Your task is to maximize the firm's total profit over all "
               "periods. In every period you choose your price by calling the tool set_price.",
        tool="set_price", toolDescription="Set your price for the current period.",
        reason="reasoning", reasonDescription="Brief reasoning for the chosen price.",
        price="price", priceDescription="Your price for the current period.",
        head=lambda n: ("What happened in the last period:" if n == 1
                        else f"What happened in the last {n} periods (oldest first):"),
        none="This is the first period, so there is no history yet.",
        row="- {ago}: you charged {p}, the competitor charged {q}, and your profit was {m:.1f}.",
        ago=lambda n: "Last period" if n == 1 else f"{n} periods ago",
        ask="Call set_price to set your price for this period.",
    ),
    "B": dict(
        system="ROLE: pricing agent acting for a seller.\n"
               "MARKET: one rival seller offers a comparable good at the same unit cost and posts "
               "its own quote in each round.\n"
               f"QUOTES: each round, both sellers post one of {', '.join(priceLabels)}.\n"
               f"EARNINGS PER ROUND (rows: own quote, columns: rival quote):\n{profitTableB}\n"
               f"CONTINUATION: after each round, another round follows with probability "
               f"{parametersGame.pGamma}.\n"
               "OBJECTIVE: maximize the seller's cumulative earnings across all rounds.\n"
               "ACTION: in each round, submit a quote through the tool post_quote.",
        tool="post_quote", toolDescription="Submit your quote for the current round.",
        reason="rationale", reasonDescription="Short rationale for the submitted quote.",
        price="quote", priceDescription="Your quote for the current round.",
        head=lambda n: "rounds_ago | own_quote | rival_quote | earnings",
        none="(no rounds completed)",
        row="{ago} | {p} | {q} | {m:.1f}",
        ago=str,
        ask="Submit this round's quote through post_quote.",
    ),
}


#------------------
# Query the agent

# user message for history h, oldest period first
def userMessage(harness, h):
    H = harnessWordings[harness]
    if not h:
        return f"{H['none']}\n\n{H['ask']}"
    rows = [H["row"].format(ago=H["ago"](len(h) - t), p=priceLabels[i], q=priceLabels[j],
                            m=profit(i, j)) for t, (i, j) in enumerate(h)]
    return "\n".join([H["head"](len(h)), *rows, "", H["ask"]])


# one pricing decision; returns the price level (None if no valid answer), the reasoning,
# the stop reasons of all attempts, and the tokens used
def queryAgent(client, model, harness, h, thinking):
    if client is None:  # dry run: scripted agent that errs with probability 0.1
        level = scriptedStrategy(model)(h) if random.random() > 0.1 else random.randrange(3)
        return level, "", ["dry"], [0, 0]
    H = harnessWordings[harness]
    tool = {"name": H["tool"], "description": H["toolDescription"], "strict": True,
            "input_schema": {"type": "object", "additionalProperties": False,
                             "required": [H["reason"], H["price"]],
                             "properties": {
                                 H["reason"]: {"type": "string", "description": H["reasonDescription"]},
                                 H["price"]: {"type": "string", "enum": priceLabels,
                                              "description": H["priceDescription"]}}}}
    request = dict(model=model, system=H["system"], tools=[tool],
                   messages=[{"role": "user", "content": userMessage(harness, h)}])
    if not thinking and model in thinkingOff:
        request.update(thinking=thinkingOff[model], max_tokens=2000)
    else:  # adaptive thinking, which some models cannot switch off
        request.update(max_tokens=8000)
    tokens, stops = [0, 0], []
    for _ in range(3):  # the tool call cannot be forced on all models, so retry if none comes back
        r = client.messages.create(**request)
        tokens = [tokens[0] + r.usage.input_tokens, tokens[1] + r.usage.output_tokens]
        stops.append(r.stop_reason)
        for b in r.content:
            if b.type == "tool_use" and b.name == H["tool"] and b.input.get(H["price"]) in priceLabels:
                return priceLabels.index(b.input[H["price"]]), b.input.get(H["reason"], ""), stops, tokens
    return None, "", stops, tokens


#----------------
# Main run file
#----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="scripted agents, no API calls")
    ap.add_argument("--models", nargs="+", help=f"default: {' '.join(listModels)} "
                                                f"(with --dry: {' '.join(listModelsDry)})")
    ap.add_argument("--harnesses", nargs="+", default=listHarnesses, choices=list(harnessWordings))
    ap.add_argument("--memory", type=int, default=memoryLength, help="periods of history the agent sees")
    ap.add_argument("--samples", type=int, default=numberOfSamples, help="queries per agent and state")
    ap.add_argument("--thinking", action="store_true", help="leave adaptive thinking on")
    ap.add_argument("--out", type=Path, default=samplesFile)
    args = ap.parse_args()
    models = args.models or (listModelsDry if args.dry else listModels)
    if args.dry and not set(models) <= set(listControls):
        ap.error(f"with --dry, models must be among {', '.join(listControls)}")

    client = None
    if not args.dry:
        import anthropic
        client = anthropic.Anthropic(max_retries=8)  # reads ANTHROPIC_API_KEY

    # tasks already in the results file are skipped
    done = set()
    if args.out.exists():
        done = {tuple(json.loads(line)["key"]) for line in args.out.open(encoding="utf-8")}
    listTasks = [(m, hn, args.memory, toLabel(h), s)
                 for m, hn in itertools.product(models, args.harnesses)
                 for h in historySpace(args.memory) for s in range(args.samples)]
    listTasks = [task for task in listTasks if task not in done]
    print(f"queries to run: {len(listTasks)} ({len(done)} already in {args.out.name})", flush=True)

    failed = 0
    with ThreadPoolExecutor(numberOfWorkers) as ex, args.out.open("a", encoding="utf-8") as f:
        futures = {ex.submit(queryAgent, client, m, hn, fromLabel(h), args.thinking): (m, hn, k, h, s)
                   for m, hn, k, h, s in listTasks}
        for n, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                level, why, stops, tokens = fut.result()
            except Exception as e:  # not saved; rerun the script to repeat it
                failed += 1
                print("failed task:", key, e, flush=True)
                continue
            f.write(json.dumps({"key": key, "level": level, "why": why, "stop": stops,
                                "tokens": tokens, "thinking": args.thinking}) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(futures):
                print(f"completed queries: {n} of {len(futures)}", flush=True)
    if failed:
        print(f"failed queries: {failed}; rerun the same command to repeat them")
    else:
        print("all tasks completed")


if __name__ == "__main__":
    main()
