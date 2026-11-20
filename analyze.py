#-----------------------------------------------------------
# This is the analysis of the pilot. It reconstructs the
# strategy graph of each pair of agents from the answers
# stored by pilot.py.
#
# For each agent (model and harness), the modal price in
# each state is its frozen strategy; ties go to the lower
# price. States are written as own price and rival price,
# e.g. ML = own M, rival L. Per pair, the analysis reports
# * outcome: the cycle that play from the opening reaches,
#   its collusion index (0 at Nash, 1 at monopoly), and the
#   long-run index of stochastic play with the sampled
#   frequencies
# * deviation test, for each firm: the gain from the best
#   deviation from the cycle (per period, in % of the firm's
#   profit on the cycle; 0 means the cycle is sustained),
#   the price of the rival in the period after a one-period
#   deviation (path of play -> after the deviation), and the
#   states after the deviation, e.g. ML>LL>MM when play
#   returns through a price war, or ML>LM>ML (loop) when it
#   never returns
# * strategy graph: all attractors with their basin sizes,
#   the bottleneck state (largest return-path betweenness),
#   and the metrics of the paper on the attractor with the
#   largest basin
# With --bootstrap N, the answers in every state are
# resampled N times. Scripted agents come first as controls.
#
# Usage
#   python analyze.py [samples.jsonl] [--why HM] [--bootstrap 300]
#-----------------------------------------------------------

import argparse, itertools, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from game import (analyzeBasin, deviationTest, evalPlay, findAttractors, graphMetrics,
                  historySpace, listControls, longRunIndex, maxBetweenness, meanIndex, priceNames,
                  scriptedStrategy, strategyGraph, swapObs, toLabel)
from pilot import samplesFile


#-----------------------------
# Fix variables for analysis
#-----------------------------

# Seed for the bootstrap
seedBootstrap = 0

# How many answers to print with --why
numberOfReasons = 5


#--------------------
# Import the answers

# counts of the answers per agent and history, the reasoning per agent and history,
# the tokens per model, and the number of queries without a valid answer
def importSamples(path):
    counts = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    reasons = defaultdict(lambda: defaultdict(list))
    tokens, invalid = defaultdict(lambda: [0, 0]), defaultdict(int)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        model, harness, k, h, _ = r["key"]
        agent = (model, harness, k)
        if r["level"] is None:
            invalid[agent] += 1
        else:
            counts[agent][h][r["level"]] += 1
            if r["why"]:
                reasons[agent][h].append(f"{priceNames[r['level']]}: {r['why']}")
        tokens[model] = [a + b for a, b in zip(tokens[model], r["tokens"])]
    return counts, reasons, tokens, invalid


#--------------------------
# Strategies of the agents

# modal price and action frequencies at every history; None if a history has no answer yet
# NOTE with rng, the answers in every history are first resampled with replacement
def modalStrategy(counts, k, rng=None):
    strategy, frequencies = {}, {}
    for h in historySpace(k):
        n = np.array(counts[toLabel(h)], float)
        if n.sum() == 0:
            return None
        if rng is not None:
            n = rng.multinomial(int(n.sum()), n / n.sum()).astype(float)
        strategy[h], frequencies[h] = int(np.argmax(n)), n / n.sum()
    return strategy, frequencies


def agentLabel(agent):
    model, harness, _ = agent
    return f"{model.removeprefix('claude-')}/{harness}"


def stateLabel(h):
    return priceNames[h[-1][0]] + priceNames[h[-1][1]]


#--------------------
# Analysis of a pair

# cycle from the opening, deviation test for both firms, and the strategy graph
def analyzePair(s1, s2, k):
    _, cycle = evalPlay(s1, s2, k)
    test1 = deviationTest(s1, s2, cycle, k)
    test2 = deviationTest(s2, s1, [swapObs(h) for h in cycle], k)
    G = strategyGraph(s1, s2, k)
    attractors = sorted(findAttractors(G), key=lambda a: -analyzeBasin(G, set(a))[0])
    largest = max(attractors, key=lambda a: (analyzeBasin(G, set(a))[0], meanIndex(a)))
    return dict(cycle=cycle, test1=test1, test2=test2, metrics=graphMetrics(G),
                bottleneck=maxBetweenness(G, set(largest))[1],
                attractors=[(">".join(stateLabel(h) for h in sorted(a)), analyzeBasin(G, set(a))[0])
                            for a in attractors])


#--------
# Output

def printPairs(title, pairs, k):
    rows = []
    for name, s1, p1, s2, p2 in pairs:
        rows.append((name, analyzePair(s1, s2, k), longRunIndex(p1, p2, k)))
    print(f"\n{title}: outcome and enforcement (firm 1, then firm 2)")
    print(f"{'pair (firm 1 | firm 2)':<30}{'cycle':<8}{'index':>6}{'long-run':>9}"
          + "".join(f"{'gain %':>8}  {'resp.':<6}{'after deviation':<18}" for _ in range(2)))
    for name, r, longRun in rows:
        cells = "".join(f"{gain:8.1f}  {response:<6}{after + (' (loop)' if back == 'never' else ''):<18}"
                        for gain, response, back, after in (r["test1"], r["test2"]))
        print(f"{name:<30}{'>'.join(stateLabel(h) for h in r['cycle']):<8}{meanIndex(r['cycle']):6.2f}"
              f"{longRun:9.2f}{cells}")
    print(f"\n{title}: strategy graph; markers on the attractor with the largest basin")
    print(f"{'pair (firm 1 | firm 2)':<30}{'attractors (basin size)':<26}{'bottleneck':>11}{'index':>7}"
          f"{'in-deg':>8}{'betw.':>7}{'path':>7}{'basin':>7}{'attr.':>6}")
    for name, r, _ in rows:
        m = r["metrics"]
        attractors = ", ".join(f"{a} ({n})" for a, n in r["attractors"])
        print(f"{name:<30}{attractors:<26}"
              f"{(stateLabel(r['bottleneck']) if r['bottleneck'] else '-'):>11}{m['index']:7.2f}"
              f"{m['inDegree']:8.2f}{m['maxBetweenness']:7.0f}{m['pathLength']:7.2f}"
              f"{m['basinFraction']:7.2f}{m['attractors']:6d}")


# share of resamples in which a pair reaches the same cycle as with all answers and sustains it;
# among resamples in which deviating once pays in the stage game for both firms, the share in
# which play returns to the cycle; and the most common values of the metrics
def printBootstrap(agents, counts, k, draws):
    rng = np.random.default_rng(seedBootstrap)
    resamples = {a: [modalStrategy(counts[a], k, rng)[0] for _ in range(draws)] for a in agents}
    strategies = {a: modalStrategy(counts[a], k)[0] for a in agents}
    print(f"\nBootstrap ({draws} resamples of the answers in every state): share of resamples in which")
    print("the pair reaches the same cycle as with all answers, and sustains it (no deviation pays for")
    print("either firm); among resamples in which a one-period deviation pays in the stage for both")
    print("firms, the share in which play returns to the cycle after either firm deviates; and the most")
    print("common values of (number of attractors, maximum betweenness, average path length)")
    print(f"{'pair (firm 1 | firm 2)':<30}{'same cycle':>11}{'sustained':>10}{'returns':>9}   markers")
    for x, y in itertools.combinations_with_replacement(agents, 2):
        reference = evalPlay(strategies[x], strategies[y], k)[1]
        same = sustained = deviations = back = 0
        markers = Counter()
        for d in range(draws):
            r = analyzePair(resamples[x][d], resamples[y][d], k)
            same += set(r["cycle"]) == set(reference)
            sustained += r["test1"][0] < 1e-6 and r["test2"][0] < 1e-6
            if r["test1"][2] != "-" and r["test2"][2] != "-":
                deviations += 1
                back += r["test1"][2] != "never" and r["test2"][2] != "never"
            m = r["metrics"]
            markers[(m["attractors"], int(m["maxBetweenness"]), round(m["pathLength"], 2))] += 1
        top = "; ".join(f"{a}, {b}, {p:.2f}: {n / draws:.0%}" for (a, b, p), n in markers.most_common(2))
        returns = f"{back / deviations:.0%}" if deviations else "-"
        print(f"{agentLabel(x) + ' | ' + agentLabel(y):<30}{same / draws:>11.0%}{sustained / draws:>10.0%}"
              f"{returns:>9}   {top}")


# modal price by last period's prices, with the share of answers that chose it
def printStrategy(agent, strategy, frequencies):
    shares = [frequencies[h][strategy[h]] for h in strategy]
    opening = strategy[()]
    print(f"\n{agentLabel(agent)}: modal price and its share by last period's prices "
          f"(mean share {np.mean(shares):.2f}); opening: {priceNames[opening]} {frequencies[()][opening]:.2f}")
    print("                rival L   rival M   rival H")
    for i in range(3):
        print(f"  own {priceNames[i]}      " + "".join(
            f"  {priceNames[strategy[((i, j),)]]} {frequencies[((i, j),)][strategy[((i, j),)]]:.2f}  "
            for j in range(3)))


#----------------
# Main run file
#----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", type=Path, default=samplesFile)
    ap.add_argument("--why", metavar="STATE", help="print the reasoning in this state, e.g. HM")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N", help="resamples (e.g. 300)")
    args = ap.parse_args()
    counts, reasons, tokens, invalid = importSamples(args.path)

    agents = {}
    for agent in sorted(counts):
        strategy = modalStrategy(counts[agent], agent[2])
        if strategy is None:
            print(f"{agentLabel(agent)}: some states have no valid answer yet; rerun pilot.py")
        else:
            agents[agent] = strategy
    for k in sorted({a[2] for a in agents}):
        print(f"\n===== memory {k}: {len(historySpace(k))} histories =====")
        controls = {c: {h: scriptedStrategy(c)(h) for h in historySpace(k)} for c in listControls}
        onehot = lambda s: {h: np.eye(3)[a] for h, a in s.items()}
        printPairs("Controls (scripted, symmetric pairs)",
                   [(f"{c} | {c}", controls[c], onehot(controls[c]), controls[c], onehot(controls[c]))
                    for c in listControls], k)
        group = [a for a in agents if a[2] == k]
        if k == 1:
            for a in group:
                printStrategy(a, *agents[a])
        printPairs("Agents", [(f"{agentLabel(x)} | {agentLabel(y)}", *agents[x], *agents[y])
                              for x, y in itertools.combinations_with_replacement(group, 2)], k)
        if args.bootstrap:
            printBootstrap(group, counts, k, args.bootstrap)

    if args.why:
        for agent in agents:
            texts = reasons[agent].get(args.why, [])
            print(f"\n{agentLabel(agent)} in state {args.why} ({len(texts)} answers):")
            for t in texts[:numberOfReasons]:
                print("  " + t.replace("\n", " ")[:300])
    print("\nTokens (input, output) by model: " + "; ".join(f"{m} {t[0]:,} / {t[1]:,}"
                                                           for m, t in tokens.items()))
    if invalid:
        print("Queries without a valid price: " + "; ".join(f"{agentLabel(a)} {n}"
                                                            for a, n in invalid.items()))


if __name__ == "__main__":
    main()
