#-----------------------------------------------------------
# This file contains the three-price game, the strategy
# graphs, and the deviation test used for the pilot.
# The game is the pricing game of Calvano et al. (2020)
# with prices restricted to the Bertrand price, the
# monopoly price, and their midpoint, as in the three-price
# game of Eschenbaum and Meylahn (2026). A state is the
# history of the last k periods, each a pair of price
# levels (own, rival), oldest first.
#
# Check the graph metrics against the 101 equilibria of
# the paper with
#   python game.py --check path/to/theory_janusz
# NOTE the folder needs symmetricNE.csv, asymmetricNE.csv,
#      and theory_results.csv of the paper
#-----------------------------------------------------------

import argparse, csv, itertools, math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


#---------------------------------------------
# Define parameters needed to specify the game

@dataclass(frozen=True)
class Parameters:
    pA0: float
    pA1: float
    pMu: float
    pC1: float
    pGamma: float
    pBertrandPrice: float
    pMonopolyPrice: float
    pMarketSize: float


# Game parameters, Calvano baseline
# NOTE pGamma enters as the continuation probability shown to the agents
# NOTE pMarketSize only rescales the profits shown to the agents
parametersGame = Parameters(
    pA0=0,
    pA1=2,
    pMu=0.25,
    pC1=1,
    pGamma=0.95,
    pBertrandPrice=1.47,
    pMonopolyPrice=1.92,
    pMarketSize=100)


#-----------------------
# Game characteristics

# Action space: Bertrand price, midpoint, monopoly price
# NOTE prices are referred to by their level 0, 1, 2 throughout, printed as L, M, H
actionSpace = (parametersGame.pBertrandPrice, 1.695, parametersGame.pMonopolyPrice)
priceNames = "LMH"


# Demand follows the logit demand in Calvano et al. (2020)
# quantity sold at own price level i and rival price level j
def demand(i, j):
    par = parametersGame
    e = lambda x: math.exp((par.pA1 - actionSpace[x]) / par.pMu)
    return par.pMarketSize * e(i) / (e(i) + e(j) + math.exp(par.pA0 / par.pMu))


# Profit function
def profit(i, j):
    return (actionSpace[i] - parametersGame.pC1) * demand(i, j)


# profit of a firm when both charge the Bertrand price, and when both charge the monopoly price
profitNash, profitMonopoly = profit(0, 0), profit(2, 2)


# Collusion index of a pair of price levels: 0 at Nash, 1 at monopoly
# NOTE uses the mean profit of both firms, as in the paper
def collusionIndex(pair):
    i, j = pair
    return ((profit(i, j) + profit(j, i)) / 2 - profitNash) / (profitMonopoly - profitNash)


# best reply in the stage game to rival price level j
def bestReply(j):
    return max(range(3), key=lambda a: profit(a, j))


#---------------------------
# State space and updating

# all histories of length k; these are the nodes of the strategy graph
def stateSpace(k):
    return list(itertools.product(itertools.product(range(3), repeat=2), repeat=k))


# all histories of length 0 to k
# NOTE the shorter ones only occur in the first periods
def historySpace(k):
    return [h for n in range(k + 1) for h in stateSpace(n)]


# the same history from the point of view of the rival
def swapObs(h):
    return tuple((j, i) for i, j in h)


# append the prices of the current period and keep the last k periods
def nextState(h, a, b, k):
    return (h + ((a, b),))[-k:]


# history as text, own price first, e.g. 'HM,LL'; '' is the opening
def toLabel(h):
    return ",".join(priceNames[i] + priceNames[j] for i, j in h)


def fromLabel(s):
    return tuple((priceNames.index(p[0]), priceNames.index(p[1])) for p in s.split(",")) if s else ()


#-------------------------------------------------------
# Scripted strategies, used as controls and in dry runs
# NOTE they condition on the last period only

listControls = ("grim", "tft", "monopoly", "nash", "myopic")


# strategy as a function of the history, from the point of view of the agent
def scriptedStrategy(name):
    def act(h):
        if not h:
            return 0 if name in ("nash", "myopic") else 2
        own, rival = h[-1]
        return {"nash": 0, "monopoly": 2, "grim": 2 if (own, rival) == (2, 2) else 0,
                "tft": rival, "myopic": bestReply(rival)}[name]
    return act


#------------------------------------------------------------
# Graph metrics
# The following four functions follow analysis_utils.py in
# open-games-learning/python_port; the check at the end of
# this file verifies that they reproduce the paper.

# all attracting components (cycles and fixed points) of the graph
def findAttractors(G):
    return list(nx.attracting_components(G))


# attractor of the largest weakly connected component (largest basin), as in the paper
# NOTE ties in the basin size are broken by tieBreak, e.g. the collusion index
def largestWccAttractor(G, tieBreak=None):
    attractors = findAttractors(G)

    def key(a):
        size, _ = analyzeBasin(G, set(a))
        return (size, tieBreak(a)) if tieBreak is not None else size

    best = max(attractors, key=key)
    return set(best), len(attractors)


# basin of an attractor: number of nodes (inclusive), and average steps to reach the attractor
def analyzeBasin(G, attractorNodes):
    reverse = G.reverse()
    queue = deque([(n, 0) for n in attractorNodes])
    visited = set(attractorNodes)
    totalPathLength = 0
    while queue:
        node, dist = queue.popleft()
        totalPathLength += dist
        for neighbor in reverse.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    basinSize = len(visited)
    return basinSize, (totalPathLength / basinSize if basinSize > 0 else 0)


# largest return-path betweenness among transient nodes, and the node where it occurs
# b(v) counts the nodes whose path to the attractor passes through v (v included)
# NOTE paths are unique in a functional graph, so b(v) is the size of the in-tree of v
def maxBetweenness(G, attractorNodes):
    reverse = G.reverse()
    order = []
    visited = set(attractorNodes)
    queue = deque(list(attractorNodes))
    while queue:
        node = queue.popleft()
        for pred in reverse.neighbors(node):
            if pred not in visited:
                visited.add(pred)
                queue.append(pred)
                order.append(pred)
    transient = visited - set(attractorNodes)
    if not transient:
        return 0.0, None
    subtreeSize = {v: 1 for v in transient}
    for node in reversed(order):  # leaves first, accumulate towards the attractor
        succ = next(iter(G.neighbors(node)))  # unique out-edge
        if succ in subtreeSize:
            subtreeSize[succ] += subtreeSize[node]
    bottleneck = max(subtreeSize, key=subtreeSize.__getitem__)
    return subtreeSize[bottleneck], bottleneck


#---------------------------
# Strategy graph of a pair

# strategy graph of firm 1 playing s1 and firm 2 playing s2
# NOTE s1 and s2 map histories to price levels, each from the point of view of the firm
# TODO stochastic strategies induce a transition matrix, not a functional graph; for now
#      the modal price defines the graph
def strategyGraph(s1, s2, k):
    G = nx.DiGraph()
    for h in stateSpace(k):
        G.add_edge(h, nextState(h, s1[h], s2[swapObs(h)], k))
    return G


# mean collusion index over a set of states, e.g. an attractor
def meanIndex(nodes):
    return float(np.mean([collusionIndex(h[-1]) for h in nodes]))


# metrics of the paper, on the attractor with the largest basin (ties: higher index)
# TODO the metrics only see the largest basin; a punishment in another component, such as
#      alternating price cuts, is missed
def graphMetrics(G):
    attractor, numberOfAttractors = largestWccAttractor(G, tieBreak=meanIndex)
    basinSize, pathLength = analyzeBasin(G, attractor)
    return dict(index=meanIndex(attractor), attractors=numberOfAttractors,
                basinFraction=basinSize / G.number_of_nodes(),
                inDegree=float(np.mean([G.in_degree(n) for n in attractor])),
                maxBetweenness=maxBetweenness(G, attractor)[0], pathLength=pathLength)


# play from history h until a state repeats; returns the path and the cycle
def evalPlay(s1, s2, k, h=()):
    path = []
    while h not in path:
        path.append(h)
        h = nextState(h, s1[h], s2[swapObs(h)], k)
    return path, path[path.index(h):]


#-----------------
# Deviation test

# value for firm 1 of following s1 against s2, and of the best reply to s2, in every state
# NOTE exact: policy evaluation by a linear solve, policy iteration for the best reply
def continuationValues(s1, s2, k):
    delta = parametersGame.pGamma
    states = stateSpace(k)
    pos = {h: i for i, h in enumerate(states)}
    rival = [s2[swapObs(h)] for h in states]
    payoff = [[profit(a, rival[i]) for a in range(3)] for i in range(len(states))]
    succ = [[pos[nextState(h, a, rival[i], k)] for a in range(3)] for i, h in enumerate(states)]

    def evaluate(pol):
        P = np.zeros((len(states), len(states)))
        P[range(len(states)), [succ[i][a] for i, a in enumerate(pol)]] = 1
        return np.linalg.solve(np.eye(len(states)) - delta * P,
                               [payoff[i][a] for i, a in enumerate(pol)])

    pol = [s1[h] for h in states]
    vPolicy = v = evaluate(pol)
    while True:  # switch an action only on a strict improvement, so the loop ends
        q = [[payoff[i][a] + delta * v[succ[i][a]] for a in range(3)] for i in range(len(states))]
        new = [a if max(q[i]) <= q[i][a] + 1e-9 else int(np.argmax(q[i])) for i, a in enumerate(pol)]
        if new == pol:
            break
        pol, v = new, evaluate(new)
    return dict(zip(states, vPolicy)), dict(zip(states, v))


# Does a deviation from the cycle pay for firm 1, and how does the rival respond?
#   gain      best deviation from the cycle, per period, in % of firm 1's profit on the cycle;
#             0 means the cycle is sustained
#   response  price of the rival in the period after firm 1 deviates once to its best reply,
#             on the path of play -> after the deviation ('-' if equal)
#   back      periods until play returns to the cycle ('never' if it settles elsewhere)
#   after     states after the deviation, until play is back on the cycle or repeats
def deviationTest(s1, s2, cycle, k):
    vPolicy, vBest = continuationValues(s1, s2, k)
    rival = lambda h: s2[swapObs(h)]
    profitCycle = np.mean([profit(s1[h], rival(h)) for h in cycle])
    gain = max(vBest[h] - vPolicy[h] for h in cycle) * (1 - parametersGame.pGamma) / profitCycle * 100
    for h in cycle:  # first state on the cycle where deviating once pays in the stage game
        b, a = rival(h), bestReply(rival(h))
        if profit(a, b) > profit(s1[h], b) + 1e-9:
            break
    else:
        return gain, "none", "-", "-"
    on, off = nextState(h, s1[h], b, k), nextState(h, a, b, k)
    response = "-" if rival(on) == rival(off) else f"{priceNames[rival(on)]}->{priceNames[rival(off)]}"
    path, loop = evalPlay(s1, s2, k, off)
    t = next((t for t, x in enumerate(path) if x in cycle), None)
    shown = path[:t + 1] if t is not None else path + [loop[0]]
    after = ">".join(priceNames[x[-1][0]] + priceNames[x[-1][1]] for x in shown)
    return gain, response, ("never" if t is None else str(t)), after


# mean collusion index over stochastic play from the opening
# NOTE p1[h] and p2[h] are the action frequencies at each history, from the point of view of each firm
def longRunIndex(p1, p2, k, periods=1000):
    dist, total = {(): 1.0}, 0.0
    for _ in range(periods):
        new = defaultdict(float)
        for h, w in dist.items():
            for a, b in itertools.product(range(3), repeat=2):
                q = w * p1[h][a] * p2[swapObs(h)][b]
                if q > 1e-15:
                    new[nextState(h, a, b, k)] += q
        dist = new
        total += sum(w * collusionIndex(h[-1]) for h, w in dist.items())
    return total / periods


#-------------------------------------------
# Check against the equilibria of the paper

# recompute the metrics of the 101 equilibria and compare with theory_results.csv
# NOTE equilibria are stored as rows of successor states, with state index = 3 * own + rival + 1
def checkEquilibria(folder):
    folder = Path(folder)
    enc = {s: ((s - 1) // 3, (s - 1) % 3) for s in range(1, 10)}  # as ENC in validate_metrics.py
    rows = [[int(x) for x in line.split(",")]
            for name in ("symmetricNE", "asymmetricNE")
            for line in (folder / f"{name}.csv").read_text().split()]
    reference = list(csv.DictReader(open(folder / "theory_results.csv")))
    columns = dict(index="delta", attractors="num_attractors", basinFraction="basin_fraction",
                   inDegree="avg_indegree", maxBetweenness="max_betweenness",
                   pathLength="avg_path_length")
    worst = dict.fromkeys(columns, 0.0)
    for row, r in zip(rows, reference, strict=True):
        G = nx.DiGraph()
        for s in range(1, 10):
            G.add_edge((enc[s],), (enc[row[s - 1]],))
        for key, value in graphMetrics(G).items():
            worst[key] = max(worst[key], abs(value - float(r[columns[key]])))
    print(f"{len(rows)} equilibria; largest absolute difference from theory_results.csv:")
    for key, d in worst.items():
        print(f"  {key:<16}{d:.2e}")


#------------
# Main file

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", metavar="FOLDER", help="folder with the equilibrium files of the paper")
    args = ap.parse_args()
    print(f"Profit of the row firm (market size {parametersGame.pMarketSize:g}):")
    print("own\\rival " + "".join(f"{p:>8}" for p in actionSpace))
    for i in range(3):
        print(f"{actionSpace[i]:<10}" + "".join(f"{profit(i, j):8.1f}" for j in range(3)))
    print("Stage best replies: " + ", ".join(f"to {priceNames[j]}: {priceNames[bestReply(j)]}"
                                             for j in range(3)))
    if args.check:
        checkEquilibria(args.check)
