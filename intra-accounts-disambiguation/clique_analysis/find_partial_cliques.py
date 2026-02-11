# prototype for an algorithm that finds all partial cliques in a graph

import networkx as nx
from itertools import combinations


def find_partial_cliques(G, min_size=3, max_size=6, tolerance=1):
    partial_cliques = []

    nodes = list(G.nodes)
    for k in range(min_size, max_size + 1):
        for subset in combinations(nodes, k):
            subgraph = G.subgraph(subset)
            num_edges = subgraph.number_of_edges()
            max_edges = k * (k - 1) // 2
            missing_edges = max_edges - num_edges

            if missing_edges <= tolerance:
                score = num_edges / max_edges
                partial_cliques.append((subset, score, missing_edges))

    partial_cliques.sort(key=lambda x: (-len(x[0]), -x[1], x[2]))  # prioritize larger cliques, higher scores, fewer missing
    return partial_cliques


G = nx.Graph()
G.add_edges_from([
    (1, 2), (1, 4), (1, 3), (2,3), (2,4), # missing (3,4) edge for completing clique 1,2,3,4
    (4, 5),  # Extra edge
    (5, 6), (5,7), (5,8), (6,7), (6,8), (7,8) # complete clique 5,6,7,8
])

cliques = find_partial_cliques(G)
for clique, score, missing in cliques:
    print(f"Clique: {clique}, Score: {score:.2f}, Missing Edges: {missing}")
