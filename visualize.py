# visualize.py
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(V, edge_cost, pareto_paths=None):
    """
    pareto_paths: list of paths (each path is a list of nodes)
    """

    G = nx.DiGraph()

    for v in V:
        G.add_node(v)

    for (u, v), cost in edge_cost.items():
        G.add_edge(u, v, cost=cost)

    pos = nx.spring_layout(G, seed=42)

    # Base graph
    nx.draw(G, pos, with_labels=True, node_size=800)
    edge_labels = {
        (u, v): str(data["cost"])
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Plot Pareto paths
    if pareto_paths:
        colors = plt.cm.tab10.colors  # up to 10 distinct colors

        for i, path in enumerate(pareto_paths):
            path_edges = list(zip(path[:-1], path[1:]))

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                width=3,
                edge_color=[colors[i % len(colors)]],
                label=f"Path {i}"
            )

    plt.title("Pareto-optimal paths")
    plt.show()

def visualize_best_path_on_graph(V, edge_cost, highlight_path=None):
    """
    highlight_path: list of nodes, e.g. [0, 1, 3]
    """

    G = nx.DiGraph()

    for v in V:
        G.add_node(v)

    for (u, v), cost in edge_cost.items():
        G.add_edge(u, v, cost=cost)

    pos = nx.spring_layout(G, seed=42)

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_size=800)

    # Edge labels: cost vectors
    edge_labels = {
        (u, v): str(data["cost"])
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Highlight a specific path
    if highlight_path is not None:
        path_edges = list(zip(highlight_path[:-1], highlight_path[1:]))
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            width=3
        )

    plt.title("Multiobjective Graph")
    plt.show()
