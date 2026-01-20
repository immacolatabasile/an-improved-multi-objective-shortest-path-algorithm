# visualize.py
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def reconstruct_path_from_label(label):
    """Reconstruct the path from a label by following predecessors."""
    path = []
    current = label
    while current is not None:
        path.append(current.node)
        current = current.pred
    return list(reversed(path))


def visualize_graph_with_labels(V, edge_cost, labels=None):
    """
    Label-aware visualization: colors paths by label.id and shows label info.

    labels: list of Label objects (with id, cost, pred attributes)
    """
    G = nx.DiGraph()

    for v in V:
        G.add_node(v)

    for (u, v), cost in edge_cost.items():
        G.add_edge(u, v, cost=cost)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(12, 8))

    # Base graph
    nx.draw(G, pos, with_labels=True, node_size=800,
            node_color='lightblue', font_size=10, font_weight='bold')
    edge_labels = {
        (u, v): str(data["cost"])
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # Plot Pareto paths colored by label.id
    legend_handles = []
    if labels:
        # Use a colormap that supports many colors
        cmap = plt.cm.get_cmap('tab20')

        for i, label in enumerate(labels):
            path = reconstruct_path_from_label(label)
            path_edges = list(zip(path[:-1], path[1:]))

            # Color based on label.id
            color = cmap(label.id % 20)

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=path_edges,
                width=3,
                edge_color=[color],
                alpha=0.8,
                style='solid',
                arrows=True,
                arrowsize=15,
                connectionstyle=f"arc3,rad={0.1 * (i % 3)}"  # offset overlapping paths
            )

            # Create legend entry: color = path, legend = cost vector
            legend_handles.append(
                mpatches.Patch(color=color, label=f"L{label.id}: {label.cost}")
            )

    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper left', fontsize=10,
                   title="Pareto paths (cost vectors)", title_fontsize=11)
    plt.title("Pareto-optimal paths (Label-aware)")
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.08)
    plt.show()


def visualize_label_graph(L, target=None):
    """
    Visualize the label graph: each label is a node, edges represent predecessor relationships.

    L: dict mapping node -> list of Label objects
    target: if specified, only show labels reachable from target labels
    """
    G = nx.DiGraph()

    # Collect all labels
    all_labels = []
    for node, labels in L.items():
        all_labels.extend(labels)

    # Add label nodes
    for label in all_labels:
        node_label = f"L{label.id}\n{label.node}\n{label.cost}"
        G.add_node(label.id, display=node_label, node=label.node, cost=label.cost)

    # Add edges from predecessor labels
    for label in all_labels:
        if label.pred is not None:
            G.add_edge(label.pred.id, label.id)

    # If target specified, filter to only reachable labels
    if target is not None:
        target_labels = L.get(target, [])
        reachable = set()
        for tl in target_labels:
            # Trace back from target
            current = tl
            while current is not None:
                reachable.add(current.id)
                current = current.pred
        G = G.subgraph(reachable).copy()

    plt.figure(figsize=(14, 10))

    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G, seed=42, k=2)

    # Color nodes by original graph node
    node_colors = []
    cmap = plt.cm.get_cmap('Set3')
    max_node = max((G.nodes[n].get('node', 0) for n in G.nodes()), default=0) + 1
    for n in G.nodes():
        node_val = G.nodes[n].get('node', 0)
        node_colors.append(cmap(node_val / max_node))

    # Draw the label graph
    nx.draw(G, pos,
            labels={n: G.nodes[n].get('display', str(n)) for n in G.nodes()},
            with_labels=True,
            node_size=2000,
            node_color=node_colors,
            font_size=8,
            arrows=True,
            arrowsize=20,
            edge_color='gray')

    plt.title("Label Graph (nodes = labels, edges = predecessor relationships)")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05)
    plt.show()


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
