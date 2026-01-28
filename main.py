from mda import mda, setup_logging, logger
from visualize import visualize_graph_with_labels, visualize_label_graph
from config import mda_config

def reconstruct_path(label):
    path = []
    current = label

    while current is not None:
        path.append(current.node)
        current = current.label_pre

    return list(reversed(path))


def extract_pareto_paths(L, target):
    """
    Returns a list of paths, one for each Pareto-optimal label.
    """
    paths = []

    for lab in L[target]:
        path = reconstruct_path(lab)
        paths.append((path, lab.cost))

    return paths


def build_instance_2cost():
    V = [0, 1, 2, 3]

    edge_cost = {
        (0, 1): (1, 5),
        (0, 2): (4, 1),
        (1, 3): (1, 1),
        (1, 2): (1, 1),
        (2, 3): (4, 1)
    }

    predecessors = {v: [] for v in V}
    successors   = {v: [] for v in V}

    for (u, v) in edge_cost:
        successors[u].append(v)
        predecessors[v].append(u)

    return V, predecessors, successors, edge_cost


def build_demo_graph(N):
    V = list(range(10))

    edge_cost = {
        (0,1):(1,5,2)[:N], (0,2):(2,2,3)[:N],
        (1,3):(2,2,2)[:N], (1,4):(3,1,4)[:N],
        (2,3):(1,4,2)[:N], (2,4):(2,2,3)[:N],
        (3,5):(2,1,1)[:N], (4,5):(1,3,2)[:N],
        (5,6):(2,2,2)[:N],
        (6,7):(1,4,3)[:N], (6,8):(3,1,2)[:N],
        (7,9):(2,2,2)[:N], (8,9):(1,3,1)[:N]
    }

    predecessors = {v: [] for v in V}
    successors   = {v: [] for v in V}

    for (u,v) in edge_cost:
        successors[u].append(v)
        predecessors[v].append(u)

    return V, predecessors, successors, edge_cost


if __name__ == "__main__":
    # Configure logging (log files are created in ./logs/)
    setup_logging()

    # ============================
    # Configuration
    # ============================
    N = mda_config.num_objectives

    # Parallel dominance check option (from config)
    PARALLEL_DOMINANCE = mda_config.parallel_dominance

    # Build instance
    V, predecessors, successors, edge_cost = build_instance_2cost()
    #predecessors = Q

    logger.info("=" * 60)
    logger.info("GRAPH DEFINITION")
    logger.info(f"Nodes: {V}")
    logger.info(f"Edges with cost vectors: {edge_cost}")
    logger.info(f"Parallel dominance: {PARALLEL_DOMINANCE}")
    logger.info("=" * 60)

    # Run the algorithm
    L = mda(V, 0, predecessors, successors, edge_cost, parallel_dominance=PARALLEL_DOMINANCE)

    logger.debug("=" * 60)
    logger.debug("FINAL RESULT - Pareto-optimal solutions")
    logger.debug("=" * 60)

    for v in sorted(L):
        costs = [lab.cost for lab in L[v]]
        logger.debug(f"Node {v}: {costs}")

    # Also print results to console in a readable format
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    for v in sorted(L):
        print(f"Node {v}:")
        for lab in L[v]:
            print(f"  Label {lab.id}: cost={lab.cost}, pred={lab.label_pre.id if lab.label_pre else None}")
        print()

    target = 3

    # Get the labels for the target node
    target_labels = L[target]

    logger.debug(f"Pareto-optimal paths to node {target}:")
    for lab in target_labels:
        path = reconstruct_path(lab)
        logger.debug(f"Label {lab.id}: path={path}, cost={lab.cost}")

    print(f"Pareto-optimal paths to node {target}:")
    for lab in target_labels:
        path = reconstruct_path(lab)
        print(f"  Path (Label {lab.id}): {path} -> cost = {lab.cost}")

    # Label-aware visualization
    visualize_graph_with_labels(V, edge_cost, labels=target_labels)

    # Optional: visualize the label graph
    # visualize_label_graph(L, target=target)
