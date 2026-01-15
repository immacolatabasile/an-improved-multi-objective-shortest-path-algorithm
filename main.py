from mda import mda, setup_logging, logger


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

    N = 2   # number of objectives

    V, predecessors, successors, edge_cost = build_instance_2cost()

    logger.info("=" * 60)
    logger.info("GRAPH DEFINITION")
    logger.info(f"Nodes: {V}")
    logger.info(f"Edges with cost vectors: {edge_cost}")
    logger.info("=" * 60)

    # Run the algorithm
    L = mda(V, 0, predecessors, successors, edge_cost)

    logger.info("=" * 60)
    logger.info("FINAL RESULT - Pareto-optimal solutions")
    logger.info("=" * 60)

    for v in sorted(L):
        costs = [lab.cost for lab in L[v]]
        logger.info(f"Node {v}: {costs}")

    # Also print results to console in a readable format
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    for v in sorted(L):
        print(f"Node {v}:")
        for lab in L[v]:
            print(f"  cost = {lab.cost}")
        print()
