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

    # Ogni arco ha N costi
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
    # Configura il logging (crea file in ./logs/)
    setup_logging()

    N = 2   # numero di obiettivi

    V, predecessors, successors, edge_cost = build_instance_2cost()

    logger.info("=" * 60)
    logger.info("GRAFO:")
    logger.info(f"Nodi: {V}")
    logger.info(f"Archi e costi: {edge_cost}")
    logger.info("=" * 60)

    # Esegui l'algoritmo
    L = mda(V, 0, predecessors, successors, edge_cost)

    logger.info("=" * 60)
    logger.info("RISULTATO FINALE - Soluzioni Pareto-ottimali:")
    logger.info("=" * 60)

    for v in sorted(L):
        costs = [lab.cost for lab in L[v]]
        logger.info(f"Nodo {v}: {costs}")

    # Stampa anche su console in formato leggibile
    print("\n" + "=" * 60)
    print("RISULTATO FINALE:")
    print("=" * 60)
    for v in sorted(L):
        print(f"Nodo {v}:")
        for lab in L[v]:
            print(f"  costo = {lab.cost}")
        print()
