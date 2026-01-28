"""
Test script to compare sequential vs parallel MDA.

Compares:
- Execution time
- Solution correctness (both must produce identical Pareto sets)
"""

import time
from mda import mda, setup_logging, logger, Label
from config import test_config


def build_instance_2cost():
    """Small test instance with 2 objectives."""
    V = [0, 1, 2, 3]

    edge_cost = {
        (0, 1): (1, 5),
        (0, 2): (4, 1),
        (1, 3): (1, 1),
        (1, 2): (1, 1),
        (2, 3): (4, 1)
    }

    predecessors = {v: [] for v in V}
    successors = {v: [] for v in V}

    for (u, v) in edge_cost:
        successors[u].append(v)
        predecessors[v].append(u)

    return V, predecessors, successors, edge_cost


def build_larger_instance(num_nodes=20, num_objectives=3):
    """
    Build a larger instance for performance testing.
    Creates a layered graph with multiple paths.
    """
    V = list(range(num_nodes))

    edge_cost = {}
    predecessors = {v: [] for v in V}
    successors = {v: [] for v in V}

    # Create edges: each node connects to next 2-3 nodes
    import random
    random.seed(test_config.random_seed)  # Reproducibility

    for u in range(num_nodes - 1):
        # Connect to next 1-3 nodes
        max_target = min(u + 4, num_nodes)
        for v in range(u + 1, max_target):
            # Random cost vector
            cost = tuple(random.randint(1, 10) for _ in range(num_objectives))
            edge_cost[(u, v)] = cost
            successors[u].append(v)
            predecessors[v].append(u)

    return V, predecessors, successors, edge_cost


def build_grid_instance(rows=10, cols=10, num_objectives=2):
    """
    Build a grid graph that generates many Pareto-optimal paths.
    Each cell (i,j) connects to (i+1,j) and (i,j+1).
    This structure tends to create exponentially many Pareto-optimal paths.
    """
    import random
    random.seed(test_config.random_seed)

    V = []
    node_id = {}

    # Create nodes
    idx = 0
    for i in range(rows):
        for j in range(cols):
            V.append(idx)
            node_id[(i, j)] = idx
            idx += 1

    edge_cost = {}
    predecessors = {v: [] for v in V}
    successors = {v: [] for v in V}

    # Create edges
    for i in range(rows):
        for j in range(cols):
            u = node_id[(i, j)]

            # Edge to right (i, j+1)
            if j + 1 < cols:
                v = node_id[(i, j + 1)]
                # Costs designed to create trade-offs
                cost = tuple(random.randint(1, 5) for _ in range(num_objectives))
                edge_cost[(u, v)] = cost
                successors[u].append(v)
                predecessors[v].append(u)

            # Edge to down (i+1, j)
            if i + 1 < rows:
                v = node_id[(i + 1, j)]
                cost = tuple(random.randint(1, 5) for _ in range(num_objectives))
                edge_cost[(u, v)] = cost
                successors[u].append(v)
                predecessors[v].append(u)

            # Diagonal edge (i+1, j+1) - adds more paths
            if i + 1 < rows and j + 1 < cols:
                v = node_id[(i + 1, j + 1)]
                cost = tuple(random.randint(1, 5) for _ in range(num_objectives))
                edge_cost[(u, v)] = cost
                successors[u].append(v)
                predecessors[v].append(u)

    return V, predecessors, successors, edge_cost


def build_conflicting_objectives_instance(num_layers=8, nodes_per_layer=5, num_objectives=2):
    """
    Build an instance with strongly conflicting objectives.
    This creates many Pareto-optimal solutions because improving one
    objective worsens another.
    """
    import random
    random.seed(test_config.random_seed)

    V = [0]  # Source
    node_id = {(0, 0): 0}
    idx = 1

    # Create layers
    for layer in range(1, num_layers + 1):
        for n in range(nodes_per_layer):
            V.append(idx)
            node_id[(layer, n)] = idx
            idx += 1

    # Add sink
    sink = idx
    V.append(sink)

    edge_cost = {}
    predecessors = {v: [] for v in V}
    successors = {v: [] for v in V}

    # Connect source to first layer
    for n in range(nodes_per_layer):
        v = node_id[(1, n)]
        # Conflicting costs: high on obj1 means low on obj2
        costs = []
        for obj in range(num_objectives):
            if obj % 2 == 0:
                costs.append(n + 1)  # Increasing
            else:
                costs.append(nodes_per_layer - n)  # Decreasing
        edge_cost[(0, v)] = tuple(costs)
        successors[0].append(v)
        predecessors[v].append(0)

    # Connect consecutive layers
    for layer in range(1, num_layers):
        for n1 in range(nodes_per_layer):
            u = node_id[(layer, n1)]
            for n2 in range(nodes_per_layer):
                v = node_id[(layer + 1, n2)]
                # Conflicting costs
                costs = []
                for obj in range(num_objectives):
                    base = random.randint(1, 3)
                    if obj % 2 == 0:
                        costs.append(base + abs(n1 - n2))
                    else:
                        costs.append(base + nodes_per_layer - abs(n1 - n2))
                edge_cost[(u, v)] = tuple(costs)
                successors[u].append(v)
                predecessors[v].append(u)

    # Connect last layer to sink
    for n in range(nodes_per_layer):
        u = node_id[(num_layers, n)]
        costs = tuple(random.randint(1, 3) for _ in range(num_objectives))
        edge_cost[(u, sink)] = costs
        successors[u].append(sink)
        predecessors[sink].append(u)

    return V, predecessors, successors, edge_cost


def extract_solution_signature(L):
    """
    Extract a comparable signature from the solution.
    Returns a dict: node -> sorted list of cost tuples
    """
    signature = {}
    for node, labels in L.items():
        costs = sorted([lab.cost for lab in labels])
        signature[node] = costs
    return signature


def compare_solutions(L_seq, L_par):
    """
    Compare two MDA solutions.
    Returns True if identical, False otherwise.
    """
    sig_seq = extract_solution_signature(L_seq)
    sig_par = extract_solution_signature(L_par)

    if sig_seq.keys() != sig_par.keys():
        print("ERROR: Different nodes in solutions!")
        return False

    for node in sig_seq:
        if sig_seq[node] != sig_par[node]:
            print(f"ERROR: Different Pareto sets at node {node}")
            print(f"  Sequential: {sig_seq[node]}")
            print(f"  Parallel:   {sig_par[node]}")
            return False

    return True


def reset_label_counter():
    """Reset the label ID counter for fair comparison."""
    Label._id_counter = 0


def run_test(name, V, predecessors, successors, edge_cost):
    """Run comparison test on a given instance."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    print(f"Nodes: {len(V)}, Edges: {len(edge_cost)}")

    # -------------------------
    # Sequential MDA
    # -------------------------
    reset_label_counter()
    print("\n[1] Running SEQUENTIAL MDA...")
    t_start = time.perf_counter()
    L_seq = mda(V, 0, predecessors, successors, edge_cost, parallel_dominance=False)
    t_seq = time.perf_counter() - t_start
    print(f"    Time: {t_seq:.6f} seconds")

    # -------------------------
    # Parallel MDA
    # -------------------------
    reset_label_counter()
    print("\n[2] Running PARALLEL MDA...")
    t_start = time.perf_counter()
    L_par = mda(V, 0, predecessors, successors, edge_cost, parallel_dominance=True)
    t_par = time.perf_counter() - t_start
    print(f"    Time: {t_par:.6f} seconds")

    # -------------------------
    # Compare solutions
    # -------------------------
    print("\n[3] Comparing solutions...")
    solutions_match = compare_solutions(L_seq, L_par)

    if solutions_match:
        print("    ✓ Solutions are IDENTICAL")
    else:
        print("    ✗ Solutions DIFFER!")

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "-"*40)
    print("SUMMARY")
    print("-"*40)
    print(f"Sequential time: {t_seq:.6f} s")
    print(f"Parallel time:   {t_par:.6f} s")

    if t_par > 0:
        speedup = t_seq / t_par
        print(f"Speedup:         {speedup:.2f}x")

        if speedup > 1:
            print(f"Parallel is {speedup:.2f}x FASTER")
        else:
            print(f"Sequential is {1/speedup:.2f}x FASTER (parallel overhead)")

    # Print Pareto set sizes
    print("\nPareto set sizes:")
    total_labels = 0
    for node in sorted(L_seq.keys()):
        size = len(L_seq[node])
        total_labels += size
        print(f"  Node {node}: {size} labels")
    print(f"  Total: {total_labels} labels")

    return solutions_match, t_seq, t_par


def main():
    # Setup logging with reduced verbosity for testing
    import logging
    setup_logging()
    # Reduce logging level to avoid debug spam
    logging.getLogger("mda").setLevel(logging.WARNING)

    print("\n" + "="*70)
    print(" MDA SEQUENTIAL vs PARALLEL COMPARISON TEST")
    print("="*70)

    all_passed = True

    # -------------------------
    # Test 1: Small instance
    # -------------------------
    V, predecessors, successors, edge_cost = build_instance_2cost()
    passed, t_seq, t_par = run_test(
        "Small instance (4 nodes, 2 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Test 2: Medium instance
    # -------------------------
    V, predecessors, successors, edge_cost = build_larger_instance(
        num_nodes=15, num_objectives=2
    )
    passed, t_seq, t_par = run_test(
        "Medium instance (15 nodes, 2 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Test 3: Larger instance
    # -------------------------
    V, predecessors, successors, edge_cost = build_larger_instance(
        num_nodes=20, num_objectives=3
    )
    passed, t_seq, t_par = run_test(
        "Larger instance (20 nodes, 3 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Test 4: Grid instance (many Pareto paths)
    # -------------------------
    V, predecessors, successors, edge_cost = build_grid_instance(
        rows=8, cols=8, num_objectives=2
    )
    passed, t_seq, t_par = run_test(
        "Grid instance (8x8=64 nodes, 2 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Test 5: Conflicting objectives (large Pareto fronts)
    # -------------------------
    V, predecessors, successors, edge_cost = build_conflicting_objectives_instance(
        num_layers=6, nodes_per_layer=6, num_objectives=2
    )
    passed, t_seq, t_par = run_test(
        "Conflicting objectives (6 layers x 6 nodes, 2 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Test 6: Very large conflicting objectives
    # -------------------------
    V, predecessors, successors, edge_cost = build_conflicting_objectives_instance(
        num_layers=8, nodes_per_layer=8, num_objectives=2
    )
    passed, t_seq, t_par = run_test(
        "Large conflicting objectives (8 layers x 8 nodes, 2 objectives)",
        V, predecessors, successors, edge_cost
    )
    all_passed = all_passed and passed

    # -------------------------
    # Final result
    # -------------------------
    print("\n" + "="*70)
    print(" FINAL RESULT")
    print("="*70)

    if all_passed:
        print("✓ ALL TESTS PASSED - Solutions are identical in both modes")
    else:
        print("✗ SOME TESTS FAILED - Check output above")

    print("\nNOTE: Using ThreadPoolExecutor with smart threshold.")
    print("      Parallelism activates only when |L[v]| >= 50 labels.")
    print("      Speedup expected only for very large Pareto fronts.")

    return all_passed


if __name__ == "__main__":
    main()
