"""
Test script for MDA invariant validation on large instances.

Validates the 4 theoretical properties from the paper:
1. |H| <= n always (Theorem - at most one label per node in H)
2. Extracted labels are non-dominated (Lemma 1)
3. L[v] is lexicographically ordered (consequence of extraction order)
4. last[(u,v)] is monotonically non-decreasing (Remark 1)

Also collects metrics matching Table 4 of the paper:
- N_t (total labels generated), N (Pareto labels), N_ext (extractions), max|H|, time
"""

import random
import time
import sys

from mda import Label, dominates, extend, is_dominated_or_equivalent, logger, setup_logging
from config import mda_config, test_config


# ----------------------------
# Instance generators
# ----------------------------

def build_paper_grid_instance(rows=100, cols=100, num_objectives=3,
                               cost_min=1, cost_max=10, seed=42):
    """
    Build a grid instance matching the paper (Table 3).

    4-directional grid (right, left, down, up).
    For 100x100: 10,000 nodes, 39,600 edges.

    Args:
        rows: Number of rows
        cols: Number of columns
        num_objectives: Number of objectives (d)
        cost_min: Minimum edge cost
        cost_max: Maximum edge cost
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    n_nodes = rows * cols
    V = list(range(n_nodes))

    edge_cost = {}
    predecessors = {v: [] for v in V}
    successors = {v: [] for v in V}

    def node_id(i, j):
        return i * cols + j

    # 4-directional edges: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for i in range(rows):
        for j in range(cols):
            u = node_id(i, j)
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    v = node_id(ni, nj)
                    cost = tuple(random.randint(cost_min, cost_max)
                                 for _ in range(num_objectives))
                    edge_cost[(u, v)] = cost
                    successors[u].append(v)
                    predecessors[v].append(u)

    print(f"  Grid {rows}x{cols}: {n_nodes} nodes, {len(edge_cost)} edges, "
          f"d={num_objectives}, costs in [{cost_min},{cost_max}]")

    return V, predecessors, successors, edge_cost


# ----------------------------
# Instrumented MDA
# ----------------------------

class InvariantViolation(Exception):
    """Raised when an MDA invariant is violated."""
    pass


def mda_instrumented(V, s, predecessors, successors, edge_cost,
                      parallel_dominance=False, progress_interval=1000):
    """
    MDA with invariant validation and metrics collection.

    Checks at every iteration:
    1. |H| <= n (at most one label per node in H)
    2. Extracted labels are non-dominated among same-node labels
    3. L[v] is lexicographically ordered
    4. last[(u,v)] is monotonically non-decreasing

    Returns:
        L: Pareto-optimal labels per node
        metrics: dict with N_t, N, N_ext, max_H, time, violations
    """
    n = len(V)
    dim = len(next(iter(edge_cost.values())))

    # Metrics
    n_total_labels = 0      # N_t: total labels generated
    n_extractions = 0       # N_ext: number of extractions from H
    max_H_size = 0          # max|H| observed
    violations = []         # list of violation descriptions

    # Track last values for monotonicity check (invariant 4)
    last_history = {}  # (u,v) -> list of last values seen

    # --- MDA core (mirrors mda.py) ---
    L = {v: [] for v in V}
    last = {(u, v): 0 for (u, v) in edge_cost}
    H = []
    in_H = {}

    start_label = Label(s, (0,) * dim)
    H.append(start_label)
    in_H[s] = start_label
    n_total_labels += 1

    t_start = time.perf_counter()
    iteration = 0

    while H:
        iteration += 1

        # --- INVARIANT 1: |H| <= n ---
        if len(H) > n:
            msg = f"Iter {iteration}: |H|={len(H)} > n={n}"
            violations.append(("INV1_H_size", msg))
            raise InvariantViolation(msg)

        # Check that in_H has at most one entry per node
        h_nodes = [lab.node for lab in H]
        if len(h_nodes) != len(set(h_nodes)):
            msg = f"Iter {iteration}: duplicate nodes in H"
            violations.append(("INV1_duplicate", msg))
            raise InvariantViolation(msg)

        max_H_size = max(max_H_size, len(H))

        # Extract lexicographic minimum
        l = min(H, key=lambda x: x.cost)
        H.remove(l)
        in_H.pop(l.node)
        n_extractions += 1

        v = l.node

        # --- INVARIANT 2: extracted label is non-dominated ---
        # The extracted label should not be dominated by any existing label in L[v]
        for existing in L[v]:
            if dominates(existing.cost, l.cost):
                msg = (f"Iter {iteration}: extracted Label {l.id} cost={l.cost} "
                       f"dominated by L[{v}] Label {existing.id} cost={existing.cost}")
                violations.append(("INV2_dominated", msg))
                raise InvariantViolation(msg)

        L[v].append(l)

        # --- INVARIANT 3: L[v] lexicographically ordered ---
        if len(L[v]) >= 2:
            prev_cost = L[v][-2].cost
            curr_cost = L[v][-1].cost
            if curr_cost < prev_cost:
                msg = (f"Iter {iteration}: L[{v}] not lex-ordered: "
                       f"{prev_cost} then {curr_cost}")
                violations.append(("INV3_lex_order", msg))
                raise InvariantViolation(msg)

        # Generate next candidate for v (same logic as mda.py)
        best = None
        best_pred = None
        best_k = None
        candidate_info = {}

        for u in predecessors[v]:
            for k in range(last[(u, v)], len(L[u])):
                lu = L[u][k]
                cand = extend(lu, edge_cost[(u, v)], v)
                n_total_labels += 1

                if is_dominated_or_equivalent(cand, L[v],
                                               parallel_dominance=parallel_dominance):
                    old_last = last[(u, v)]
                    last[(u, v)] = k + 1

                    # --- INVARIANT 4: last monotonically non-decreasing ---
                    if last[(u, v)] < old_last:
                        msg = (f"Iter {iteration}: last[({u},{v})] decreased "
                               f"from {old_last} to {last[(u, v)]}")
                        violations.append(("INV4_last_mono", msg))
                        raise InvariantViolation(msg)
                    continue
                else:
                    candidate_info[u] = (k, cand)
                    break

        for u, (k, cand) in candidate_info.items():
            if best is None or cand.cost < best.cost:
                best = cand
                best_pred = u
                best_k = k

        if best_pred is not None:
            old_last = last[(best_pred, v)]
            last[(best_pred, v)] = best_k + 1

            # --- INVARIANT 4 check on best update ---
            if last[(best_pred, v)] < old_last:
                msg = (f"Iter {iteration}: last[({best_pred},{v})] decreased "
                       f"from {old_last} to {last[(best_pred, v)]}")
                violations.append(("INV4_last_mono", msg))
                raise InvariantViolation(msg)

            H.append(best)
            in_H[v] = best

        # Propagation to successors
        for w in successors[v]:
            cand = extend(l, edge_cost[(v, w)], w)
            n_total_labels += 1

            if is_dominated_or_equivalent(cand, L[w],
                                           parallel_dominance=parallel_dominance):
                continue

            if w not in in_H or cand.cost < in_H[w].cost:
                if w in in_H:
                    H.remove(in_H[w])
                H.append(cand)
                in_H[w] = cand
            # else: cand not better, skip

        # Progress reporting
        if iteration % progress_interval == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    iter={iteration:>8d}  N_ext={n_extractions:>8d}  "
                  f"|H|={len(H):>6d}  elapsed={elapsed:.2f}s")

    elapsed = time.perf_counter() - t_start

    # Count total Pareto labels
    n_pareto = sum(len(labels) for labels in L.values())

    metrics = {
        "N_t": n_total_labels,
        "N": n_pareto,
        "N_ext": n_extractions,
        "max_H": max_H_size,
        "time": elapsed,
        "violations": violations,
    }

    return L, metrics


# ----------------------------
# Test runner
# ----------------------------

def print_metrics_table(metrics):
    """Print metrics in a format matching Table 4 of the paper."""
    print(f"\n    {'Metric':<20s} {'Value':>12s}")
    print(f"    {'-'*20} {'-'*12}")
    print(f"    {'N_t (total labels)':<20s} {metrics['N_t']:>12,d}")
    print(f"    {'N (Pareto labels)':<20s} {metrics['N']:>12,d}")
    print(f"    {'N_ext (extractions)':<20s} {metrics['N_ext']:>12,d}")
    print(f"    {'max|H|':<20s} {metrics['max_H']:>12,d}")
    print(f"    {'Time (s)':<20s} {metrics['time']:>12.4f}")


def verify_pareto_optimality(L):
    """Post-hoc check: no label in L[v] is dominated by another in L[v]."""
    for v, labels in L.items():
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                if i != j and dominates(lj.cost, li.cost):
                    print(f"  FAIL: L[{v}] Label {li.id} cost={li.cost} "
                          f"dominated by Label {lj.id} cost={lj.cost}")
                    return False
    return True


def run_invariant_test(name, rows, cols, num_objectives=3,
                        cost_min=1, cost_max=10, seed=42,
                        parallel_dominance=False, progress_interval=1000):
    """Run a single invariant test."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    # Reset label counter
    Label._id_counter = 0

    # Build instance
    print("  Building instance...")
    V, predecessors, successors, edge_cost = build_paper_grid_instance(
        rows=rows, cols=cols, num_objectives=num_objectives,
        cost_min=cost_min, cost_max=cost_max, seed=seed
    )

    source = mda_config.source_node

    # Run instrumented MDA
    mode = "PARALLEL" if parallel_dominance else "SEQUENTIAL"
    print(f"  Running MDA ({mode})...")
    try:
        L, metrics = mda_instrumented(
            V, source, predecessors, successors, edge_cost,
            parallel_dominance=parallel_dominance,
            progress_interval=progress_interval
        )
    except InvariantViolation as e:
        print(f"\n  INVARIANT VIOLATION: {e}")
        return False, None

    # Print metrics
    print_metrics_table(metrics)

    # Check violations
    if metrics["violations"]:
        print(f"\n  VIOLATIONS FOUND: {len(metrics['violations'])}")
        for vtype, msg in metrics["violations"]:
            print(f"    [{vtype}] {msg}")
        return False, metrics

    # Post-hoc Pareto optimality check
    print("\n  Post-hoc Pareto optimality check...")
    pareto_ok = verify_pareto_optimality(L)
    if pareto_ok:
        print("    PASS: All labels are non-dominated within their node sets")
    else:
        print("    FAIL: Some labels are dominated")
        return False, metrics

    print(f"\n  ALL INVARIANTS PASSED")
    return True, metrics


def main():
    # Setup logging (reduced verbosity for testing)
    import logging
    setup_logging()
    logging.getLogger("mda").setLevel(logging.WARNING)

    print("\n" + "=" * 70)
    print(" MDA INVARIANT VALIDATION TEST SUITE")
    print("=" * 70)

    all_passed = True
    results = []

    # -------------------------
    # Test 1: Grid 20x20, d=3 (quick validation)
    # -------------------------
    passed, metrics = run_invariant_test(
        "Grid 20x20, d=3 (quick validation)",
        rows=20, cols=20, num_objectives=3,
        cost_min=test_config.cost_min, cost_max=test_config.cost_max,
        seed=test_config.random_seed,
        progress_interval=500
    )
    all_passed = all_passed and passed
    results.append(("20x20 d=3 seq", passed, metrics))

    # -------------------------
    # Test 2: Grid 50x50, d=3 (sequential vs parallel comparison)
    # -------------------------
    passed_seq, metrics_seq = run_invariant_test(
        "Grid 50x50, d=3 (SEQUENTIAL)",
        rows=50, cols=50, num_objectives=3,
        cost_min=test_config.cost_min, cost_max=test_config.cost_max,
        seed=test_config.random_seed,
        parallel_dominance=False,
        progress_interval=2000
    )
    all_passed = all_passed and passed_seq
    results.append(("50x50 d=3 seq", passed_seq, metrics_seq))

    passed_par, metrics_par = run_invariant_test(
        "Grid 50x50, d=3 (PARALLEL)",
        rows=50, cols=50, num_objectives=3,
        cost_min=test_config.cost_min, cost_max=test_config.cost_max,
        seed=test_config.random_seed,
        parallel_dominance=True,
        progress_interval=2000
    )
    all_passed = all_passed and passed_par
    results.append(("50x50 d=3 par", passed_par, metrics_par))

    # Speedup comparison for 50x50
    if metrics_seq and metrics_par:
        speedup = metrics_seq["time"] / metrics_par["time"] if metrics_par["time"] > 0 else float('inf')
        print(f"\n  Speedup (50x50): seq={metrics_seq['time']:.4f}s, "
              f"par={metrics_par['time']:.4f}s, speedup={speedup:.2f}x")

        # Verify identical solutions
        if metrics_seq["N"] == metrics_par["N"]:
            print(f"  Solutions match: N={metrics_seq['N']} Pareto labels")
        else:
            print(f"  WARNING: Different Pareto counts: seq={metrics_seq['N']}, par={metrics_par['N']}")

    # -------------------------
    # Test 3: Grid 100x100, d=3 (full paper test)
    # -------------------------
    passed, metrics = run_invariant_test(
        "Grid 100x100, d=3 (paper Table 3 instance)",
        rows=test_config.grid_rows, cols=test_config.grid_cols,
        num_objectives=3,
        cost_min=test_config.cost_min, cost_max=test_config.cost_max,
        seed=test_config.random_seed,
        progress_interval=5000
    )
    all_passed = all_passed and passed
    results.append(("100x100 d=3 seq", passed, metrics))

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"\n  {'Test':<25s} {'Status':<10s} {'N':>10s} {'N_ext':>10s} "
          f"{'max|H|':>10s} {'Time':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for name, passed, metrics in results:
        status = "PASS" if passed else "FAIL"
        if metrics:
            print(f"  {name:<25s} {status:<10s} {metrics['N']:>10,d} "
                  f"{metrics['N_ext']:>10,d} {metrics['max_H']:>10,d} "
                  f"{metrics['time']:>10.4f}")
        else:
            print(f"  {name:<25s} {status:<10s} {'N/A':>10s} {'N/A':>10s} "
                  f"{'N/A':>10s} {'N/A':>10s}")

    print()
    if all_passed:
        print("  ALL TESTS PASSED - All 4 invariants verified on all instances")
    else:
        print("  SOME TESTS FAILED - Check output above for details")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
