import configparser
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


# ----------------------------
# Setup logging
# ----------------------------
def load_logging_config(config_path="logging_config.ini"):
    """Carica la configurazione del logging da file .ini"""
    config = configparser.ConfigParser()

    # Valori di default
    defaults = {
        'file_level': 'INFO',
        'console_level': 'INFO',
        'log_dir': 'logs',
        'file_enabled': 'true',
        'console_enabled': 'true'
    }

    if os.path.exists(config_path):
        config.read(config_path)
        return {
            'file_level': config.get('logging', 'file_level', fallback=defaults['file_level']),
            'console_level': config.get('logging', 'console_level', fallback=defaults['console_level']),
            'log_dir': config.get('logging', 'log_dir', fallback=defaults['log_dir']),
            'file_enabled': config.getboolean('logging', 'file_enabled', fallback=True),
            'console_enabled': config.getboolean('logging', 'console_enabled', fallback=True)
        }
    return defaults


def setup_logging(config_path="logging_config.ini"):
    """
    Configura il logging con file separato per ogni esecuzione.
    Legge la configurazione da logging_config.ini
    Nome file: mda_YYYYMMDD_HHMMSS_<uuid>.log
    """
    cfg = load_logging_config(config_path)

    log_dir = cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    log_filename = f"mda_{timestamp}_{run_id}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configura il logger
    mda_logger = logging.getLogger("mda")
    mda_logger.setLevel(logging.DEBUG)  # Livello base, i handler filtrano

    # Rimuovi handler esistenti
    mda_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # File handler
    if cfg['file_enabled']:
        file_level = getattr(logging, cfg['file_level'].upper(), logging.INFO)
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        mda_logger.addHandler(file_handler)

    # Console handler
    if cfg['console_enabled']:
        console_level = getattr(logging, cfg['console_level'].upper(), logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        mda_logger.addHandler(console_handler)

    if cfg['file_enabled']:
        mda_logger.info(f"Log file created: {log_path}")
        mda_logger.info(f"Logging config: file_level={cfg['file_level']}, console_level={cfg['console_level']}")

    return mda_logger, log_path


# Logger globale (inizializzato da setup_logging)
logger = logging.getLogger("mda")


class Label:
    _id_counter = 0

    def __init__(self, node, cost, pred=None):
        self.id = Label._id_counter
        Label._id_counter += 1

        self.node = node      # nodo corrente
        self.cost = tuple(cost)  # vettore dei costi (tuple)
        self.pred = pred      # label precedente (per ricostruire il cammino)


# ----------------------------
# Dominanza di Pareto
# ----------------------------
def dominates(a, b):
    """
    Check if cost vector a dominates cost vector b.
    a dominates b if: a[i] <= b[i] for all i AND a[j] < b[j] for at least one j
    """
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# Helper functions for parallel execution (must be at module level for pickling)
def _check_dominates(args):
    """Helper for parallel dominance check."""
    existing_cost, new_cost = args
    return dominates(existing_cost, new_cost)


def _check_dominates_or_equivalent(args):
    """Helper for parallel dominance/equivalence check."""
    existing_cost, new_cost = args
    return existing_cost == new_cost or dominates(existing_cost, new_cost)


# ----------------------------
# Sequential dominance checks (baseline)
# ----------------------------
def is_dominated_sequential(label, labels):
    """Sequential dominance check: O(n) where n = |labels|."""
    for l in labels:
        if dominates(l.cost, label.cost):
            return True
    return False


def is_dominated_or_equivalent_sequential(label, labels):
    """Sequential check if label is dominated OR has equivalent cost."""
    for l in labels:
        if l.cost == label.cost:  # equivalente
            return True
        if dominates(l.cost, label.cost):  # dominata
            return True
    return False


# ----------------------------
# Parallel dominance checks (Version 1 parallel MDA)
# ----------------------------
# Threshold: only use parallelism if Pareto front is large enough
PARALLEL_THRESHOLD = 50


def is_dominated_parallel(label, labels, max_workers=None):
    """
    Parallel dominance check using ThreadPoolExecutor.
    Intended for nodes with large Pareto fronts.
    """
    if not labels:
        return False

    # Prepare arguments for parallel execution
    args_list = [(l.cost, label.cost) for l in labels]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_check_dominates, args_list)
        # Use short-circuit: return True as soon as one dominates
        for result in results:
            if result:
                return True
        return False


def is_dominated_or_equivalent_parallel(label, labels, max_workers=None):
    """
    Parallel check if label is dominated OR has equivalent cost.
    Uses ThreadPoolExecutor for large Pareto fronts.
    """
    if not labels:
        return False

    # Prepare arguments for parallel execution
    args_list = [(l.cost, label.cost) for l in labels]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(_check_dominates_or_equivalent, args_list)
        # Use short-circuit: return True as soon as one matches
        for result in results:
            if result:
                return True
        return False


# ----------------------------
# Unified interfaces
# ----------------------------
def is_dominated(label, labels, parallel_dominance=False, max_workers=None):
    """
    Unified dominance check interface.

    Args:
        label: Label to check
        labels: List of existing labels to check against
        parallel_dominance: If True, use parallel processing (only if |labels| >= threshold)
        max_workers: Max workers for parallel execution (None = auto)

    Returns:
        True if label is dominated by any label in labels
    """
    # Smart threshold: only use parallelism if worth the overhead
    if parallel_dominance and len(labels) >= PARALLEL_THRESHOLD:
        return is_dominated_parallel(label, labels, max_workers)
    else:
        return is_dominated_sequential(label, labels)


def is_dominated_or_equivalent(label, labels, parallel_dominance=False, max_workers=None):
    """
    Unified interface for dominance/equivalence check.

    Args:
        label: Label to check
        labels: List of existing labels to check against
        parallel_dominance: If True, use parallel processing (only if |labels| >= threshold)
        max_workers: Max workers for parallel execution (None = auto)

    Returns:
        True if label is dominated by or equivalent to any label in labels
    """
    # Smart threshold: only use parallelism if worth the overhead
    if parallel_dominance and len(labels) >= PARALLEL_THRESHOLD:
        return is_dominated_or_equivalent_parallel(label, labels, max_workers)
    else:
        return is_dominated_or_equivalent_sequential(label, labels)


# ----------------------------
# Estensione di una label
# ----------------------------
def extend(label, edge_cost, v):
    new_cost = tuple(x + y for x, y in zip(label.cost, edge_cost))
    return Label(v, new_cost, label)


def next_candidate_label(v, predecessors, L, edge_cost, last, parallel_dominance=False):
    """
    Find the next candidate label for node v.

    Args:
        v: Target node
        predecessors: Dict of predecessor lists
        L: Dict of permanent labels per node
        edge_cost: Dict of edge costs
        last: Dict of last processed indices
        parallel_dominance: If True, use parallel dominance checking
    """
    best = None
    best_pred = None
    best_k = None

    candidate_info = {}

    logger.debug(f"  [nextCandidate] Searching candidate for node {v}")
    logger.debug(f"    Predecessors: {predecessors[v]}")
    logger.debug(f"    L[{v}] current: {[(l.id, l.cost) for l in L[v]]}")

    for u in predecessors[v]:
        logger.debug(f"    Exploring predecessor {u}, last[({u},{v})]={last[(u,v)]}, |L[{u}]|={len(L[u])}")
        logger.debug(f"      L[{u}] = {[(l.id, l.cost) for l in L[u]]}")

        # Inner loop: find first valid label
        for k in range(last[(u, v)], len(L[u])):
            lu = L[u][k]
            cand = extend(lu, edge_cost[(u, v)], v)

            logger.debug(f"      k={k}: Label {lu.id} cost={lu.cost} -> cand Label {cand.id} cost={cand.cost}")

            if is_dominated_or_equivalent(cand, L[v], parallel_dominance=parallel_dominance):
                logger.debug(f"        Label {cand.id} dominated or equivalent, SKIP and update last")
                # Dominated/equivalent label: update last and continue
                last[(u, v)] = k + 1
                continue
            else:
                logger.debug(f"        Label {cand.id} valid! Saving as candidate for predecessor {u}")
                # Valid label found: save and move to next predecessor
                candidate_info[u] = (k, cand)
                break  # move to next predecessor

    # Choose best among all candidates
    for u, (k, cand) in candidate_info.items():
        if best is None or cand.cost < best.cost:
            best = cand
            best_pred = u
            best_k = k
            logger.debug(f"    Comparing candidates: Label {cand.id} cost={cand.cost} from pred {u} -> new best")

    # Update last ONLY for the predecessor that produced the best
    if best_pred is not None:
        last[(best_pred, v)] = best_k + 1
        logger.debug(f"    Updating last[({best_pred},{v})] = {best_k + 1}")

    logger.debug(f"  [nextCandidate] Result for node {v}: Label {best.id if best else None}, cost={best.cost if best else None}")

    return best


# ----------------------------
# Algoritmo MDA
# ----------------------------
def mda(V, s, predecessors, successors, edge_cost, parallel_dominance=False):
    """
    Multiobjective Dijkstra Algorithm (MDA).

    Args:
        V: List of vertices
        s: Source node
        predecessors: Dict mapping node -> list of predecessors
        successors: Dict mapping node -> list of successors
        edge_cost: Dict mapping (u, v) -> cost tuple
        parallel_dominance: If True, use parallel dominance checking (Version 1 parallel MDA)
                           If False, use sequential dominance checking (classic MDA)

    Returns:
        L: Dict mapping node -> list of Pareto-optimal labels
    """
    if parallel_dominance:
        logger.info("MDA running with PARALLEL dominance checking")
    else:
        logger.info("MDA running with SEQUENTIAL dominance checking")

    # label permanenti
    L = {v: [] for v in V}

    # indici per next candidate
    last = {(u, v): 0 for (u, v) in edge_cost}

    # priority queue (semplice lista)
    H = []
    in_H = {}

    dim = len(next(iter(edge_cost.values())))
    start = Label(s, (0,) * dim)

    H.append(start)
    in_H[s] = start

    iteration = 0
    while H:
        iteration += 1
        # estrai minimo lessicografico
        l = min(H, key=lambda x: x.cost)
        H.remove(l)
        in_H.pop(l.node)

        v = l.node

        logger.debug(f"\n{'='*60}")
        logger.debug(f"ITERATION {iteration}")
        logger.debug(f"Extracted: Label {l.id}, node={v}, cost={l.cost}, pred={l.pred.id if l.pred else None}")
        logger.debug(f"H after extraction: {[(lab.id, lab.node, lab.cost) for lab in H]}")

        L[v].append(l)

        logger.debug(f"L[{v}] now contains: {[(lab.id, lab.cost) for lab in L[v]]}")

        # Generate next candidate for v
        nxt = next_candidate_label(v, predecessors, L, edge_cost, last,
                                   parallel_dominance=parallel_dominance)
        if nxt is not None:
            H.append(nxt)
            in_H[v] = nxt
            logger.debug(f"Added nextCandidate for {v}: Label {nxt.id}, cost={nxt.cost}")

        # Propagation to successors
        logger.debug(f"Propagation from {v} to successors: {successors[v]}")

        for w in successors[v]:
            cand = extend(l, edge_cost[(v, w)], w)

            logger.debug(f"  -> {w}: Label {cand.id}, cost={cand.cost}, pred=Label {l.id}")

            if is_dominated_or_equivalent(cand, L[w], parallel_dominance=parallel_dominance):
                logger.debug(f"     Label {cand.id} dominated/equivalent in L[{w}]={[(lab.id, lab.cost) for lab in L[w]]}, SKIP")
                continue

            if w not in in_H or cand.cost < in_H[w].cost:
                if w in in_H:
                    logger.debug(f"     Label {cand.id} better than in_H[{w}]=Label {in_H[w].id} cost={in_H[w].cost}, REPLACING")
                    H.remove(in_H[w])
                else:
                    logger.debug(f"     {w} not in H, ADDING Label {cand.id}")
                H.append(cand)
                in_H[w] = cand
            else:
                logger.debug(f"     Label {cand.id} not better than in_H[{w}]=Label {in_H[w].id} cost={in_H[w].cost}, SKIP")

        logger.debug(f"End of iteration. H={[(lab.id, lab.node, lab.cost) for lab in H]}")

    return L
