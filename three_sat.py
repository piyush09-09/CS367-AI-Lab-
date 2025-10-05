import random
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

# 1. Random k-SAT generator
def generate_k_sat(k, m, n):
    clauses = []
    for _ in range(m):
        variables = random.sample(range(1, n + 1), k)
        clause = [v if random.random() < 0.5 else -v for v in variables]
        clauses.append(clause)
    return clauses

# 2. Evaluation functions
def clause_satisfied(clause, state):
    for literal in clause:
        var_index = abs(literal) - 1
        if literal > 0 and state[var_index] == 1:
            return True
        if literal < 0 and state[var_index] == 0:
            return True
    return False

def formula_satisfied(clauses, state):
    return sum(clause_satisfied(c, state) for c in clauses)

# 3. Heuristic functions
def heuristic_literals_imbalance(clauses, state):
    score = 0
    for clause in clauses:
        for literal in clause:
            var_index = abs(literal) - 1
            val = state[var_index]
            if literal > 0 and val == 1:
                score += 1
            elif literal < 0 and val == 0:
                score += 1
    return score

def heuristic_clause_weighted(clauses, state):
    score = 0
    for clause in clauses:
        weight = 1 / len(clause)
        if clause_satisfied(clause, state):
            score += weight
    return score

# 4. Successor generation
def flip_one_var(state):
    neighbors = []
    for i in range(len(state)):
        neighbor = state.copy()
        neighbor[i] = 1 - neighbor[i]
        neighbors.append(neighbor)
    return neighbors

# 5. Hill Climbing
def hill_climb(clauses, n, heuristic, max_iter=1000):
    state = [random.randint(0, 1) for _ in range(n)]
    best_score = heuristic(clauses, state)
    best_state = state.copy()
    for _ in range(max_iter):
        neighbors = flip_one_var(state)
        scores = [heuristic(clauses, nb) for nb in neighbors]
        max_score = max(scores)
        if max_score <= best_score:
            break
        idx = scores.index(max_score)
        state = neighbors[idx]
        best_score = max_score
        best_state = state.copy()
        if formula_satisfied(clauses, best_state) == len(clauses):
            return True, best_state
    return formula_satisfied(clauses, best_state) == len(clauses), best_state

# 6. Beam Search
def beam_search(clauses, n, heuristic, beam_width=3, max_iter=500):
    population = [[random.randint(0, 1) for _ in range(n)] for _ in range(beam_width)]
    for _ in range(max_iter):
        all_successors = []
        for state in population:
            all_successors.append(state)
            all_successors.extend(flip_one_var(state))
        # Remove duplicates
        unique_successors = []
        seen = set()
        for s in all_successors:
            t = tuple(s)
            if t not in seen:
                seen.add(t)
                unique_successors.append(s)
        unique_successors.sort(key=lambda s: heuristic(clauses, s), reverse=True)
        population = unique_successors[:beam_width]
        for s in population:
            if formula_satisfied(clauses, s) == len(clauses):
                return True, s
    return False, population[0]


# 7. Variable Neighborhood Descent
def vnd(clauses, n, heuristic, max_iter=500):
    state = [random.randint(0, 1) for _ in range(n)]
    best_state = state.copy()
    best_score = heuristic(clauses, state)
    def neighborhood_1(s): return flip_one_var(s)
    def neighborhood_2(s):
        neighbors = []
        for i, j in itertools.combinations(range(n), 2):
            nb = s.copy(); nb[i] = 1 - nb[i]; nb[j] = 1 - nb[j]; neighbors.append(nb)
        return neighbors
    def neighborhood_3(s):
        neighbors = []
        for i, j, k in itertools.combinations(range(n), 3):
            nb = s.copy(); nb[i] = 1 - nb[i]; nb[j] = 1 - nb[j]; nb[k] = 1 - nb[k]; neighbors.append(nb)
        return neighbors
    neighborhoods = [neighborhood_1, neighborhood_2, neighborhood_3]
    for _ in range(max_iter):
        improved = False
        for N in neighborhoods:
            neighbors = N(state)
            scores = [heuristic(clauses, nb) for nb in neighbors]
            max_score = max(scores)
            if max_score > best_score:
                idx = scores.index(max_score)
                state = neighbors[idx]
                best_score = max_score
                best_state = state.copy()
                improved = True
                break
        if not improved: break
        if formula_satisfied(clauses, best_state) == len(clauses): return True, best_state
    return formula_satisfied(clauses, best_state) == len(clauses), best_state

# 8. Penetrance calculation
def compute_penetrance(algorithm, heuristic, k, m, n, trials=20, **kwargs):
    success = 0
    for _ in range(trials):
        clauses = generate_k_sat(k, m, n)
        solved, _ = algorithm(clauses, n, heuristic, **kwargs)
        if solved: success += 1
    return 100 * success / trials

# 12. Runtime calculation
def compute_penetrance_and_runtime(algorithm, heuristic, k, m, n, trials=10, **kwargs):
    success = 0
    total_time = 0
    for _ in range(trials):
        clauses = generate_k_sat(k, m, n)
        start = time.time()
        solved, _ = algorithm(clauses, n, heuristic, **kwargs)
        total_time += time.time() - start
        if solved: success += 1
    avg_runtime = total_time / trials
    success_rate = 100 * success / trials
    return success_rate, avg_runtime

# 10. Grid/Table visualization of 3-SAT
def plot_3sat_grid(clauses, n):
    m = len(clauses)
    grid = np.zeros((m, n))
    for i, clause in enumerate(clauses):
        for literal in clause:
            var_index = abs(literal) - 1
            grid[i, var_index] = 1 if literal > 0 else -1
    cmap = plt.cm.RdYlGn
    plt.figure(figsize=(n, m))
    plt.imshow(grid, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(ticks=[-1, 0, 1], label="Literal Value")
    plt.xticks(range(n), [f"x{i+1}" for i in range(n)])
    plt.yticks(range(m), [f"C{i+1}" for i in range(m)])
    plt.title("Random 3-SAT Instance (Grid)")
    plt.show()

# 11. Graph-style visualization of 3-SAT
def plot_3sat_graph(clauses, n):
    G = nx.Graph()
    for i in range(n): G.add_node(f"x{i+1}", bipartite=0, type="var")
    for j in range(len(clauses)): G.add_node(f"C{j+1}", bipartite=1, type="clause")
    edge_colors = []
    for j, clause in enumerate(clauses):
        for literal in clause:
            var_name = f"x{abs(literal)}"; clause_name = f"C{j+1}"
            G.add_edge(clause_name, var_name)
            edge_colors.append("green" if literal > 0 else "red")
    pos = nx.bipartite_layout(G, nodes=[f"x{i+1}" for i in range(n)])
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True,
            node_color=["skyblue" if G.nodes[n]['type']=="var" else "orange" for n in G.nodes],
            node_size=1000, edge_color=edge_colors, width=2, font_size=12)
    plt.title("Random 3-SAT Instance (Graph)")
    plt.show()

# 9. Main experiment + runtime graph
if __name__ == "__main__":
    random.seed(123)
    test_cases = [(3, 15, 15), (3, 25, 25), (3, 45, 45)]
    heuristics = [("Literals Imbalance", heuristic_literals_imbalance),
                  ("Clause Weighted", heuristic_clause_weighted)]

    # Run experiments and print penetrance
    print("=== Hill Climbing ===")
    for hname, hfunc in heuristics:
        for k, m, n in test_cases:
            rate = compute_penetrance(hill_climb, hfunc, k, m, n)
            print(f"{hname} (m={m}, n={n}): {rate:.1f}%")
    
    print("\n=== Beam Search ===")
    for beam_width in [3, 4]:
        for hname, hfunc in heuristics:
            for k, m, n in test_cases:
                rate = compute_penetrance(beam_search, hfunc, k, m, n, beam_width=beam_width)
                print(f"{hname}, beam={beam_width} (m={m}, n={n}): {rate:.1f}%")
    
    print("\n=== Variable Neighborhood Descent ===")
    for hname, hfunc in heuristics:
        for k, m, n in test_cases:
            rate = compute_penetrance(vnd, hfunc, k, m, n)
            print(f"{hname} (m={m}, n={n}): {rate:.1f}%")

    # Demo Visualization
    print("\n=== Demo Visualization ===")
    k, m, n = 3, 8, 6
    demo_clauses = generate_k_sat(k, m, n)
    print("Random 3-SAT clauses:")
    for c in demo_clauses: print(c)
    plot_3sat_grid(demo_clauses, n)
    plot_3sat_graph(demo_clauses, n)

    # Average runtime graph
    print("\n=== Average Runtime Graph ===")
    algorithms = [("Hill Climbing", hill_climb, {}),
                  ("Beam Search (w=3)", beam_search, {"beam_width": 3}),
                  ("VND", vnd, {})]

    results = []
    for algo_name, algo_func, algo_args in algorithms:
        for hname, hfunc in heuristics:
            for k, m, n in test_cases:
                _, avg_time = compute_penetrance_and_runtime(algo_func, hfunc, k, m, n, **algo_args)
                results.append((algo_name, avg_time))

    algo_names = list(set(r[0] for r in results))
    avg_times = []
    for algo in algo_names:
        times = [r[1] for r in results if r[0] == algo]
        avg_times.append(sum(times)/len(times))

    plt.figure(figsize=(8,5))
    plt.bar(algo_names, avg_times, color=["#007acc","#ff9933","#66cc66"])
    plt.ylabel("Average Runtime (seconds)")
    plt.title("Average Runtime Comparison of Search Algorithms")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()

