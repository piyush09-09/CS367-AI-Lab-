import heapq
import string
import nltk
import matplotlib.pyplot as plt

# ===================================================================
# 1. HELPER FUNCTIONS
# ===================================================================

def preprocess(text):
    sentences = nltk.sent_tokenize(text)
    normalized_sentences = []
    translator = str.maketrans('', '', string.punctuation)
    for sentence in sentences:
        lower = sentence.lower()
        no_punct = lower.translate(translator)
        normalized_sentences.append(no_punct.strip())
    return normalized_sentences

def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]

# ===================================================================
# 2. A* SEARCH ALGORITHM
# ===================================================================

def a_star_align(doc1_sents, doc2_sents):
    def gap_penalty(sentence):
        return len(sentence)
    start_state = (0,0)
    goal_state = (len(doc1_sents), len(doc2_sents))
    def heuristic(i,j):
        return 0  # Dijkstra-like behavior

    g_start = 0
    h_start = heuristic(0,0)
    f_start = g_start + h_start

    frontier = [(f_start, g_start, [], start_state)]
    explored = {}

    while frontier:
        f_cost, g_cost, path, current_state = heapq.heappop(frontier)
        i, j = current_state

        if current_state in explored and explored[current_state] <= g_cost:
            continue
        explored[current_state] = g_cost

        if current_state == goal_state:
            return path, g_cost

        # --- Generate neighbors ---
        if i < len(doc1_sents) and j < len(doc2_sents):
            align_cost = levenshtein_distance(doc1_sents[i], doc2_sents[j])
            new_path = path + [('align', i, j, align_cost)]
            heapq.heappush(frontier, (g_cost+align_cost, g_cost+align_cost, new_path, (i+1, j+1)))

        if i < len(doc1_sents):
            cost = gap_penalty(doc1_sents[i])
            new_path = path + [('skip_doc1', i, cost)]
            heapq.heappush(frontier, (g_cost+cost, g_cost+cost, new_path, (i+1, j)))

        if j < len(doc2_sents):
            cost = gap_penalty(doc2_sents[j])
            new_path = path + [('skip_doc2', j, cost)]
            heapq.heappush(frontier, (g_cost+cost, g_cost+cost, new_path, (i, j+1)))

    return None, float('inf')

# ===================================================================
# 3. VISUALIZATION FUNCTIONS
# ===================================================================

def visualize_alignment(path, doc1_sents, doc2_sents, name):
    """
    Visualizes the alignment path.
    Diagonal = alignment, Horizontal/Vertical = skips
    """
    x, y, costs = [], [], []
    i_pos, j_pos = 0, 0
    for step in path:
        if step[0] == 'align':
            x.append(step[1])
            y.append(step[2])
            costs.append(step[3])
            i_pos += 1
            j_pos += 1
        elif step[0] == 'skip_doc1':
            x.append(step[1])
            y.append(j_pos)
            costs.append(step[2])
            i_pos += 1
        elif step[0] == 'skip_doc2':
            x.append(i_pos)
            y.append(step[1])
            costs.append(step[2])
            j_pos += 1

    plt.figure(figsize=(8,6))
    for xi, yi, cost in zip(x, y, costs):
        plt.scatter(xi, yi, c='blue')
        plt.text(xi+0.05, yi+0.05, str(cost), fontsize=9)
    plt.plot(x, y, 'r--', alpha=0.5)
    plt.title(f"Alignment Path: {name}")
    plt.xlabel("Doc1 Sentence Index")
    plt.ylabel("Doc2 Sentence Index")
    plt.grid(True)
    plt.show()

def visualize_total_costs(test_cases):
    """
    Bar chart for total alignment costs.
    """
    names = [tc[0] for tc in test_cases]
    costs = [tc[1] for tc in test_cases]

    plt.figure(figsize=(8,5))
    plt.bar(names, costs, color='skyblue')
    plt.ylabel("Total Alignment Cost")
    plt.title("Comparison of Alignment Costs")
    plt.show()

# ===================================================================
# 4. ANALYZE AND PRINT RESULTS
# ===================================================================

def analyze_and_print_results(name, doc1, doc2, plagiarism_threshold=15, visualize=False):
    print(f"\n{'='*20} {name} {'='*20}")
    doc1_sents_raw = nltk.sent_tokenize(doc1)
    doc2_sents_raw = nltk.sent_tokenize(doc2)
    doc1_sents = preprocess(doc1)
    doc2_sents = preprocess(doc2)

    alignment, total_cost = a_star_align(doc1_sents, doc2_sents)
    
    if not alignment:
        print("Could not find an alignment.")
        return None

    plagiarism_found = False
    for step in alignment:
        if step[0] == 'align' and step[3] <= plagiarism_threshold:
            plagiarism_found = True

    if visualize:
        visualize_alignment(alignment, doc1_sents, doc2_sents, name)

    print(f"Total Alignment Cost: {total_cost}")
    return total_cost

# ===================================================================
# 5. TEST CASES
# ===================================================================

if __name__ == "__main__":
    
    test_cases_data = [
        ("Test Case 1: Identical", 
         "A* search is a graph traversal algorithm. It finds the shortest path from a start to a goal. It uses a heuristic to guide its search.",
         "A* search is a graph traversal algorithm. It finds the shortest path from a start to a goal. It uses a heuristic to guide its search."),
        
        ("Test Case 2: Slightly Modified",
         "The A* algorithm is a popular pathfinding algorithm in AI. It efficiently finds the shortest route between two points.",
         "The A star algorithm is a well-known pathfinding method in artificial intelligence. It quickly finds the shortest path between two nodes."),
        
        ("Test Case 3: Completely Different",
         "The sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma.",
         "Python is an interpreted, high-level, general-purpose programming language. Created by Guido van Rossum and first released in 1991."),
        
        ("Test Case 4: Partial Overlap",
         "Plagiarism detection is very important for academic integrity. The A* search algorithm can be applied to text alignment. This helps find similar sentence structures. It is a complex but powerful tool.",
         "Heuristic search is a key topic in AI. The A* search algorithm can be applied to text alignment. This helps identify copied content. We will evaluate its performance on several test cases.")
    ]

    results = []
    for name, doc1, doc2 in test_cases_data:
        visualize = True if "Partial Overlap" in name else False
        cost = analyze_and_print_results(name, doc1, doc2, visualize=visualize)
        results.append((name, cost))

    visualize_total_costs(results)
