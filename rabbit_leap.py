from collections import deque
import heapq

# --- Define State Space ---
INITIAL_STATE = ['E', 'E', 'E', '_', 'W', 'W', 'W']
GOAL_STATE = ['W', 'W', 'W', '_', 'E', 'E', 'E']

# --- Generate all valid moves ---
def get_successors(state):
    successors = []
    i = state.index('_')  # empty space

    for j in range(len(state)):
        if state[j] == '_':
            continue

        rabbit = state[j]
        direction = 1 if rabbit == 'E' else -1

        # Step move
        if j + direction == i:
            new_state = state.copy()
            new_state[j], new_state[i] = new_state[i], new_state[j]
            successors.append(new_state)

        # Jump move: over exactly one opposite rabbit
        if j + 2*direction == i:
            mid = j + direction
            if state[mid] != rabbit and state[mid] != '_':
                new_state = state.copy()
                new_state[j], new_state[i] = new_state[i], new_state[j]
                successors.append(new_state)

    return successors

# --- BFS (Optimal) ---
def bfs(start, goal):
    queue = deque([[start]])
    visited = {tuple(start)}
    explored = 0
    max_queue_size = 1

    while queue:
        max_queue_size = max(max_queue_size, len(queue))
        path = queue.popleft()
        current = path[-1]
        explored += 1

        if current == goal:
            return {
                "steps": len(path) - 1,
                "visited_nodes": len(visited),
                "max_queue_size": max_queue_size,
                "path": path
            }

        for next_state in get_successors(current):
            tup = tuple(next_state)
            if tup not in visited:
                visited.add(tup)
                queue.append(path + [next_state])

    return None

# --- DFS (Non-optimal) ---
def dfs(start, goal, limit=1000):
    stack = [[start]]
    visited = {tuple(start)}
    explored = 0
    max_stack_size = 1

    while stack:
        max_stack_size = max(max_stack_size, len(stack))
        path = stack.pop()
        current = path[-1]
        explored += 1

        if current == goal:
            return {
                "steps": len(path) - 1,
                "visited_nodes": len(visited),
                "max_stack_size": max_stack_size,
                "path": path
            }

        if len(path) < limit:
            for next_state in get_successors(current):
                tup = tuple(next_state)
                if tup not in visited:
                    visited.add(tup)
                    stack.append(path + [next_state])

    return None

# --- Heuristic for A* ---
def heuristic(state):
    h = 0
    for i, val in enumerate(state):
        if val == 'E' and i < 4:
            h += 1
        elif val == 'W' and i > 2:
            h += 1
    return h

# --- A* Search ---
def a_star(start, goal):
    pq = []
    heapq.heappush(pq, (heuristic(start), 0, start, [start]))
    visited = {tuple(start): 0}
    max_queue_size = 1

    while pq:
        max_queue_size = max(max_queue_size, len(pq))
        f, g, current, path = heapq.heappop(pq)

        if current == goal:
            return {
                "steps": len(path) - 1,
                "visited_nodes": len(visited),
                "max_queue_size": max_queue_size,
                "path": path
            }

        for next_state in get_successors(current):
            new_g = g + 1
            tup = tuple(next_state)
            if tup not in visited or new_g < visited[tup]:
                visited[tup] = new_g
                heapq.heappush(pq, (new_g + heuristic(next_state), new_g, next_state, path + [next_state]))

    return None

# --- Main Execution ---
if __name__ == "__main__":
    bfs_result = bfs(INITIAL_STATE, GOAL_STATE)
    dfs_result = dfs(INITIAL_STATE, GOAL_STATE)
    astar_result = a_star(INITIAL_STATE, GOAL_STATE)

    print("B. Rabbit Leap Problem\n")

    print("1) BFS Results:")
    print(f"• Number of steps to reach the solution: {bfs_result['steps']}")
    print(f"• Different states visited: {bfs_result['visited_nodes']}")
    print(f"• Maximum queue size: {bfs_result['max_queue_size']}\n")

    print("2) DFS Results:")
    print(f"• Number of steps to reach to solution: {dfs_result['steps']}")
    print(f"• Different states visited: {dfs_result['visited_nodes']}")
    print(f"• Maximum stack size: {dfs_result['max_stack_size']}\n")

    print("3) A* Results:")
    print(f"• Number of steps to reach the solution: {astar_result['steps']}")
    print(f"• Different states visited: {astar_result['visited_nodes']}")
    print(f"• Maximum queue size: {astar_result['max_queue_size']}")
