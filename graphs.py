"""This module contains my personal study of graphs..."""
import string
from collections import deque, namedtuple
from queue import Queue

graph = {
    "a": ["b", "c"],
    "b": ["d"],
    "c": ["e"],
    "d": ["f"],
    "e": [],
    "f": [],
}


def depthFirstTraversal(graph: dict[str, list], start: str):
    stack = [start]

    while stack:
        current = stack.pop()
        print(current)

        # Current must exist on graph...
        neighbours = graph.get(current)

        for neighbour in neighbours:
            stack.append(neighbour)


def depthFirstTraversalRecursion(graph: dict[str, list], source: str):
    print(source)

    neighbours = graph.get(source)
    for neighbour in neighbours:
        depthFirstTraversalRecursion(graph, neighbour)


def breadthFirstTraversal(graph: dict[str, list], source: str):
    queue = Queue()
    queue.put(source)

    while not queue.empty():
        current = queue.get()
        print(current)

        # Current must exist on graph...
        neighbours = graph.get(current)

        for neighbour in neighbours:
            queue.put(neighbour)


connected_component_graph = {
    0: [8, 1, 5],
    1: [0],
    5: [0, 8],
    8: [0, 5],
    2: [3, 4],
    3: [2, 4],
    4: [3, 2],
}


def connectedComponents(graph: dict[int, list[int]]) -> int:
    def depthFirstTraversal(graph: dict[str, list], start: str):
        stack = [start]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)

                # Current must exist on graph...
                neighbours = graph.get(current)

                for neighbour in neighbours:
                    stack.append(neighbour)

    visited = set()
    count = 0

    for node in graph:
        if node not in visited:
            depthFirstTraversal(graph, node)
            count += 1

    return count


island = [
    ["W", "L", "W", "W", "W"],
    ["W", "L", "W", "W", "W"],
    ["W", "W", "W", "L", "W"],
    ["W", "W", "L", "L", "W"],
    ["L", "W", "W", "L", "L"],
    ["L", "L", "W", "W", "W"],
]


def island_count(island: list[list]) -> int:
    def explore_grid(grid: list[list], row: int, col: int) -> bool:
        row_bounds = 0 <= row and row < len(grid)
        col_bounds = 0 <= col and col < len(grid[0])
        pos = (row, col)

        if not row_bounds or not col_bounds:
            return False

        if grid[row][col] == "W":
            return False

        if pos in visited:
            return False

        # Unvisited peace of Land...
        visited.add(pos)
        explore_grid(grid, row - 1, col)
        explore_grid(grid, row + 1, col)
        explore_grid(grid, row, col - 1)
        explore_grid(grid, row, col + 1)
        return True

    visited = set()
    count = 0

    for row in range(len(island)):
        for col in range(len(island[row])):
            if explore_grid(island, row, col):
                count += 1

    return count


def minimum_island_count(island: list[list]) -> int:
    def explore_grid(grid: list[list], row: int, col: int) -> int:
        row_bounds = 0 <= row and row < len(grid)
        col_bounds = 0 <= col and col < len(grid[0])
        pos = (row, col)

        if not row_bounds or not col_bounds:
            return 0

        if grid[row][col] == "W":
            return 0

        if pos in visited:
            return 0

        # Unvisited peace of Land...
        visited.add(pos)
        top = explore_grid(grid, row - 1, col)
        bottom = explore_grid(grid, row + 1, col)
        left = explore_grid(grid, row, col - 1)
        right = explore_grid(grid, row, col + 1)
        return 1 + top + bottom + left + right

    visited = set()
    min_island = float("inf")

    for row in range(len(island)):
        for col in range(len(island[row])):
            current_island = explore_grid(island, row, col)
            if current_island > 0:
                min_island = min(min_island, current_island)

    return min_island


Coordinate = namedtuple("Coordinate", ("x", "y"))

maze_sample = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0],
]

start_coor = Coordinate(0, 0)
end_coor = Coordinate(4, 4)


def search_maze(
    maze: list[list[int]], start: Coordinate, end: Coordinate
) -> list[Coordinate]:
    WHITE, BLACK = range(2)

    def search_maze_helper(cur: Coordinate) -> bool:
        # Check if within bounds and White...
        if not (
            0 <= cur.x < len(maze)
            and 0 <= cur.y < len(maze[cur.x])
            and maze[cur.x][cur.y] == WHITE
        ):
            return False

        path.append(cur)
        maze[cur.x][cur.y] = BLACK  # Visited so update, memory efficient...

        if cur == end:
            return True

        if any(
            map(
                search_maze_helper,
                (
                    Coordinate(cur.x - 1, cur.y),
                    Coordinate(cur.x + 1, cur.y),
                    Coordinate(cur.x, cur.y - 1),
                    Coordinate(cur.x, cur.y + 1),
                ),
            )
        ):
            return True

        del path[-1]  # Remove entry as no path found...

        return False

    path = []
    if not search_maze_helper(start):
        return []

    return path


class GraphVertex:
    def __init__(self):
        # Distance attribute for BFS traversal
        self.d = -1
        # List to store neighboring vertices
        self.edges = []


def is_any_placement(G: list[GraphVertex]):
    """
    Checks if it's possible to assign each vertex of the input graph G
    to one of two sets such that no two connected vertices are in the same set.

    Parameters:
    - G: List of GraphVertex objects representing the graph.

    Returns:
    - True if the graph is bipartite, False otherwise.
    """

    def bfs(s: GraphVertex):
        """
        Performs BFS traversal starting from a given vertex s.

        Parameters:
        - s: Starting vertex for BFS traversal.

        Returns:
        - False if two vertices at the same distance have an edge between them, indicating a non-bipartite graph.
        - True otherwise.
        """
        # Initialize distance of the source vertex s to 0
        s.d = 0
        # Use a deque for BFS queue
        q = deque([s])

        while q:
            # Process neighbors of the front vertex in the queue
            for t in q[0].edges:
                if t.d == -1:
                    # If t has not been visited, assign distance and enqueue
                    t.d = q[0].d + 1
                    q.append(t)
                elif t.d == q[0].d:
                    # If t has the same distance, indicating an edge between vertices at the same level, the graph is not bipartite
                    return False

            # Dequeue the front vertex
            del q[0]

        # If the entire traversal is completed without finding any non-bipartite case, return True
        return True

    # Return True if the BFS traversal succeeds for all unvisited vertices in the graph
    return all(bfs(v) for v in G if v.d == -1)


v1 = GraphVertex()
v2 = GraphVertex()
v3 = GraphVertex()

v1.edges = [v2, v3]
v2.edges = [v1, v3]
v3.edges = [v2, v1]

G = [v1, v2, v3]


region_A = [
    ["B", "B", "B", "B"],
    ["W", "B", "W", "B"],
    ["B", "W", "W", "B"],
    ["B", "B", "B", "B"],
]

region_B = [
    ["B", "B", "B", "B"],
    ["W", "W", "W", "B"],
    ["B", "W", "W", "B"],
    ["B", "B", "B", "B"],
]


def fill_surrounded_regions(board: list[list]):
    n, m = len(board), len(board[0])

    q = deque(
        [(i, j) for k in range(n) for i, j in ((k, 0), (k, m - 1))]
        + [(i, j) for k in range(m) for i, j in ((0, k), (n - 1, k))]
    )  # Intialize queue with all boundaries...

    while q:
        x, y = q.popleft()
        if 0 <= x < n and 0 <= y < m and board[x][y] == "W":
            board[x][y] = "T"
            q.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
    board[:] = [["B" if col != "T" else "W" for col in row] for row in board]
    return board


transform = {"bat", "cot", "dog", "dag", "dot", "cat"}


def transform_string(D: set, s: str, t: str) -> int:
    StringWithDistance = namedtuple(
        "StringWithDistance", ("candidate_string", "distance")
    )
    q = deque([StringWithDistance(s, 0)])
    D.remove(s)

    while q:
        f = q.popleft()

        if f.candidate_string == t:
            return f.distance

        # Try all possible transformation of f.candidate_string...
        for i in range(len(f.candidate_string)):
            for c in string.ascii_lowercase:
                # Iterate through a to z and changing one letter...
                cand = f.candidate_string[:i] + c + f.candidate_string[i + 1 :]
                if cand in D:
                    q.append(StringWithDistance(cand, f.distance + 1))
                    D.remove(cand)

    return -1  # No match found...


# depthFirstTraversal(graph, "a")
# depthFirstTraversalRecursion(graph, "a")
# breadthFirstTraversal(graph, "a")
# print(connectedComponents(connected_component_graph))
# print(island_count(island))
# print(minimum_island_count(island))
# print(search_maze(maze_sample, start_coor, end_coor))
# print(fill_surrounded_regions(region_A))
# print(fill_surrounded_regions(region_B))
# print(is_any_placement(G))
print(transform_string(transform, "cat", "dog"))
