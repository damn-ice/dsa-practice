"""This module contains my personal study of graphs..."""
from collections import namedtuple
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


# depthFirstTraversal(graph, "a")
# depthFirstTraversalRecursion(graph, "a")
# breadthFirstTraversal(graph, "a")
# print(connectedComponents(connected_component_graph))
# print(island_count(island))
# print(minimum_island_count(island))
print(search_maze(maze_sample, start_coor, end_coor))
