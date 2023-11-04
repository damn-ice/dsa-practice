"""This module contains my personal study of graphs..."""
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


# depthFirstTraversal(graph, "a")
# depthFirstTraversalRecursion(graph, "a")
# breadthFirstTraversal(graph, "a")
print(connectedComponents(connected_component_graph))
