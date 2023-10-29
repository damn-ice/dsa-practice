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


# depthFirstTraversal(graph, "a")
# depthFirstTraversalRecursion(graph, "a")
breadthFirstTraversal(graph, "a")
