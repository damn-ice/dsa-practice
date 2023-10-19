"""Personal study of Greedy Algorithms..."""

from collections import namedtuple

PairedTasks = namedtuple("PairedTasks", ("task_1", "task_2"))

task = [1, 4, 6, 10, 26, 9]


def optimum_task_assignment(task_durations: list[int]):
    """Optimum Task is the corresponding pair of biggest and smallest."""
    task_durations.sort()

    return [
        PairedTasks(task_durations[i], task_durations[~i])
        for i in range(len(task_durations) // 2)
    ]


print(optimum_task_assignment(task))
