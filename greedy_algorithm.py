"""Personal study of Greedy Algorithms..."""

from collections import namedtuple

PairedTasks = namedtuple("PairedTasks", ("task_1", "task_2"))

task = [1, 4, 6, 10, 26, 9]


def optimum_task_assignment(task_durations: list[int]) -> list[PairedTasks]:
    """Optimum Task is the corresponding pair of biggest and smallest."""
    task_durations.sort()

    return [
        PairedTasks(task_durations[i], task_durations[~i])
        for i in range(len(task_durations) // 2)
    ]


services = [2, 1, 5, 3]


def minimum_total_waiting_time(service_times: list[int]) -> int:
    service_times.sort()
    total_waiting_time = 0

    for i, service_time in enumerate(service_times):
        num_remaining_queries = len(service_times) - (i + 1)
        total_waiting_time += num_remaining_queries * service_time

    return total_waiting_time


# print(optimum_task_assignment(task))
print(minimum_total_waiting_time(services))
