"""Personal study of Greedy Algorithms..."""

import operator
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


Interval = namedtuple("Interval", ("left", "right"))

intervals = [
    Interval(1, 3),
    Interval(4, 6),
    Interval(3, 5),
    Interval(10, 12),
    Interval(2, 15),
]


def minimum_visit(intervals: list[Interval]):
    intervals.sort(key=operator.attrgetter("right"))
    last_visit_time, num_visits = float("-inf"), 0

    for interval in intervals:
        if interval.left > last_visit_time:
            last_visit_time = interval.right
            num_visits += 1
    return num_visits


search_input = ["a", "b", "b", "a", "c", "b", "b"]


def majority_search(input_stream: list):
    """Majority element must be > 50% input"""
    candidate, candidate_count = None, 0

    for input in input_stream:
        if candidate_count == 0:
            candidate, candidate_count = input, candidate_count + 1
        elif candidate == input:
            candidate_count += 1
        else:
            candidate_count -= 1

    return candidate


# print(optimum_task_assignment(task))
# print(minimum_total_waiting_time(services))
# print(minimum_visit(intervals))
print(majority_search(search_input))
