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


trapped_water_input = [1, 3, 2, 4, 3, 5, 4, 6, 5, 7]


def max_trapped_water(heights: list[int]) -> int:
    i, j, max_water = 0, len(heights) - 1, 0

    while i < j:
        width = j - i
        max_water = max(max_water, width * min(heights[i], heights[j]))

        if heights[i] <= heights[j]:
            i += 1
        else:
            j -= 1

    return max_water


CityAndRemainGas = namedtuple("CityAndRemainGas", ("city", "remaining_gallons"))

gallons = [50, 20, 5, 30, 25, 10, 10]
distances = [900, 600, 200, 400, 600, 200, 100]


def find_ample_city(gallons: list[int], distances: list[int]):
    """The city with the minimum gallons on entry is the ideal starting point"""
    MPG = 20
    remaining_gallons = 0

    # It's okay to start from 0 gallons as the remaining gas will end in 0...
    # Hence if it doesn't go below zero position 0 is the ideal starting point...
    city_remaining_gallons_pair = CityAndRemainGas(0, 0)
    num_cities = len(gallons)

    for i in range(1, num_cities):
        remaining_gallons += gallons[i - 1] - distances[i - 1] // MPG
        if remaining_gallons < city_remaining_gallons_pair.remaining_gallons:
            city_remaining_gallons_pair = CityAndRemainGas(i, remaining_gallons)

    return city_remaining_gallons_pair.city


# print(optimum_task_assignment(task))
# print(minimum_total_waiting_time(services))
# print(minimum_visit(intervals))
# print(majority_search(search_input))
# print(max_trapped_water(trapped_water_input))
print(find_ample_city(gallons, distances))
