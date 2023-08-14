"""Dynamic Programming notes..."""

import itertools
from collections import namedtuple


def fibonacci_number(n: int) -> int:
    if n <= 1:
        return n
    fib_minus_2, fib_minus_1 = 0, 1

    for _ in range(1, n):
        f = fib_minus_2 + fib_minus_1
        fib_minus_2, fib_minus_1 = fib_minus_1, f
    return fib_minus_1


A = [-100, 40, 123, 12, -100, -200, -150, 481, -50]


def find_maximum_subarray(A):
    min_sum = max_sum = 0
    for running_sum in itertools.accumulate(A):
        min_sum = min(min_sum, running_sum)
        max_sum = max(max_sum, running_sum - min_sum)
    return max_sum


def num_combinations_for_final_score(final_score, individual_p1ay_scores):
    # One way to reach 0.
    num_combinations_for_score = [
        [1] + [0] * final_score for _ in individual_p1ay_scores
    ]
    for i in range(len(individual_p1ay_scores)):
        for j in range(1, final_score + 1):
            without_this_play = num_combinations_for_score[i - 1][j] if i >= 1 else 0
            with_this_play = (
                num_combinations_for_score[i][j - individual_p1ay_scores[i]]
                if j >= individual_p1ay_scores[i]
                else 0
            )
            num_combinations_for_score[i][j] = without_this_play + with_this_play
    return num_combinations_for_score[-1][-1]


def levenshtein_distance(A: str, B: str):
    def compute_distance_between_prefixes(A_idx, B_idx):
        if A_idx < 0:
            # A is empty so add all B's characters...
            return B_idx + 1

        if B_idx < 0:
            # B is empty so add all A's characters...
            return A_idx + 1

        if distance_between_prefixes[A_idx][B_idx] == -1:
            if A[A_idx] == B[B_idx]:
                distance_between_prefixes[A_idx][
                    B_idx
                ] = compute_distance_between_prefixes(A_idx - 1, B_idx - 1)
            else:
                subsititue_last = compute_distance_between_prefixes(
                    A_idx - 1, B_idx - 1
                )
                add_last = compute_distance_between_prefixes(A_idx - 1, B_idx)
                delete_last = compute_distance_between_prefixes(A_idx, B_idx - 1)
                distance_between_prefixes[A_idx][B_idx] = 1 + min(
                    subsititue_last, add_last, delete_last
                )
        return distance_between_prefixes[A_idx][B_idx]

    distance_between_prefixes = [[-1] * len(B) for _ in A]
    return compute_distance_between_prefixes(len(A) - 1, len(B) - 1)


def number_of_ways(n: int, m: int) -> int:
    def compute_number_of_ways_to_xy(x, y):
        if x == y == 0:
            return 1

        if number_of_ways[x][y] == 0:
            ways_top = 0 if x == 0 else compute_number_of_ways_to_xy(x - 1, y)
            ways_left = 0 if y == 0 else compute_number_of_ways_to_xy(x, y - 1)
            number_of_ways[x][y] = ways_top + ways_left
        return number_of_ways[x][y]

    number_of_ways = [[0] * m for _ in range(n)]
    return compute_number_of_ways_to_xy(n - 1, m - 1)


def is_pattern_contained_in_grid(grid: list[list], S: list) -> bool:
    def is_pattern_suffix_contained_starting_at_xy(x, y, offset):
        if len(S) == offset:
            # Nothing left to complete...
            return True

        if (
            (0 <= x < len(grid) and 0 <= y < len(grid[x]))
            and grid[x][y] == S[offset]
            and (x, y, offset) not in previous_attempts
            and any(
                is_pattern_suffix_contained_starting_at_xy(x + a, y + b, offset + 1)
                for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1))
            )
        ):
            return True

        previous_attempts.add((x, y, offset))
        return False

    previous_attempts = set()
    return any(
        is_pattern_suffix_contained_starting_at_xy(i, j, 0)
        for i in range(len(grid))
        for j in range(len(grid[i]))
    )


Item = namedtuple("Item", ("weight", "value"))

items = [
    Item(weight=2, value=10),
    Item(weight=6, value=15),
    Item(weight=5, value=70),
    Item(weight=7, value=40),
]


def knapsack_problem(items: list[Item], capacity: int):
    """Max when a value is picked and when it's not picked"""
    # Returns the optimum value when we choose from items[:k + 1] and have a
    # capacity of availabe_capacity.
    def optimum_subject_to_item_and_capacity(k: int, available_capacity: int):
        if k < 0:  # No items can be chosen...
            return 0

        if V[k][available_capacity] == -1:
            without_curr_item = optimum_subject_to_item_and_capacity(
                k - 1, available_capacity
            )

            with_curr_item = (
                0
                if available_capacity < items[k].weight
                else (
                    items[k].value
                    + optimum_subject_to_item_and_capacity(
                        k - 1, available_capacity - items[k].weight
                    )
                )
            )

            V[k][available_capacity] = max(without_curr_item, with_curr_item)
        return V[k][available_capacity]

    # Unit by Unit (0 to capacity) for the number of items...
    V = [[-1] * (capacity + 1) for _ in items]
    return optimum_subject_to_item_and_capacity(len(items) - 1, capacity)


def decompose_into_dictionary_words(domain: str, dictionary: list[str]) -> list:
    last_length = [-1] * len(domain)
    for i in range(len(domain)):
        if domain[: i + 1] in dictionary:
            last_length[i] = i + 1

        if last_length[i] == -1:
            for j in range(i):
                # Check if another word exist from last word...
                if last_length[j] != -1 and domain[j + 1 : i + 1] in dictionary:
                    last_length[i] = i - j
                    break

    decomposition = []
    if last_length[-1] != -1:  # All strings found...
        idx = len(domain) - 1
        while idx >= 0:
            # Append the words in reverse appearance...
            decomposition.append(domain[idx + 1 - last_length[idx] : idx + 1])
            idx -= last_length[idx]

        decomposition = decomposition[::-1]
    return decomposition


def maximum_revenue(coins: list):
    """Maximum revenue while the opponents is also maximizing"""

    def compute_maximum_revenue_for_range(a: int, b: int):
        # No coins left...
        if a > b:
            return 0

        if maximum_revenue_for_range[a][b] == 0:
            # Select coin plus the minimum due to opponent also maximizing revenue...
            max_revenue_a = coins[a] + min(
                compute_maximum_revenue_for_range(a + 2, b),
                compute_maximum_revenue_for_range(a + 1, b - 1),
            )

            max_revenue_b = coins[b] + min(
                compute_maximum_revenue_for_range(a + 1, b - 1),
                compute_maximum_revenue_for_range(a, b - 2),
            )
            maximum_revenue_for_range[a][b] = max(max_revenue_a, max_revenue_b)
        return maximum_revenue_for_range[a][b]

    maximum_revenue_for_range = [[0] * len(coins) for _ in coins]
    return compute_maximum_revenue_for_range(0, len(coins) - 1)


def number_of_ways_to_top(top: int, maximum_step: int) -> int:
    def compute_number_of_ways_to_h(h: int):
        if h <= 1:
            return 1

        if number_of_ways_to_h[h] == 0:
            number_of_ways_to_h[h] = sum(
                compute_number_of_ways_to_h(h - i)
                # The min ensure it's within the range of both max_steps and h...
                for i in range(1, min(maximum_step, h) + 1)
            )

        return number_of_ways_to_h[h]

    number_of_ways_to_h = [0] * (top + 1)
    return compute_number_of_ways_to_h(top)


# print(fibonacci_number(8))
# print(find_maximum_subarray(A))
# print(levenshtein_distance("Saturdays", "Sundays"))
# print(number_of_ways(30, 30))
# print(
#     is_pattern_contained_in_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 5, 8])
# )  # True...
# print(
#     is_pattern_contained_in_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3, 4])
# )  # False...
# print(knapsack_problem(items, 10))
# print(
#     decompose_into_dictionary_words(
#         "appleorangebanana", ["apple", "banana", "pear", "orange"]
#     )
# )
# print(decompose_into_dictionary_words("ABABC", ["A", "B", "AB", "C"]))
# print(maximum_revenue([5, 25, 10, 1]))
print(number_of_ways_to_top(4, 2))
