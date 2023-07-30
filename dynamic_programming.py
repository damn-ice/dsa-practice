"""Dynamic Programming notes..."""

import itertools


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


# print(fibonacci_number(8))
# print(find_maximum_subarray(A))
# print(levenshtein_distance("Saturdays", "Sundays"))
print(number_of_ways(3, 3))
