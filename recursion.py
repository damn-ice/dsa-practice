
def compute_tower_hanoi(num_rings: int) -> list[list]:
    NUM_PEGS = 3
    def compute_tower_hanoi_helper(num_rings_to_move: int, from_peg: int, to_peg: int, use_peg: int):
        if num_rings_to_move > 0:
            compute_tower_hanoi_helper(num_rings_to_move - 1, from_peg, use_peg, to_peg)
            pegs[to_peg].append(pegs[from_peg].pop())
            compute_tower_hanoi_helper(num_rings_to_move - 1, use_peg, to_peg, from_peg)
    # Initialize the pegs with rings...
    pegs = [list(reversed(range(1, num_rings + 1)))] + [[] for _ in range(1, NUM_PEGS)]
    compute_tower_hanoi_helper(num_rings, 0, 1, 2)
    return pegs

def n_queens(n: int) -> list[list]:
    def solve_n_queens(row: int):
        # We have completed a non-attacking placement...
        if row == n:
            result.append(list(col_placement))
            return
        for col in range(n):
            # Attacking placement for a queen in other rows is 0 and difference between other row and the queen's row... 
            if all(abs(col_p - col) not in (0, row - row_p) for row_p, col_p in enumerate(col_placement[: row])):
                col_placement[row] = col
                # solve the queen placement for the next row...
                solve_n_queens(row + 1)

    result, col_placement = [], [0] * n
    solve_n_queens(0)
    return result

def generate_permutation(A: list) -> list[list]:
    def generate_permutation_helper(i: int):
        if i == len(A) - 1:
            result.append(A.copy())
            return

        for j in range(i, len(A)):
            A[i], A[j] = A[j], A[i]
            generate_permutation_helper(i + 1)
            # Swap back the array...
            A[i], A[j] = A[j], A[i]
    result = []
    generate_permutation_helper(0)
    return result

def generate_power_set(input_set: list) -> list[list]:
    """Union without the element and the element"""
    def generate_power_set_helper(to_be_selected: int, selected_so_far: list):
        if to_be_selected == len(input_set):
            result.append(list(selected_so_far))
            return
        # Without the element
        generate_power_set_helper(to_be_selected + 1, selected_so_far)
        generate_power_set_helper(to_be_selected + 1, selected_so_far + [input_set[to_be_selected]])

    result = []
    generate_power_set_helper(0, [])
    return result

print(generate_power_set([0, 1, 2]))
# print(generate_permutation([2,3,5]))
# print(n_queens(4))
# print(compute_tower_hanoi(8))





