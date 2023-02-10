import math
from itertools import product


class BSTNode:
    def __init__(self, data=None, left=None, right=None):
        self.data, self.left, self.right = data, left, right

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

def palindrome_decomposition(input: str) -> list[list]: #backtracking...
    def palindrome_decomposition_helper(offset: int, partial_result: list):
        if offset == len(input):
            result.append(list(partial_result))
            return
        for i in range(offset + 1, len(input) + 1):
            prefix = input[offset: i]
            if prefix == prefix[::-1]:
                palindrome_decomposition_helper(i, partial_result + [prefix])

    result = []
    palindrome_decomposition_helper(0, [])
    return result

def generate_all_binary_tree(num_nodes: int) -> list[BSTNode]:
    if num_nodes == 0:
        return [None]

    result = []
    for num_left_tree_nodes in range(num_nodes):
        num_right_tree_nodes = num_nodes - 1 - num_left_tree_nodes
        left_subtree = generate_all_binary_tree(num_left_tree_nodes)
        right_subtree = generate_all_binary_tree(num_right_tree_nodes)
        # Generate all combination of left and right trees...
        result += [BSTNode(0, left, right) for left in left_subtree for right in right_subtree]
    return result

sudoku = [
    [5, 6, 0, 8, 4, 7, 0, 0, 0],
    [3, 0, 9, 0, 0, 0, 6, 0, 0],
    [0, 0, 8, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 8, 0, 0, 4, 0],
    [7, 9, 0, 6, 0, 2, 0, 1, 8],
    [0, 5, 0, 0, 3, 0, 0, 9, 0],
    [0, 0, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 6, 0, 0, 0, 8, 0, 7],
    [0, 0, 0, 3, 1, 6, 0, 5, 9],
]

def solve_sudoku(partial_sudoku: list[list]) -> list[list]: #backtracking...
    def solve_sudoku_helper(col: int, row: int):
        if col == len(partial_sudoku): #base case...
            col = 0 # move to begin of column and move to next row...
            row += 1
            if row == len(partial_sudoku):
                return True

        if partial_sudoku[col][row] != EMPTY_ENTRY:
           return solve_sudoku_helper(col + 1, row)

        def valid_sudoku(val: int, col: int, row: int) -> bool:
            # check column...
            if val in partial_sudoku[col]:
                return False
            # check row...
            if any(val == partial_sudoku[k][row] for k in range(len(partial_sudoku))):
                return False
            # check region...
            region_size = int(math.sqrt(len(partial_sudoku)))
            i = col // region_size
            j = row // region_size
            return not any(val == partial_sudoku[region_size * i + a][region_size * j + b] for a, b in product(range(region_size), repeat=2))

        for val in range(1, len(partial_sudoku) + 1):
            if valid_sudoku(val, col, row):
                partial_sudoku[col][row] = val
                if solve_sudoku_helper(col + 1, row):
                    return True
        # No valid entry reset and continue previous function call loop...
        partial_sudoku[col][row] = EMPTY_ENTRY
        return False
        
    EMPTY_ENTRY = 0
    solve_sudoku_helper(0, 0)
    return partial_sudoku

print(sudoku)
print(solve_sudoku(sudoku))    
# print(generate_all_binary_tree(5))
# print(palindrome_decomposition("0204451881"))
# print(generate_power_set([0, 1, 2]))
# print(generate_permutation([2,3,5]))
# print(n_queens(4))
# print(compute_tower_hanoi(8))





