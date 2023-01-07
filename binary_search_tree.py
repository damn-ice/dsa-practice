"""Binary search tree can do same job as a hashmap (insertion, deleting, lookup) and also find min, max, successor, predecessor efficiently"""

import bintrees
import math

class BSTNode:
    def __init__(self, data=None, left=None, right=None):
        self.data, self.left, self.right = data, left, right

A = BSTNode()
B = BSTNode()
C = BSTNode()
D = BSTNode()
E = BSTNode()
F = BSTNode()
G = BSTNode()
H = BSTNode()
I = BSTNode()
J = BSTNode()
K = BSTNode()
L = BSTNode()
M = BSTNode()
N = BSTNode()
O = BSTNode()
P = BSTNode()

class ABSqrt2:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.val = a + b * math.sqrt(2)

    def __lt__(self, other: object) -> bool:
        return self.val < other.val
    
    def __eq__(self, other: object) -> bool:
        return self.val == other.val

A.data, A.left, A.right = 19, B, I
B.data, B.left, B.right = 7, C, F
C.data, C.left, C.right = 3, D, E
D.data, E.data = 2, 5
F.data, F.right = 11, G
G.data, G.left = 17, H
H.data = 13
I.data, I.left, I.right = 43, J, O
J.data, J.right = 23, K
K.data, K.left, K.right = 37, L, N
L.data, L.right, N.data = 29, M, 41
M.data = 31
O.data, O.right, P.data = 47, P, 53

def inorder_traversal_recursion(root: BSTNode):
    if root:
        inorder_traversal_recursion(root.left)
        print(root.data)
        inorder_traversal_recursion(root.right)

def search_bst(root: BSTNode, key: int) -> tuple:
    return None if not root else (root.data, root) if root.data == key  else search_bst(root.left, key) if root.data > key else search_bst(root.right, key)

# This solution is better than the direct approach because you don't need to worry if a node has a left or/ and right subtree, each node just handles it's own test...
def is_binary_tree_bst(root: BSTNode, low=float("-inf"), high=float("inf")) -> bool:
    if not root:
        return True
    elif not low <= root.data <= high:
        return False
    return (is_binary_tree_bst(root.left, low, root.data) and is_binary_tree_bst(root.right, root.data, high))

def inorder_traversal_is_binary_tree(root: BSTNode) -> bool:
    s, prev = [], float("-inf")
    while s or root:
        if root:
            s.append(root)
            root = root.left
        else:
            root = s.pop()
            if prev > root.data:
                return False
            prev, root = root.data, root.right
    return True

def find_first_greater_than_k(root: BSTNode, k: int) -> tuple:
    first_so_far = None
    while root:
        if root.data > k:
            first_so_far, root = (root.data, root), root.left
        else:
            root = root.right
    return first_so_far

def find_k_largest_bst(tree: BSTNode, k: int) -> list[tuple]:
    """Reverse inorder traversal"""
    def k_largest_bst_helper(tree: BSTNode):
        if tree and len(k_largest) < k:
            k_largest_bst_helper(tree.right)
            if len(k_largest) < k:
                k_largest.append((tree.data, tree))
                k_largest_bst_helper(tree.left)
    k_largest = []
    k_largest_bst_helper(tree)
    return k_largest

def find_LCA_BST(tree: BSTNode, a: BSTNode, b: BSTNode) -> BSTNode:
    """First node that is bigger than small but less than big is the LCA"""
    small = a if a.data < b.data else b
    big = b if b.data > a.data else a
    while tree.data < small.data or tree.data > big.data:
        while tree.data < small.data:
            tree = tree.right
        while tree.data > big.data:
            tree = tree.left
    return (tree.data, tree)

def rebuild_bst_from_pre_order(pre_order_sequence: list) -> BSTNode: #O(n^2) (left skewed BST)
    if not pre_order_sequence:
        return None
    transition_point = next((i for i, node in enumerate(pre_order_sequence) if node > pre_order_sequence[0]), len(pre_order_sequence)) #O(n) operation...

    return BSTNode(pre_order_sequence[0], rebuild_bst_from_pre_order(pre_order_sequence[1: transition_point]), rebuild_bst_from_pre_order(pre_order_sequence[transition_point:]))

def rebuild_bst_from_preorder_O_n(preorder_sequence: list) -> BSTNode: #O(n) worst case...
    """Builds the left and right subtree in the same recursion unlike the previous"""
    def rebuild_bst_helper(lower_bound, upper_bound) -> BSTNode:
        if root_idx[0] == len(preorder_sequence):
            return None
        root = preorder_sequence[root_idx[0]]
        if not lower_bound <= root <= upper_bound:
            return None
        root_idx[0] += 1
        # The order of subtree calls is important because root_idx is updated by the call...
        left_subtree = rebuild_bst_helper(lower_bound, root)
        right_subtree = rebuild_bst_helper(root, upper_bound)
        return BSTNode(root, left_subtree, right_subtree)
    root_idx = [0]
    return rebuild_bst_helper(float("-inf"), float("inf"))

closest = [[1, 2, 4, 25], [3, 6, 9, 12, 15], [8, 17, 24]]
def find_closest_elements_sorted_arrays(sorted_arrays: list[list]) -> int: #O(nlogK) where K is the number of arrays and n is the sum of the arrays...
    min_so_far = float("inf")
    bin_tree = bintrees.RBTree()
    for idx, array in enumerate(sorted_arrays):
        it = iter(array)
        first_min = next(it, None)
        # Add first elements to binary tree... 
        bin_tree.insert((first_min, idx), it)
    
    while True:
        min_value, min_idx = bin_tree.min_key()
        max_value = bin_tree.max_key()[0]
        min_so_far = min(max_value - min_value, min_so_far)
        # Remove the minimum and add the next element in the array...
        it = bin_tree.pop_min()[1]
        next_min = next(it, None)
        # An array is empty...
        if next_min is None:
            return min_so_far
        bin_tree.insert((next_min, min_idx), it)

def generate_first_k_a_b_sqrt(k: int) -> list: #O(klogk)
    # We could also use a min-heap for this...
    candidates = bintrees.RBTree([(ABSqrt2(0, 0), None)])
    result = []
    while len(result) < k:
        next_smallest = candidates.pop_min()[0]
        result.append(next_smallest.val)
        candidates.insert(ABSqrt2(next_smallest.a + 1, next_smallest.b), None)
        candidates.insert(ABSqrt2(next_smallest.a, next_smallest.b + 1), None)
    return result

def generate_first_k_a_b_sqrt_2(k: int) -> list: #O(k)
    result = [ABSqrt2(0, 0)]
    i = j = 0
    for _ in range(1, k):
        cand_i = ABSqrt2(result[i].a + 1, result[i].b )
        cand_j = ABSqrt2(result[j].a , result[j].b + 1 )
        result.append(min(cand_i, cand_j))
        # Last item could be same for both i & j, in such case increment both...
        if result[-1].val == cand_i.val:
            i += 1
        if result[-1].val == cand_j.val:
            j += 1
    return [i.val for i in result]

build_sorted_bst_array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def build_minimum_height_bst_from_sorted_array(array: list[int]) -> BSTNode:
    if not array:
        return None
    root_index = len(array) // 2
    # list slicing is an O(n) operation, we don't really need to create a new copy. We could simply use a start and end pointer... 
    return BSTNode(array[root_index], build_minimum_height_bst_from_sorted_array(array[:root_index]), build_minimum_height_bst_from_sorted_array(array[root_index + 1:]))

range_lookup = (16, 31)

def range_lookup_bst(tree: BSTNode, interval: tuple) -> list[int]:
    def range_lookup_bst_helper(tree: BSTNode):
        if not tree:
            return None
        # tree within the range...
        if interval[0] <= tree.data <= interval[1]:
            range_lookup_bst_helper(tree.left)
            result.append(tree.data)
            range_lookup_bst_helper(tree.right)
        elif interval[0] > tree.data:
            range_lookup_bst_helper(tree.right)
        elif interval[1] < tree.data:
            range_lookup_bst_helper(tree.left)

    result = []
    range_lookup_bst_helper(tree)
    return result

print(range_lookup_bst(A, range_lookup))
# inorder_traversal_recursion(build_minimum_height_bst_from_sorted_array(build_sorted_bst_array))
# print(generate_first_k_a_b_sqrt(7))
# print(generate_first_k_a_b_sqrt_2(7))
# print(find_closest_elements_sorted_arrays(closest))
# print(rebuild_bst_from_pre_order([43, 23, 37, 29, 31, 41, 47, 53]))
# print(find_LCA_BST(A, L, P))
# print(find_k_largest_bst(A, 5))
# print(find_first_greater_than_k(A, 24))
# print(inorder_traversal_is_binary_tree(A))
# print(is_binary_tree_bst(A))
# print(search_bst(A, 2))
# inorder_traversal_recursion(rebuild_bst_from_preorder_O_n([43, 23, 37, 29, 31, 41, 47, 53]))
