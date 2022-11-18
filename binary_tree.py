from collections import deque, namedtuple
from unittest import result

class BinaryTree:
    def __init__(self, val) -> None:
        self.val = val
        self.left = None
        self.right = None

a = BinaryTree("a")
b = BinaryTree("b")
c = BinaryTree("c")
d = BinaryTree("d")
e = BinaryTree("e")
f = BinaryTree("f")
# g = BinaryTree("g")

a.left = b
a.right = c
b.left = d
b.right = e
# c.left = g
c.right = f

def traverse_binary_tree(root: BinaryTree) -> None:
    if root:
        print("preorder")
        print(root.val)
        traverse_binary_tree(root.left)
        print("inorder")
        print(root.val)
        traverse_binary_tree(root.right)
        print("postorder")
        print(root.val)

# print(a.left.val)

def traverse_binary_tree_iteration(root: BinaryTree) -> None:
    stack = [root]
    while len(stack) and root:
        cur = stack.pop()
        print(cur.val)
        # if cur.right:
        #     stack.append(cur.right)
        # if cur.left:
        #     stack.append(cur.left)
        cur.right and stack.append(cur.right)
        cur.left and stack.append(cur.left)

def breadth_first_tree_iteration(root: BinaryTree) -> None:
    queue = deque()
    queue.append(root)
    while len(queue) and root:
        cur = queue.popleft()
        print(cur.val)
        cur.left and queue.append(cur.left)
        cur.right and queue.append(cur.right)

def depth_first_search(root: BinaryTree, target: str) -> bool:
    if not root:
        return False
    if root.val == target:
        return True
    return depth_first_search(root.left, target) or depth_first_search(root.right, target)
    
def sum_depth_first_recursion(root: BinaryTree) -> str:
    if root == None:
        return 0
    return root.val + sum_depth_first_recursion(root.left) + sum_depth_first_recursion(root.right)

def min_depth_first_recursion(root: BinaryTree) -> int:
    if root == None:
        return float("inf")
    return min(root.val, min_depth_first_recursion(root.left), min_depth_first_recursion(root.right))
def max_path_depth_first_recursion(root: BinaryTree) -> int:
    if root == None:
        # This would be wrong if we have a negative value... use float("-inf")
        return 0
    return root.val + max(max_path_depth_first_recursion(root.left), max_path_depth_first_recursion(root.right))
# breadth_first_tree_iteration(a)

# A = BinaryTree(314)
# B = BinaryTree(6)
# C = BinaryTree(271)
# D = BinaryTree(28)
# E = BinaryTree(0)
# F = BinaryTree(561)
# G = BinaryTree(3)
# H = BinaryTree(17)
# I = BinaryTree(6)
# J = BinaryTree(2)
# K = BinaryTree(1)
# L = BinaryTree(401)
# M = BinaryTree(641)
# N = BinaryTree(257)
# O = BinaryTree(271)
# P = BinaryTree(28)
A = BinaryTree("A")
B = BinaryTree("B")
C = BinaryTree("C")
D = BinaryTree("D")
E = BinaryTree("E")
F = BinaryTree("F")
G = BinaryTree("G")
H = BinaryTree("H")
I = BinaryTree("I")
J = BinaryTree("J")
K = BinaryTree("K")
L = BinaryTree("L")
M = BinaryTree("M")
N = BinaryTree("N")
O = BinaryTree("O")
P = BinaryTree("P")
A.left = B
A.right = I
B.left = C
B.right = F
C.left = D
C.right = E
F.right = G
G.left = H
I.left = J
I.right = O
J.right = K
K.left = L
K.right = N
L.right = M
O.right = P

def is_balanced_binary_tree(root: BinaryTree) -> bool:
    BalancedStatusWithHeight = namedtuple("BalancedStatusWithHeight", ("balanced", "height"))
    def check_height(root: BinaryTree) -> BalancedStatusWithHeight:
        if not root:
            return BalancedStatusWithHeight(True, -1)
        left_result = check_height(root.left)
        if not left_result.balanced:
            return BalancedStatusWithHeight(False, 0)
        right_result = check_height(root.right)
        if not right_result.balanced:
            return BalancedStatusWithHeight(False, 0)
        is_balanced = abs(left_result.height - right_result.height) <= 1
        height = max(left_result.height, right_result.height) + 1
        return BalancedStatusWithHeight(is_balanced, height)
    return check_height(root).balanced

def is_symmetric(root: BinaryTree) -> bool:
    def check_symmetry(subtree_0: BinaryTree, subtree_1: BinaryTree) -> bool:
        if not subtree_0 and not subtree_1:
            return True
        elif subtree_1 and subtree_0:
            return (subtree_0 == subtree_1 and check_symmetry(subtree_0.right, subtree_1.left) and check_symmetry(subtree_1.right, subtree_0.left))
        return False
    return not root or check_symmetry(root.left, root.right)

def lca(root: BinaryTree, node1: BinaryTree, node2: BinaryTree) -> BinaryTree:
    Status = namedtuple("Status", ("num_nodes", "ancestor"))
    def lca_helper(root: BinaryTree, node1: BinaryTree, node2: BinaryTree) -> BinaryTree:
        if not root:
            return Status(0, None)  #Base Case...
        left_result = lca_helper(root.left, node1, node2)
        if left_result.num_nodes == 2:
            return left_result
        right_result = lca_helper(root.right, node1, node2)
        if right_result.num_nodes == 2:
            return right_result
        num_nodes = left_result.num_nodes + right_result.num_nodes + int(root is node1 ) + int(root is node2)
        return Status(num_nodes, root if num_nodes == 2 else None)
    return lca_helper(root, node1, node2).ancestor

def lca_with_parent(root: BinaryTree, node1: BinaryTree, node2: BinaryTree) -> BinaryTree:
    def get_depth(node: BinaryTree) -> int:
        depth = 0
        while node:
            node = node.parent
            depth += 1
        return depth
    # Get the target nodes depth...
    depth1, depth2 = get_depth(node1), get_depth(node2)
    if depth2 > depth1:
        node1, node2 = node2, node1
    depth_diff = abs(depth2 - depth1)
    # Ascend to make target nodes at same depth...
    while depth_diff:
        node = node.parent
        depth_diff -= 1
    # Ascend both nodes...
    while node1 is not node2:
        node1, node2 = node1.parent, node2.parent
    return node1

A1 = BinaryTree(1)
B1 = BinaryTree(1)
C1 = BinaryTree(0)
D1 = BinaryTree(0)
E1 = BinaryTree(1)
F1 = BinaryTree(1)
G1 = BinaryTree(1)
H1 = BinaryTree(0)
I1 = BinaryTree(1)
J1 = BinaryTree(0)
K1 = BinaryTree(0)
L1 = BinaryTree(1)
M1 = BinaryTree(1)
N1 = BinaryTree(0)
O1 = BinaryTree(0)
P1 = BinaryTree(0)

A1.left = B1
A1.right = I1
B1.left = C1
B1.right = F1
C1.left = D1
C1.right = E1
F1.right = G1
G1.left = H1
I1.left = J1
I1.right = O1
J1.right = K1
K1.left = L1
K1.right = N1
L1.right = M1
O1.right = P1

def sum_root_to_leaf(tree: BinaryTree, partial_path_sum=0):
    if not tree:
        return 0
    # Converts from binary to base 10...
    partial_path_sum = partial_path_sum * 2 + tree.val
    if not tree.left and not tree.right:
        return partial_path_sum
    return (sum_root_to_leaf(tree.left, partial_path_sum) + sum_root_to_leaf(tree.right, partial_path_sum))

def preorder_traversal(tree: BinaryTree):
    stack, result = [], []
    if not tree:
        return result
    stack.append(tree)
    while stack:
        current = stack.pop()
        result.append(current.val)
        current.right and stack.append(current.right)
        current.left and stack.append(current.left)
    return result

def inorder_traversal_O_1_space(tree: BinaryTree) -> list[str]:
    prev, result = None, []
    while tree:
        if prev is tree.parent:
            if tree.left:
                next_ = tree.left
            else:
                result.append(tree.val)
                next_ = tree.parent or tree.right
        elif tree.left is prev:
            result.append(tree.val)
            next_ = tree.right or tree.parent
        else: 
            next_ = tree.parent
        prev, tree = tree, next_

    return result

def exterior_binary_tree(tree: BinaryTree) -> list[str]:
    def is_leaf(node: BinaryTree) -> bool:
        return not node.left and not node.right
    def left_boundary_and_leaves(subtree: BinaryTree, is_boundary: bool) -> list[str]:
        if not subtree:
            return []
        return (([subtree.val] if is_boundary or is_leaf(subtree) else []) + left_boundary_and_leaves(subtree.left, is_boundary) + left_boundary_and_leaves(subtree.right, is_boundary and not subtree.left))

    def right_boundary_and_leaves(subtree: BinaryTree, is_boundary: bool) -> list[str]:
        if not subtree:
            return []
        return (([subtree.val] if is_boundary or is_leaf(subtree) else []) + right_boundary_and_leaves(subtree.right, is_boundary) + right_boundary_and_leaves(subtree.left, is_boundary and not subtree.right))
    return [tree.val] + left_boundary_and_leaves(tree.left, is_boundary=True) + right_boundary_and_leaves(tree.right, is_boundary=True)

# INCOMPLETE...

# def binary_tree_from_inorder_preorder(preorder: list, inorder: list):
#     node_to_inorder_idx = {data: i for i, data in enumerate(inorder)}
#     def binary_tree_from_inorder_preorder_helper(preorder_start: int, preorder_end: int, inorder_start: int, inorder_end:int):
#         if preorder_end <= preorder_start or inorder_end <= inorder_start:
#             return None
#         root_inorder_idx = node_to_inorder_idx[preorder[preorder_start]]
#         left_subtree_size = root_inorder_idx - inorder_start

# traverse_binary_tree(A1)
# print(depth_first_search(a, "k"))
# print(sum_depth_first_recursion(a))
# print(min_depth_first_recursion(a))
# print(max_path_depth_first_recursion(a))
# print(is_balanced_binary_tree(a))
# print(is_symmetric(A))
# print(lca(A, F, H ))
# print(sum_root_to_leaf(B1))
# print(preorder_traversal(A))
print(exterior_binary_tree(A))