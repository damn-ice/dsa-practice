class NodeList:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next

a = NodeList("a")
b = NodeList("b")
c = NodeList("c")
d = NodeList("d")
e = NodeList("e")
f = NodeList("f")
g = NodeList("g")
h = NodeList("h")
i = NodeList("i")
j = NodeList("j")
k = NodeList("k")


a.next, b.next, c.next, d.next, e.next, f.next, g.next, h.next, i.next, j.next= b, c, d, e, f, g, h, i,j, k

def traverse_linked_list(node: NodeList):
    start = node
    while start:
        print(start.val)
        start = start.next

# traverse_linked_list(a)

def traverse_linked_list_recursive(node: NodeList, sum=0):
    if node == None:
        print(sum)
        return 
    # print(node.val)
    sum += node.val
    traverse_linked_list_recursive(node.next, sum)

# traverse_linked_list_recursive(a)
# li = [0,1,2,3,4,5,6,7]

# This is wrong...
# def reverse_subarray(li, s, f):
#     iteration = 0
#     for i in range(s, f):
#         li[i], li[f-iteration] = li[f-iteration], li[i]
#         iteration+=1
#     print(li)

# reverse_subarray(li, 1, 4)

def has_cycle(head):
    def cycle_len(end):
        start , step = end, 0
        while True:
            step += 1
            start = start.next
            if start is end:
                return step 
    fast=slow=head
    while fast and fast.next and fast.next.next:
        slow, fast = slow.next, fast.next.next
        print(slow, fast)
        print(slow == fast)
        print(slow.val, fast.val)
        if slow is fast:
            # Finds the start of the cycle.
            cycle_len_advanced_iter = head
            for _ in range(cycle_len(slow)):
                cycle_len_advanced_iter = cycle_len_advanced_iter.next
            it = head
            # Both iterators advance in tandem.
            while it is not cycle_len_advanced_iter:
                it = it.next
                cycle_1en_advanced_iter = cycle_len_advanced_iter.next
            return it # iter is the start of cycle.
    return None # No cycle.

print(has_cycle(a))
# print(a, b , c, d, e, f)