import heapq
import operator
import random
from itertools import islice

item = [3, -1, 2, 6, 4, 5, 8]

def sort_aproximately_sorted(seq: list, k: int):
    result = []
    min_heap = []

    # Add k items to heap...
    for i in islice(seq, k):
        heapq.heappush(min_heap, i)
    # Add remaining and pop.. nlog(k)
    for i in seq[k:]:
        smallest = heapq.heappushpop(min_heap, i)
        result.append(smallest)
    while min_heap: #klogk
        smallest = heapq.heappop(min_heap)
        result.append(smallest)
    return result

median_input = [1, 0, 3, 5, 2, 0, 1]
def running_median(seq: list) -> list:
    # max_heap stores the smaller half seen so far...
    # min_heap stores the bigger half seen so far...
    min_heap, max_heap, result = [], [], []
    for i in seq:
        heapq.heappush(max_heap, -heapq.heappushpop(min_heap, i))
        if len(max_heap) > len(min_heap):
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        result.append(0.5 * (min_heap[0] + (-max_heap[0])) if len(min_heap) == len(max_heap) else min_heap[0])
    return result
    
def k_largest_binary_heap(A, k: int) -> list:
    if k <= 0:
        return []
    result, candidate_max_heap = [], []
    candidate_max_heap.append((-A[0], 0)) 
    for _ in range(k):
        candidate_idx = candidate_max_heap[0][1]
        result.append(-heapq.heappop(candidate_max_heap)[0])
        left_idx = 2 * candidate_idx + 1
        if left_idx < len(A):
            heapq.heappush(candidate_max_heap, (-A[left_idx], left_idx))

        right_idx = 2 * candidate_idx + 2
        if right_idx < len(A):
            heapq.heappush(candidate_max_heap, (-A[right_idx], right_idx))
    return result

# print(sort_aproximately_sorted(item, 2))
# print(running_median(median_input))