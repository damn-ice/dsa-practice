import bisect
import collections
import operator
import random

sorted_array = [-14, -10, 2, 108, 108, 243, 285, 285, 285, 401]

sorted_array_2 = [-2, 0, 2, 3, 6, 7, 9]

def first_occurence_of_k(li: list, k: int) -> int:
    index = bisect.bisect_left(li, k) #O(logn)
    return index if li[index] == k else -1

def first_occurence_of_k_non_pythonic(li: list, k:int) -> int:
    left, right, result = 0, len(li) - 1, -1
    while left <= right:
        mid = (left + right) // 2
        if li[mid] < k:
            left = mid + 1
        elif li[mid] == k:
            result = mid
            right = mid - 1
        else:
            right = mid - 1
    return result

def search_entry_equal_index(li: list) -> int:
    left, right, result = 0, len(li) - 1, -1
    while left <= right:
        mid = (left + right) // 2
        difference = li[mid] - mid
        if difference == 0:
            result = mid
            return result
        elif difference > 0:
            right = mid - 1
        elif difference < 0:
            left = mid + 1
    return result

def square_root(k: int) -> int:
    left, right = 0, k
    while left <= right:
        mid = (left + right) // 2
        mid_squared = mid * mid
        if mid_squared > k:
            right = mid - 1
        else:
            left = mid + 1
    return left - 1

A = [[-1, 3], [1,5], [3,8], [4,8]]
def matrix_search_2d(A: list[list], x: int) -> bool: #o(m+n)
    row, col = 0, len(A[0]) - 1
    while row < len(A) and col >= 0:
        if A[row][col] == x:
            return True
        elif A[row][col] > x:
            col -= 1
        else:
            row += 1
    return False

min_max = [5, 1, 2, 3, 5, 16, 4, 8]
MinMax = collections.namedtuple("MinMax", ("smallest", "largest"))
def minmax(A: list) -> MinMax:
    minimum = maximum = A[0]
    for i in range(1, len(A)):
        if A[i] < minimum:
            minimum = A[i]
        if A[i] > maximum:
            maximum = A[i]
    return MinMax(minimum, maximum)


k_largest = [4, 5, 8, 6, 0, 3, 1]
def kth_largest(k: int, A: list) -> int:
    def find_kth(comp):
        def partition_around_pivot(left: int, right: int, pivot_index: int) -> int:
            pivot_value = A[pivot_index]
            new_pivot_index = left
            A[right], A[pivot_index] = A[pivot_index], A[right]
            for i in range(left, right):
                if comp(A[i], pivot_value):
                    A[i], A[new_pivot_index] = A[new_pivot_index], A[i]
                    new_pivot_index += 1
            A[new_pivot_index], A[right] = A[right], A[new_pivot_index]
            return new_pivot_index
        left, right = 0, len(A) - 1
        while left <= right:
            pivot_index = random.randint(left, right)
            new_pivot_index = partition_around_pivot(left, right, pivot_index)
            if new_pivot_index == k - 1:
                return A[new_pivot_index]
            elif new_pivot_index < k -1:
                left = new_pivot_index + 1
            elif new_pivot_index > k - 1:
                right = new_pivot_index - 1
    return find_kth(operator.gt)
print(kth_largest(3, k_largest))

# print(minmax(min_max))
# print(matrix_search_2d(A, -1))
# print("joel")
# print(square_root(250))
# print(first_occurence_of_k(sorted_array, 108))
# print(first_occurence_of_k_non_pythonic(sorted_array, 108))
# print(search_entry_equal_index(sorted_array_2))