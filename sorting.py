from collections import namedtuple

A = [2, 3, 3, 5, 7, 11]
B = [3, 3, 7, 15, 31]
def intersection_of_two_sorted_array(A: list[int], B: list[int]) -> list[int]: #O(m+n)
    i, j, intersection = 0, 0, []

    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            # We don't need a duplicate in the result...
            if i == 0 or A[i] != A[i-1]:
                intersection.append(A[i])
            i, j = i + 1, j + 1
        elif A[i] < B[j]:
            i += 1
        else:
            j += 1
    return intersection

smallest_nonconstructible = [12, 2, 1, 15, 2, 4]
def smallest_nonconstructible_value(A: list[int]) -> int:
    max_constructible = 0
    for i in sorted(A):
        # if iter greater than current sum plus 1, it can't be used to create the next in the sequence...
        if i > max_constructible + 1:
            break
        max_constructible += i
    return max_constructible + 1

Event = namedtuple("Event", ("start", "end"))
EndPoint = namedtuple("EndPoint", ("time", "is_start"))

def find_maximum_running_events(events: list[Event]) -> int: #O(nlogn)
    points = [(EndPoint(event.start, True)) for event in events] + [(EndPoint(event.end, False)) for event in events]
    # start time before end time for collisions...
    points.sort(key=lambda e: (e.time, not e.is_start))
    max_num_running_events, num_running_events = 0, 0
    for point in points:
        if point.is_start:
            # An event was started...
            num_running_events += 1
            max_num_running_events = max(max_num_running_events, num_running_events)
        else:
            # An event was ended...
            num_running_events -= 1
    return max_num_running_events


# print(intersection_of_two_sorted_array(A, B))
print(smallest_nonconstructible_value(smallest_nonconstructible))


