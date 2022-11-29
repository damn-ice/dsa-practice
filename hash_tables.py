from collections import Counter, OrderedDict, namedtuple

can = "abobo"
cant = "aboboo"
def can_form_palindrome(s: str) -> bool:
    return sum(i % 2 for i in Counter(s).values()) <= 1 #O(n)

def is_letter_constructible_from_magazine_pythonic(letter: str, magazine: str) -> bool:
    return (not Counter(letter) - Counter(magazine))


class LRUcache:
    def __init__(self, capacity: int):
        self._isbn_cache = OrderedDict() # Regular dict as from python3.7 they track order...
        self._capacity = capacity
    
    def lookup(self, isbn: str):
        """Checks if item in cache and update"""
        if isbn not in self._isbn_cache:
            return -1
        price = self._isbn_cache.pop(isbn)
        self._isbn_cache[isbn] = price
        return price
    
    def insert(self, isbn: str, price: str):
        """Insert new isbn if cache greater than capacity remove LRU"""
        if isbn in self._isbn_cache:
            # if it exist don't update price...
            price = self._isbn_cache.pop(isbn)
        elif self._capacity >= len(self._isbn_cache):
            self._isbn_cache.popitem(last=False)
        self._isbn_cache[isbn] = price

    def delete(self, isbn: str)-> bool:
        """Delete item from cache and returns boolean"""
        return self._isbn_cache.pop(isbn, None) is not None

find_nearest = ["All", "work", "and", "no", "play", "makes", "for", "no", "work", "no", "fun", "and", "no", "results"]
def find_nearest_repetition(paragraph: list[str]) -> int:
    """Returns the nearest repetion in a list of words, O(n) time complexity and O(D) space complexity where D is the number of distinct words"""
    nearest, word_latest_index = float("inf"), {}
    for i, word in enumerate(paragraph):
        if word in word_latest_index:
            latest_word = word_latest_index[word]
            nearest = min(nearest, i - latest_word)
        # Always update ...
        word_latest_index[word] = i
    return nearest if nearest != float("inf") else -1 

SubArray = namedtuple("SubArray", ("start", "end"))
find_smallest_paragraph = ["apple", "banana", "apple", "apple", "dog", "cat", "apple", "dog", "banana", "cat", "dog"]
find_smallest_keyword = ["apple", "cat"]

def find_smallest_subarray(paragraph: list[str], keywords: list[str]) -> SubArray:
    result = SubArray(-1, -1)
    remaining_to_cover = len(keywords)
    keywords_to_cover = Counter(keywords)
    left = 0
    for right, p in enumerate(paragraph):
        if p in keywords:
            keywords_to_cover[p] -= 1
            if keywords_to_cover[p] >= 0:
                remaining_to_cover -= 1
        
        while remaining_to_cover == 0:
            if result == (-1, -1) or right - left < result.end - result.start:
                result = SubArray(left, right)
            pl = paragraph[left]
            if pl in keywords:
                keywords_to_cover[pl] += 1
                if keywords_to_cover[pl] > 0:
                    remaining_to_cover += 1
            left += 1
    return result

find_smallest_paragraph_sequentially = ["joel", "apple", "banana", "cat", "pear", "apple"]

find_smallest_keyword_sequentially = ["banana", "cat", "apple"]

def find_smallest_sequentially_covering_subset(paragraph: list[str], keywords: list[str]) -> SubArray:
    keywords_to_idx = {keyword: i for i, keyword in enumerate(keywords)}
    shortest_subarray_length = [float("inf")] * len(keywords)
    latest_occurence = [-1] * len(keywords)
    shortest_distance = float("inf")
    result = SubArray(-1, -1)

    for i, word in enumerate(paragraph):
        if word in keywords:
            keyword_idx = keywords_to_idx[word]
            if keyword_idx == 0:
                shortest_subarray_length[keyword_idx] = 1
            # Only if the previous has been found...
            elif shortest_subarray_length[keyword_idx - 1] != float("inf"):
                distance_to_previous_keyword = i - latest_occurence[keyword_idx - 1]
                shortest_subarray_length[keyword_idx] = distance_to_previous_keyword + shortest_subarray_length[keyword_idx - 1]
            latest_occurence[keyword_idx] = i

            if (keyword_idx == len(keywords) - 1 and shortest_subarray_length[-1] < shortest_distance):
                shortest_distance = shortest_subarray_length[-1]
                result = SubArray(i - shortest_distance + 1, i)
    return result

longest_subarray_distinct =  ["f", "s", "f", "e", "t", "w", "e", "s", "w", "e"]
def longest_subarray_with_distinct_entries(A: list[str]) -> int:
    most_recent_occurence = {}
    result = longest_dup_start_idx = 0
    for idx, item in enumerate(A):
        if item in most_recent_occurence:
            dup_idx = most_recent_occurence[item]
            if dup_idx >= longest_dup_start_idx:
                result = max(result, idx - longest_dup_start_idx)
                longest_dup_start_idx = dup_idx + 1
        most_recent_occurence[item] = idx
    # if all elements are distinct, second argument is returned...
    return max(result, len(A) - longest_dup_start_idx)

longest_contained = [3,-2,7,9,8,1,2,0,-1,5,8]
longest_contained2 = [10,5,3,11,6,100,4]
def longest_contained_range(A: list[int]) -> int:
    unprocessed_entries = set(A)
    max_range = 0
    while unprocessed_entries:
        a = unprocessed_entries.pop()
        lower_bound = a - 1
        while lower_bound in unprocessed_entries:
            unprocessed_entries.remove(lower_bound)
            lower_bound -= 1
        upper_bound = a + 1
        while upper_bound in unprocessed_entries:
            unprocessed_entries.remove(upper_bound)
            upper_bound += 1
        max_range = max(max_range, upper_bound - lower_bound - 1)
    return max_range

sentence_find = "amanaplanacanal"
word_find = ["can", "apl", "ana"]
def find_all_substrings(sentence: str, words: list[str]) -> list[int]:
    def match_all_words_in_dict(start: int) -> bool:
        counter_frequency = Counter()
        for i in range(start, start + unit_size * len(words), unit_size):
            curr_word = sentence[i : i + unit_size]
            it = word_to_freq[curr_word]
            if it == 0:
                return False
            counter_frequency[curr_word] += 1
            # curr_word appears more times than required...
            if counter_frequency[curr_word] > it:
                return False
        return True
    word_to_freq = Counter(words)
    unit_size = len(words[0])
    return [i for i in  range(len(sentence) - unit_size * len(words) + 1) if match_all_words_in_dict(i)]

def test_collatz(n: int) -> bool:
    # Store verified odd numbers...
    verified_number = set()
    for i in range(3, n+1):
        sequence = set()
        test_i = i
        # test_i gets below i means we already tested it... i moves from 0 to infinity...
        while test_i >= i:
            if test_i in sequence:
                return False # Loop encountered...
            sequence.add(test_i)
            if test_i % 2:
                if test_i in verified_number:
                    break # we've already verified it...
                verified_number.add(test_i)
                test_i = 3 * test_i + 1
            else:
                test_i //= 2
    return True
            

# print(can_form_palindrome(can))
# print(can_form_palindrome(cant))
# print(find_nearest_repetition(find_nearest))
# print(find_smallest_subarray(find_smallest_paragraph, find_smallest_keyword))
# print(find_smallest_sequentially_covering_subset(find_smallest_paragraph_sequentially, find_smallest_keyword_sequentially))
# print(longest_subarray_with_distinct_entries(longest_subarray_distinct))
# print(longest_contained_range(longest_contained))
# print(find_all_substrings(sentence_find, word_find))
print(test_collatz(1855))