def add_one_to_list(array: list) -> list:
    array[-1] += 1
    for i in reversed(range(1, len(array))):
        if array[i] != 10:
            break
        array[i] = 0
        array[i - 1] += 1
    if array[0] == 10:
        array[0] = 1
        array.append(0)
    return array

def can_reach_end(A: list) -> bool:
    """[3,3,1,0,2,0,1]"""
    max_reach, last_index = 0, len(A) - 1
    i = 0
    while i <= max_reach and max_reach < last_index:
        max_reach = max(max_reach, i + A[i])
        i += 1
    return max_reach >= last_index

def buy_and_sell_stock_once(prices: list) -> int:
    min_price_so_far, max_profit = float("inf"), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return  max_profit

def apply_permutation(perm: list, A: list) -> list:
    """perm = [2013], A = [abcd], return=[bcad] """
    for i in range(len(A)):
        next_iteration = i
        while perm[next_iteration] >= 0:
            A[i], A[perm[next_iteration]] = A[perm[next_iteration]], A[i]
            # Store the next permutation...
            temp = perm[next_iteration]
            perm[next_iteration] -= len(perm)
            next_iteration = temp
    return A

def intToRoman(num):
    """
    :type num: int
    :rtype: str
    """
    result = []
    integer_to_roman = {1000: "M", 900: "CM", 500: "D", 400: "CD", 100: "C", 90: "XC", 50: "L", 40: "XL", 10: "X", 9: "IX", 5: "V", 4: "IV", 1: "I"}
    while num > 0:
        current = 0
        # we need to iterate through the dictionary and break when the iter is less than or equal to num
        # subtract the iter value from num, append the value of iter to result and continue the loop,...
        # After the while loop is done, we return the joined the result array...
        for integer in integer_to_roman:
            if integer <= num:
                current = integer 
                break
        result.append(integer_to_roman[current])
        num -= current
    return "".join(result)

print(intToRoman(58))
# print(can_reach_end([3,3,1,0,2,0,1]))
# print(buy_and_sell_stock_once([310, 315, 275, 295, 260, 270, 290, 230, 255, 250]))
# print(apply_permutation([2,0,1,3], ["a", "b", "c", "d"]))
# print(add_one_to_list([9,9,9])) 
# print(add_one_to_list([3, 5, 3]))