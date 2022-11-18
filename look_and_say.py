"312211" "13112221"
def look_say(number: int) -> str:
    s = "1"
    def next_say(s):
        result, i = [], 0
        while i < len(s):
            count = 1
            while i + 1 > len(s) and s[i] == s[i + 1]:
                count += 1
                i += 1
                

    for _ in (1, number):
        s = next_say(s)
    return s