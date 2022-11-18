MAPPING = ('0', '1', 'ABC', 'DEF', 'GHI', 'JKL', 'MNO', 'PQRS', 'TUV', 'WXYZ')

def phone(phone_number):
    def phone_helper(digit):
        if digit == len(phone_number):
            mnemonics.append("".join(partial_mnemonic))
        else:
            for c in MAPPING[int(phone_number[digit])]:
                partial_mnemonic[digit] = c
                phone_helper(digit+1)
    mnemonics, partial_mnemonic = [] , [0] * len(phone_number)
    # O(n4^n)
    phone_helper(0)
    return mnemonics

print(phone("08168910164"))        
print(len(phone("08168910164")))