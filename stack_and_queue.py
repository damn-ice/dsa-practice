
def well_formedness(characters: str) -> bool:
    storage, lookup_right = [], {"(": ")", "{": "}", "[": "]"}
    for character in characters:
        if character in lookup_right:
            storage.append(character)
        elif not lookup_right or lookup_right[storage.pop()] != character:
            return False
    return not storage


print(well_formedness("[[()]]"))