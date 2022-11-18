def create_function(x):
    return lambda y: y + x

# increment_by_i = [create_function(i) for i in range(10)]
increment_by_i = [lambda x: x+i for i in range(10)]
# print(increment_by_i)
def hi(r=[]):
    r.append(1)
    return r
print(hi())
print(hi())
# print(increment_by_i[0](4))