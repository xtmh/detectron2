
s = 'this is a pen'
print(s)
# this is a pen

l = [0, 1, 2]
print(l)
# [0, 1, 2]

print(l[0])
# 0

d = {'a': 0, 'b': 1, 'c': 2}
print(d)
# {'a': 0, 'b': 1, 'c': 2}

print(d['b'])
# 1

f = 1.00000
print(f)
# 1.0

print('abc', end='---')
print('xyz')
# abc---xyz

print('abc', end='')
print('xyz')
# abcxyz

i=100
print('apple', i, 0.123, sep='\n')
# apple
# 100
# 0.123

#リスト
l = [0, 1, 2]
print('list  ', end='')
print(l)
# [0, 1, 2]

#タプル
l = (0, 1, 2)
print('tuple ', end='')
print(l)
# (0, 1, 2)

print(*l)  # => print(0, 1, 2)
# 0 1 2

print(*l, sep='')
# 012

print(*l, sep='-')
# 0-1-2

s = 'Alice'
i = 25
print('Alice is %d years old' %i)
# Alice is 25 years old

print('%s is %d years old' %(s, i))
# Alice is 25 years old

#辞書（dict）・・・キーと値のペア
d = {'Yamada': 30, 'Suzuki': 40, 'Tanaka': 80}
for a, b in d.items():
#    print(a, b)            # Tanaka 80, Yamada 30, Suzuki 40
    print(a, b, end=", ")            # Tanaka 80, Yamada 30, Suzuki 40

x = 10 if y 