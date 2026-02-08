fruits = ["apple", "banana", "orange"]

for i, fruit in enumerate(fruits):
    if i == 1:
        print(fruit)


greet = lambda: "Hello!"

greet()


name = ["Alice"]
age = [30]
for name in dict(zip(name, age)).keys():
    print(name)

lol = {"name": "Alice", "age": 30}

lol = {"name": "Alice", "age": 30}

for key, value in lol.items():
    if key == "name":
        print(value)

text = "python programming"
a = [1, 2, 3, 4, 5]
for i, _ in enumerate(a):
    print(2 * i + 2)
