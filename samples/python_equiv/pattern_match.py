from dataclasses import dataclass
from typing import Union
import math

@dataclass
class Circle:
    r: float

@dataclass
class Rect:
    w: float
    h: float

Shape = Union[Circle, Rect]

def area(shape: Shape) -> float:
    match shape:
        case Circle(r=r):
            return 3.14 * r * r
        case Rect(w=w, h=h):
            return w * h

def square(x):
    return x * x

def cube(x):
    return x * x * x

nums = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, nums))
evens = list(filter(lambda x: x % 2 == 0, nums))
total = sum(nums)

print("area circle r=5:", area(Circle(5)))
print("area rect 3x4:", area(Rect(3, 4)))
print("square 7:", square(7))
print("doubled:", doubled)
print("evens:", evens)
print("sum:", total)
