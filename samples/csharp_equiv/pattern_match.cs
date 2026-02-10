using System;
using System.Linq;
using System.Collections.Generic;

abstract record Shape;
record Circle(double R) : Shape;
record Rect(double W, double H) : Shape;

static class MathModule
{
    public static double Square(double x) => x * x;
    public static double Cube(double x) => x * x * x;
}

class Program
{
    static double Area(Shape shape) => shape switch
    {
        Circle c => 3.14 * c.R * c.R,
        Rect r => r.W * r.H,
        _ => throw new ArgumentException()
    };

    static void Main()
    {
        var nums = new List<int> { 1, 2, 3, 4, 5 };
        var doubled = nums.Select(x => x * 2).ToList();
        var evens = nums.Where(x => x % 2 == 0).ToList();
        var sum = nums.Sum();

        Console.WriteLine($"area circle r=5: {Area(new Circle(5))}");
        Console.WriteLine($"area rect 3x4: {Area(new Rect(3, 4))}");
        Console.WriteLine($"square 7: {MathModule.Square(7)}");
        Console.WriteLine($"doubled: [{string.Join(", ", doubled)}]");
        Console.WriteLine($"evens: [{string.Join(", ", evens)}]");
        Console.WriteLine($"sum: {sum}");
    }
}
