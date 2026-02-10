using System;
using System.Diagnostics;

class Program
{
    static int SafeDiv(int a, int b)
    {
        Debug.Assert(b != 0, "precondition: b != 0");
        int r = a / b;
        Debug.Assert(r >= 0, "postcondition: r >= 0");
        return r;
    }

    static void Main()
    {
        Debug.Assert(SafeDiv(10, 2) == 5);
        Debug.Assert(SafeDiv(100, 10) == 10);
        Console.WriteLine("All contract tests passed");
    }
}
