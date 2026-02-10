using System;

class Program
{
    static int Fib(int n)
    {
        if (n <= 1) return n;
        return Fib(n - 1) + Fib(n - 2);
    }

    static void Main()
    {
        for (int i = 0; i < 10; i++)
            Console.WriteLine(Fib(i));
    }
}
