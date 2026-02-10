using Axol.Interpreter.Values;

namespace Axol.Interpreter.BuiltinModules;

internal static class MathBuiltins
{
    private static readonly Random _random = new();

    public static void Register(Environment env)
    {
        env.Define("floor", new BuiltinFunctionVal("floor", args =>
            new IntVal((long)Math.Floor(Axol.Interpreter.Builtins.ToDouble(args[0])))));

        env.Define("ceil", new BuiltinFunctionVal("ceil", args =>
            new IntVal((long)Math.Ceiling(Axol.Interpreter.Builtins.ToDouble(args[0])))));

        env.Define("round", new BuiltinFunctionVal("round", args =>
            new IntVal((long)Math.Round(Axol.Interpreter.Builtins.ToDouble(args[0])))));

        env.Define("sin", new BuiltinFunctionVal("sin", args =>
            new FloatVal(Math.Sin(Axol.Interpreter.Builtins.ToDouble(args[0])))));

        env.Define("cos", new BuiltinFunctionVal("cos", args =>
            new FloatVal(Math.Cos(Axol.Interpreter.Builtins.ToDouble(args[0])))));

        env.Define("pow", new BuiltinFunctionVal("pow", args =>
        {
            var b = Axol.Interpreter.Builtins.ToDouble(args[0]);
            var e = Axol.Interpreter.Builtins.ToDouble(args[1]);
            var result = Math.Pow(b, e);
            // Return int if both args are int and result is integral
            if (args[0] is IntVal && args[1] is IntVal && result == Math.Floor(result) && !double.IsInfinity(result))
                return new IntVal((long)result);
            return new FloatVal(result);
        }));

        env.Define("log", new BuiltinFunctionVal("log", args =>
        {
            var val = Axol.Interpreter.Builtins.ToDouble(args[0]);
            if (args.Count > 1)
            {
                var logBase = Axol.Interpreter.Builtins.ToDouble(args[1]);
                return new FloatVal(Math.Log(val, logBase));
            }
            return new FloatVal(Math.Log(val));
        }));

        env.Define("random", new BuiltinFunctionVal("random", args =>
        {
            if (args.Count == 0)
                return new FloatVal(_random.NextDouble());
            if (args.Count == 1)
                return new IntVal(_random.Next((int)Axol.Interpreter.Builtins.ToLong(args[0])));
            var min = (int)Axol.Interpreter.Builtins.ToLong(args[0]);
            var max = (int)Axol.Interpreter.Builtins.ToLong(args[1]);
            return new IntVal(_random.Next(min, max));
        }));
    }
}
