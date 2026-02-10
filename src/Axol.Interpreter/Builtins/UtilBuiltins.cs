using Axol.Interpreter.Values;

namespace Axol.Interpreter.BuiltinModules;

internal static class UtilBuiltins
{
    public static void Register(Environment env)
    {
        env.Define("format", new BuiltinFunctionVal("format", args =>
        {
            if (args.Count == 0)
                throw new AxolRuntimeException("format: expected at least 1 argument");

            if (args[0] is StrVal fmt)
            {
                var template = fmt.Value;
                for (int i = 1; i < args.Count; i++)
                    template = template.Replace($"{{{i - 1}}}", args[i].Display());
                return new StrVal(template);
            }
            throw new AxolRuntimeException("format: first argument must be string template");
        }));

        env.Define("assert_eq", new BuiltinFunctionVal("assert_eq", args =>
        {
            if (args.Count < 2)
                throw new AxolRuntimeException("assert_eq: expected 2 arguments");
            if (!Axol.Interpreter.Builtins.ValuesEqual(args[0], args[1]))
            {
                var msg = args.Count > 2 ? args[2].Display() : $"Expected {args[0].Display()} == {args[1].Display()}";
                throw new AssertionFailedException($"assert_eq failed: {msg}");
            }
            return UnitVal.Instance;
        }));
    }
}
