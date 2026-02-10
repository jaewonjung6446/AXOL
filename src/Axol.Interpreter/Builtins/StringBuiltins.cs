using Axol.Interpreter.Values;

namespace Axol.Interpreter.BuiltinModules;

internal static class StringBuiltins
{
    public static void Register(Environment env)
    {
        env.Define("upper", new BuiltinFunctionVal("upper", args =>
        {
            if (args[0] is StrVal s)
                return new StrVal(s.Value.ToUpperInvariant());
            throw new AxolRuntimeException("upper: expected string");
        }));

        env.Define("lower", new BuiltinFunctionVal("lower", args =>
        {
            if (args[0] is StrVal s)
                return new StrVal(s.Value.ToLowerInvariant());
            throw new AxolRuntimeException("lower: expected string");
        }));

        env.Define("split", new BuiltinFunctionVal("split", args =>
        {
            if (args[0] is StrVal s && args[1] is StrVal sep)
            {
                var parts = s.Value.Split(sep.Value);
                return new ListVal(parts.Select(p => (AxolValue)new StrVal(p)).ToList());
            }
            throw new AxolRuntimeException("split: expected (split string separator)");
        }));

        env.Define("join", new BuiltinFunctionVal("join", args =>
        {
            if (args[0] is StrVal sep && args[1] is ListVal list)
            {
                var strs = list.Items.Select(i => i.Display());
                return new StrVal(string.Join(sep.Value, strs));
            }
            throw new AxolRuntimeException("join: expected (join separator list)");
        }));

        env.Define("trim", new BuiltinFunctionVal("trim", args =>
        {
            if (args[0] is StrVal s)
                return new StrVal(s.Value.Trim());
            throw new AxolRuntimeException("trim: expected string");
        }));

        env.Define("replace", new BuiltinFunctionVal("replace", args =>
        {
            if (args[0] is StrVal s && args[1] is StrVal old && args[2] is StrVal @new)
                return new StrVal(s.Value.Replace(old.Value, @new.Value));
            throw new AxolRuntimeException("replace: expected (replace string old new)");
        }));

        env.Define("starts_with", new BuiltinFunctionVal("starts_with", args =>
        {
            if (args[0] is StrVal s && args[1] is StrVal prefix)
                return new BoolVal(s.Value.StartsWith(prefix.Value));
            throw new AxolRuntimeException("starts_with: expected (starts_with string prefix)");
        }));

        env.Define("ends_with", new BuiltinFunctionVal("ends_with", args =>
        {
            if (args[0] is StrVal s && args[1] is StrVal suffix)
                return new BoolVal(s.Value.EndsWith(suffix.Value));
            throw new AxolRuntimeException("ends_with: expected (ends_with string suffix)");
        }));

        env.Define("slice", new BuiltinFunctionVal("slice", args =>
        {
            if (args[0] is StrVal s)
            {
                var start = (int)Axol.Interpreter.Builtins.ToLong(args[1]);
                if (args.Count > 2)
                {
                    var end = (int)Axol.Interpreter.Builtins.ToLong(args[2]);
                    end = Math.Min(end, s.Value.Length);
                    start = Math.Max(0, start);
                    return new StrVal(s.Value[start..end]);
                }
                start = Math.Max(0, start);
                return new StrVal(s.Value[start..]);
            }
            if (args[0] is ListVal list)
            {
                var start = (int)Axol.Interpreter.Builtins.ToLong(args[1]);
                if (args.Count > 2)
                {
                    var end = (int)Axol.Interpreter.Builtins.ToLong(args[2]);
                    end = Math.Min(end, list.Items.Count);
                    start = Math.Max(0, start);
                    return new ListVal(list.Items.GetRange(start, end - start));
                }
                start = Math.Max(0, start);
                return new ListVal(list.Items.GetRange(start, list.Items.Count - start));
            }
            throw new AxolRuntimeException("slice: expected string or list");
        }));
    }
}
