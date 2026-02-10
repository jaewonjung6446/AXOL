using Axol.Interpreter.BuiltinModules;
using Axol.Interpreter.Values;

namespace Axol.Interpreter;

public delegate AxolValue FnCaller(AxolValue fn, IReadOnlyList<AxolValue> args);

public static class Builtins
{
    public static void Register(Environment env, TextWriter? output = null, FnCaller? callFn = null)
    {
        var writer = output ?? Console.Out;

        env.Define("print", new BuiltinFunctionVal("print", args =>
        {
            writer.WriteLine(string.Join(" ", args.Select(a => a.Display())));
            return UnitVal.Instance;
        }));

        env.Define("len", new BuiltinFunctionVal("len", args =>
        {
            var arg = args[0];
            return arg switch
            {
                ListVal l => new IntVal(l.Items.Count),
                StrVal s => new IntVal(s.Value.Length),
                MapVal m => new IntVal(m.Entries.Count),
                _ => throw new AxolRuntimeException($"len: unsupported type {arg.GetType().Name}")
            };
        }));

        env.Define("max", new BuiltinFunctionVal("max", args =>
        {
            if (args.Count == 1 && args[0] is ListVal list)
            {
                var nums = list.Items.Cast<IntVal>().Select(v => v.Value);
                return new IntVal(nums.Max());
            }
            var a = ToLong(args[0]);
            var b = ToLong(args[1]);
            return new IntVal(Math.Max(a, b));
        }));

        env.Define("min", new BuiltinFunctionVal("min", args =>
        {
            if (args.Count == 1 && args[0] is ListVal list)
            {
                var nums = list.Items.Cast<IntVal>().Select(v => v.Value);
                return new IntVal(nums.Min());
            }
            var a = ToLong(args[0]);
            var b = ToLong(args[1]);
            return new IntVal(Math.Min(a, b));
        }));

        env.Define("abs", new BuiltinFunctionVal("abs", args =>
        {
            return args[0] switch
            {
                IntVal iv => new IntVal(Math.Abs(iv.Value)),
                FloatVal fv => new FloatVal(Math.Abs(fv.Value)),
                _ => throw new AxolRuntimeException("abs: expected number")
            };
        }));

        env.Define("sqrt", new BuiltinFunctionVal("sqrt", args =>
        {
            var val = ToDouble(args[0]);
            return new FloatVal(Math.Sqrt(val));
        }));

        env.Define("push", new BuiltinFunctionVal("push", args =>
        {
            if (args[0] is ListVal list)
            {
                list.Items.Add(args[1]);
                return list;
            }
            throw new AxolRuntimeException("push: expected list");
        }));

        env.Define("pop", new BuiltinFunctionVal("pop", args =>
        {
            if (args[0] is ListVal list && list.Items.Count > 0)
            {
                var last = list.Items[^1];
                list.Items.RemoveAt(list.Items.Count - 1);
                return last;
            }
            throw new AxolRuntimeException("pop: expected non-empty list");
        }));

        env.Define("str", new BuiltinFunctionVal("str", args =>
            new StrVal(args[0].Display())));

        env.Define("int", new BuiltinFunctionVal("int", args =>
        {
            return args[0] switch
            {
                IntVal iv => iv,
                FloatVal fv => new IntVal((long)fv.Value),
                StrVal sv => new IntVal(long.Parse(sv.Value)),
                BoolVal bv => new IntVal(bv.Value ? 1 : 0),
                _ => throw new AxolRuntimeException("int: cannot convert")
            };
        }));

        env.Define("float", new BuiltinFunctionVal("float", args =>
        {
            return args[0] switch
            {
                FloatVal fv => fv,
                IntVal iv => new FloatVal(iv.Value),
                StrVal sv => new FloatVal(double.Parse(sv.Value, System.Globalization.CultureInfo.InvariantCulture)),
                _ => throw new AxolRuntimeException("float: cannot convert")
            };
        }));

        env.Define("type", new BuiltinFunctionVal("type", args =>
        {
            return new StrVal(args[0] switch
            {
                IntVal => "i",
                FloatVal => "f",
                StrVal => "s",
                BoolVal => "b",
                ListVal => "*",
                MapVal => "%",
                FunctionVal => "fn",
                BuiltinFunctionVal => "fn",
                StructVal sv => sv.TypeName,
                NilVal => "n",
                UnitVal => "u",
                _ => "unknown"
            });
        }));

        env.Define("input", new BuiltinFunctionVal("input", args =>
        {
            if (args.Count > 0)
                Console.Write(args[0].Display());
            var line = Console.ReadLine() ?? "";
            return new StrVal(line);
        }));

        env.Define("range", new BuiltinFunctionVal("range", args =>
        {
            long start, end, step;
            if (args.Count == 1)
            {
                start = 0; end = ToLong(args[0]); step = 1;
            }
            else if (args.Count == 2)
            {
                start = ToLong(args[0]); end = ToLong(args[1]); step = 1;
            }
            else
            {
                start = ToLong(args[0]); end = ToLong(args[1]); step = ToLong(args[2]);
            }
            var items = new List<AxolValue>();
            if (step > 0)
                for (long i = start; i < end; i += step) items.Add(new IntVal(i));
            else
                for (long i = start; i > end; i += step) items.Add(new IntVal(i));
            return new ListVal(items);
        }));

        env.Define("concat", new BuiltinFunctionVal("concat", args =>
        {
            return args[0] switch
            {
                StrVal s1 when args[1] is StrVal s2 => new StrVal(s1.Value + s2.Value),
                ListVal l1 when args[1] is ListVal l2 => new ListVal(new List<AxolValue>(l1.Items.Concat(l2.Items))),
                _ => throw new AxolRuntimeException("concat: incompatible types")
            };
        }));

        env.Define("keys", new BuiltinFunctionVal("keys", args =>
        {
            if (args[0] is MapVal m)
                return new ListVal(m.Entries.Keys.Select(k => (AxolValue)new StrVal(k)).ToList());
            if (args[0] is StructVal s)
                return new ListVal(s.Fields.Keys.Select(k => (AxolValue)new StrVal(k)).ToList());
            throw new AxolRuntimeException("keys: expected map or struct");
        }));

        env.Define("values", new BuiltinFunctionVal("values", args =>
        {
            if (args[0] is MapVal m)
                return new ListVal(m.Entries.Values.ToList());
            if (args[0] is StructVal s)
                return new ListVal(s.Fields.Values.ToList());
            throw new AxolRuntimeException("values: expected map or struct");
        }));

        env.Define("contains", new BuiltinFunctionVal("contains", args =>
        {
            if (args[0] is MapVal m && args[1] is StrVal key)
                return new BoolVal(m.Entries.ContainsKey(key.Value));
            if (args[0] is ListVal l)
            {
                foreach (var item in l.Items)
                    if (ValuesEqual(item, args[1])) return new BoolVal(true);
                return new BoolVal(false);
            }
            throw new AxolRuntimeException("contains: expected map or list");
        }));

        // Register sub-module builtins
        StringBuiltins.Register(env);
        MathBuiltins.Register(env);
        IoBuiltins.Register(env);
        UtilBuiltins.Register(env);

        if (callFn != null)
            ArrayBuiltins.Register(env, callFn);
    }

    internal static long ToLong(AxolValue v) => v switch
    {
        IntVal iv => iv.Value,
        FloatVal fv => (long)fv.Value,
        BoolVal bv => bv.Value ? 1 : 0,
        _ => throw new AxolRuntimeException($"Expected number, got {v.GetType().Name}")
    };

    internal static double ToDouble(AxolValue v) => v switch
    {
        IntVal iv => iv.Value,
        FloatVal fv => fv.Value,
        _ => throw new AxolRuntimeException($"Expected number, got {v.GetType().Name}")
    };

    internal static bool ValuesEqual(AxolValue a, AxolValue b) => (a, b) switch
    {
        (IntVal ia, IntVal ib) => ia.Value == ib.Value,
        (FloatVal fa, FloatVal fb) => Math.Abs(fa.Value - fb.Value) < double.Epsilon,
        (IntVal ia, FloatVal fb) => Math.Abs(ia.Value - fb.Value) < double.Epsilon,
        (FloatVal fa, IntVal ib) => Math.Abs(fa.Value - ib.Value) < double.Epsilon,
        (StrVal sa, StrVal sb) => sa.Value == sb.Value,
        (BoolVal ba, BoolVal bb) => ba.Value == bb.Value,
        (NilVal, NilVal) => true,
        (UnitVal, UnitVal) => true,
        _ => false
    };
}
