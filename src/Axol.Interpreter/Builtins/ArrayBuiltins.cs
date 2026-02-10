using Axol.Interpreter.Values;

namespace Axol.Interpreter.BuiltinModules;

internal static class ArrayBuiltins
{
    public static void Register(Environment env, FnCaller callFn)
    {
        env.Define("map", new BuiltinFunctionVal("map", args =>
        {
            if (args[0] is ListVal list)
            {
                var fn = args[1];
                var result = list.Items.Select(item => callFn(fn, new List<AxolValue> { item })).ToList();
                return new ListVal(result);
            }
            throw new AxolRuntimeException("map: expected (map list fn)");
        }));

        env.Define("filter", new BuiltinFunctionVal("filter", args =>
        {
            if (args[0] is ListVal list)
            {
                var fn = args[1];
                var result = list.Items.Where(item => callFn(fn, new List<AxolValue> { item }).IsTruthy).ToList();
                return new ListVal(result);
            }
            throw new AxolRuntimeException("filter: expected (filter list fn)");
        }));

        env.Define("reduce", new BuiltinFunctionVal("reduce", args =>
        {
            if (args[0] is ListVal list)
            {
                var init = args[1];
                var fn = args[2];
                var acc = init;
                foreach (var item in list.Items)
                    acc = callFn(fn, new List<AxolValue> { acc, item });
                return acc;
            }
            throw new AxolRuntimeException("reduce: expected (reduce list init fn)");
        }));

        env.Define("sort", new BuiltinFunctionVal("sort", args =>
        {
            if (args[0] is ListVal list)
            {
                var sorted = new List<AxolValue>(list.Items);
                sorted.Sort((a, b) =>
                {
                    var ad = Axol.Interpreter.Builtins.ToDouble(a);
                    var bd = Axol.Interpreter.Builtins.ToDouble(b);
                    return ad.CompareTo(bd);
                });
                return new ListVal(sorted);
            }
            throw new AxolRuntimeException("sort: expected list");
        }));

        env.Define("reverse", new BuiltinFunctionVal("reverse", args =>
        {
            if (args[0] is ListVal list)
            {
                var reversed = new List<AxolValue>(list.Items);
                reversed.Reverse();
                return new ListVal(reversed);
            }
            throw new AxolRuntimeException("reverse: expected list");
        }));

        env.Define("flatten", new BuiltinFunctionVal("flatten", args =>
        {
            if (args[0] is ListVal list)
            {
                var flat = new List<AxolValue>();
                foreach (var item in list.Items)
                {
                    if (item is ListVal inner)
                        flat.AddRange(inner.Items);
                    else
                        flat.Add(item);
                }
                return new ListVal(flat);
            }
            throw new AxolRuntimeException("flatten: expected list");
        }));

        env.Define("zip", new BuiltinFunctionVal("zip", args =>
        {
            if (args[0] is ListVal a && args[1] is ListVal b)
            {
                var len = Math.Min(a.Items.Count, b.Items.Count);
                var result = new List<AxolValue>();
                for (int i = 0; i < len; i++)
                    result.Add(new ListVal(new List<AxolValue> { a.Items[i], b.Items[i] }));
                return new ListVal(result);
            }
            throw new AxolRuntimeException("zip: expected two lists");
        }));

        env.Define("find", new BuiltinFunctionVal("find", args =>
        {
            if (args[0] is ListVal list)
            {
                var fn = args[1];
                foreach (var item in list.Items)
                {
                    if (callFn(fn, new List<AxolValue> { item }).IsTruthy)
                        return item;
                }
                return NilVal.Instance;
            }
            throw new AxolRuntimeException("find: expected (find list fn)");
        }));

        env.Define("any", new BuiltinFunctionVal("any", args =>
        {
            if (args[0] is ListVal list)
            {
                var fn = args[1];
                foreach (var item in list.Items)
                {
                    if (callFn(fn, new List<AxolValue> { item }).IsTruthy)
                        return new BoolVal(true);
                }
                return new BoolVal(false);
            }
            throw new AxolRuntimeException("any: expected (any list fn)");
        }));

        env.Define("all", new BuiltinFunctionVal("all", args =>
        {
            if (args[0] is ListVal list)
            {
                var fn = args[1];
                foreach (var item in list.Items)
                {
                    if (!callFn(fn, new List<AxolValue> { item }).IsTruthy)
                        return new BoolVal(false);
                }
                return new BoolVal(true);
            }
            throw new AxolRuntimeException("all: expected (all list fn)");
        }));
    }
}
