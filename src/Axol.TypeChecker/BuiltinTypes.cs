namespace Axol.TypeChecker;

public static class BuiltinTypes
{
    public static void Register(TypeEnvironment env)
    {
        // I/O
        env.Define("print", new FnType(new[] { UnknownType.Instance }, UnitType.Instance));
        env.Define("input", new FnType(Array.Empty<AxolType>(), StringType.Instance));

        // Numeric
        env.Define("len", new FnType(new[] { UnknownType.Instance }, IntType.Instance));
        env.Define("max", new FnType(new AxolType[] { IntType.Instance, IntType.Instance }, IntType.Instance));
        env.Define("min", new FnType(new AxolType[] { IntType.Instance, IntType.Instance }, IntType.Instance));
        env.Define("abs", new FnType(new AxolType[] { IntType.Instance }, IntType.Instance));
        env.Define("sqrt", new FnType(new AxolType[] { FloatType.Instance }, FloatType.Instance));
        env.Define("range", new FnType(new AxolType[] { IntType.Instance }, new ListType(IntType.Instance)));

        // Collections
        env.Define("push", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("pop", new FnType(new[] { UnknownType.Instance }, UnknownType.Instance));
        env.Define("concat", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("keys", new FnType(new[] { UnknownType.Instance }, new ListType(StringType.Instance)));
        env.Define("values", new FnType(new[] { UnknownType.Instance }, new ListType(UnknownType.Instance)));
        env.Define("contains", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, BoolType.Instance));

        // Conversion
        env.Define("str", new FnType(new[] { UnknownType.Instance }, StringType.Instance));
        env.Define("int", new FnType(new[] { UnknownType.Instance }, IntType.Instance));
        env.Define("float", new FnType(new[] { UnknownType.Instance }, FloatType.Instance));
        env.Define("type", new FnType(new[] { UnknownType.Instance }, StringType.Instance));

        // String builtins
        env.Define("upper", new FnType(new[] { StringType.Instance }, StringType.Instance));
        env.Define("lower", new FnType(new[] { StringType.Instance }, StringType.Instance));
        env.Define("split", new FnType(new AxolType[] { StringType.Instance, StringType.Instance }, new ListType(StringType.Instance)));
        env.Define("join", new FnType(new AxolType[] { StringType.Instance, new ListType(StringType.Instance) }, StringType.Instance));
        env.Define("trim", new FnType(new[] { StringType.Instance }, StringType.Instance));
        env.Define("replace", new FnType(new AxolType[] { StringType.Instance, StringType.Instance, StringType.Instance }, StringType.Instance));
        env.Define("starts_with", new FnType(new AxolType[] { StringType.Instance, StringType.Instance }, BoolType.Instance));
        env.Define("ends_with", new FnType(new AxolType[] { StringType.Instance, StringType.Instance }, BoolType.Instance));
        env.Define("slice", new FnType(new AxolType[] { UnknownType.Instance, IntType.Instance }, UnknownType.Instance));

        // Array HOF
        env.Define("map", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("filter", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("reduce", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("sort", new FnType(new[] { UnknownType.Instance }, UnknownType.Instance));
        env.Define("reverse", new FnType(new[] { UnknownType.Instance }, UnknownType.Instance));
        env.Define("flatten", new FnType(new[] { UnknownType.Instance }, UnknownType.Instance));
        env.Define("zip", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("find", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnknownType.Instance));
        env.Define("any", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, BoolType.Instance));
        env.Define("all", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, BoolType.Instance));

        // Math
        env.Define("floor", new FnType(new[] { FloatType.Instance }, IntType.Instance));
        env.Define("ceil", new FnType(new[] { FloatType.Instance }, IntType.Instance));
        env.Define("round", new FnType(new[] { FloatType.Instance }, IntType.Instance));
        env.Define("sin", new FnType(new[] { FloatType.Instance }, FloatType.Instance));
        env.Define("cos", new FnType(new[] { FloatType.Instance }, FloatType.Instance));
        env.Define("pow", new FnType(new AxolType[] { FloatType.Instance, FloatType.Instance }, FloatType.Instance));
        env.Define("log", new FnType(new[] { FloatType.Instance }, FloatType.Instance));
        env.Define("random", new FnType(Array.Empty<AxolType>(), FloatType.Instance));

        // IO
        env.Define("read_file", new FnType(new[] { StringType.Instance }, StringType.Instance));
        env.Define("write_file", new FnType(new AxolType[] { StringType.Instance, StringType.Instance }, UnitType.Instance));

        // Util
        env.Define("format", new FnType(new[] { StringType.Instance }, StringType.Instance));
        env.Define("assert_eq", new FnType(new AxolType[] { UnknownType.Instance, UnknownType.Instance }, UnitType.Instance));
    }
}
