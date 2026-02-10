namespace Axol.Interpreter.Values;

public abstract record AxolValue
{
    public virtual bool IsTruthy => true;
    public abstract string Display();
}

public sealed record IntVal(long Value) : AxolValue
{
    public override string Display() => Value.ToString();
}

public sealed record FloatVal(double Value) : AxolValue
{
    public override string Display() => Value.ToString(System.Globalization.CultureInfo.InvariantCulture);
}

public sealed record StrVal(string Value) : AxolValue
{
    public override string Display() => Value;
}

public sealed record BoolVal(bool Value) : AxolValue
{
    public override bool IsTruthy => Value;
    public override string Display() => Value ? "true" : "false";
}

public sealed record ListVal(List<AxolValue> Items) : AxolValue
{
    public override string Display() => "[" + string.Join(" ", Items.Select(i => i.Display())) + "]";
}

public sealed record MapVal(Dictionary<string, AxolValue> Entries) : AxolValue
{
    public override string Display() =>
        "{" + string.Join(" ", Entries.Select(kv => $"{kv.Key}:{kv.Value.Display()}")) + "}";
}

public sealed record StructVal(string TypeName, Dictionary<string, AxolValue> Fields) : AxolValue
{
    public override string Display() =>
        $"({TypeName} " + string.Join(" ", Fields.Select(kv => $"{kv.Key}:{kv.Value.Display()}")) + ")";
}

public sealed record FunctionVal(
    string Name,
    IReadOnlyList<string> Params,
    IReadOnlyList<Axol.Core.Ast.AstNode> Body,
    Environment Closure) : AxolValue
{
    public override string Display() => $"<fn {Name}/{Params.Count}>";
}

public sealed record BuiltinFunctionVal(
    string Name,
    Func<IReadOnlyList<AxolValue>, AxolValue> Impl) : AxolValue
{
    public override string Display() => $"<builtin {Name}>";
}

public sealed record UnitVal : AxolValue
{
    public static readonly UnitVal Instance = new();
    public override bool IsTruthy => false;
    public override string Display() => "unit";
}

public sealed record NilVal : AxolValue
{
    public static readonly NilVal Instance = new();
    public override bool IsTruthy => false;
    public override string Display() => "nil";
}

public sealed record EnumVariantVal(string EnumName, string VariantName, AxolValue? Data) : AxolValue
{
    public override string Display() => Data != null ? $"{EnumName}.{VariantName}({Data.Display()})" : $"{EnumName}.{VariantName}";
}
