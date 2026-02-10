using Axol.Interpreter.Values;

namespace Axol.Interpreter;

public sealed class Environment
{
    private readonly Dictionary<string, (AxolValue Value, bool Mutable)> _bindings = new();
    private readonly Environment? _parent;

    public Environment(Environment? parent = null)
    {
        _parent = parent;
    }

    public Environment CreateChild() => new(this);

    public void Define(string name, AxolValue value, bool mutable = false)
    {
        _bindings[name] = (value, mutable);
    }

    public AxolValue Get(string name)
    {
        if (_bindings.TryGetValue(name, out var entry))
            return entry.Value;
        if (_parent != null)
            return _parent.Get(name);
        throw new AxolRuntimeException($"Undefined variable: {name}");
    }

    public bool TryGet(string name, out AxolValue value)
    {
        if (_bindings.TryGetValue(name, out var entry))
        {
            value = entry.Value;
            return true;
        }
        if (_parent != null)
            return _parent.TryGet(name, out value);
        value = NilVal.Instance;
        return false;
    }

    public Dictionary<string, AxolValue> GetAllBindings()
    {
        var result = new Dictionary<string, AxolValue>();
        foreach (var kv in _bindings)
            result[kv.Key] = kv.Value.Value;
        return result;
    }

    public void Set(string name, AxolValue value)
    {
        if (_bindings.TryGetValue(name, out var entry))
        {
            if (!entry.Mutable)
                throw new AxolRuntimeException($"Cannot reassign immutable variable: {name}");
            _bindings[name] = (value, true);
            return;
        }
        if (_parent != null)
        {
            _parent.Set(name, value);
            return;
        }
        throw new AxolRuntimeException($"Undefined variable: {name}");
    }
}

public class AxolRuntimeException : Exception
{
    public string? JsonError { get; }

    public AxolRuntimeException(string message, string? jsonError = null) : base(message)
    {
        JsonError = jsonError;
    }
}

public class ReturnSignal : Exception
{
    public AxolValue Value { get; }
    public ReturnSignal(AxolValue value) : base("return") { Value = value; }
}

public class BreakSignal : Exception
{
    public BreakSignal() : base("break") { }
}

public class ContinueSignal : Exception
{
    public ContinueSignal() : base("continue") { }
}

public class AssertionFailedException : AxolRuntimeException
{
    public AssertionFailedException(string message, string? jsonError = null) : base(message, jsonError) { }
}

public class ContractViolationException : AxolRuntimeException
{
    public ContractViolationException(string message, string? jsonError = null) : base(message, jsonError) { }
}
