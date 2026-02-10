using Axol.Interpreter.Values;

namespace Axol.Interpreter;

public sealed class ModuleRegistry
{
    private readonly Dictionary<string, Environment> _modules = new();

    public void Register(string name, Environment env) => _modules[name] = env;

    public bool TryGet(string name, out Environment env) => _modules.TryGetValue(name, out env!);

    public Environment Get(string name) =>
        _modules.TryGetValue(name, out var env)
            ? env
            : throw new AxolRuntimeException($"Module not found: {name}");
}
