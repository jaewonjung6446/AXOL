namespace Axol.TypeChecker;

public sealed class TypeEnvironment
{
    private readonly Dictionary<string, AxolType> _bindings = new();
    private readonly TypeEnvironment? _parent;

    public TypeEnvironment(TypeEnvironment? parent = null) => _parent = parent;

    public TypeEnvironment CreateChild() => new(this);

    public void Define(string name, AxolType type) => _bindings[name] = type;

    public AxolType? Lookup(string name)
    {
        if (_bindings.TryGetValue(name, out var t)) return t;
        return _parent?.Lookup(name);
    }

    public HashSet<int> FreeTypeVars()
    {
        var free = new HashSet<int>();
        foreach (var t in _bindings.Values)
            CollectFreeVars(t, free);
        if (_parent != null)
            free.UnionWith(_parent.FreeTypeVars());
        return free;
    }

    private static void CollectFreeVars(AxolType t, HashSet<int> vars)
    {
        switch (t)
        {
            case TypeVar tv:
                vars.Add(tv.Id);
                break;
            case FnType fn:
                foreach (var p in fn.Params) CollectFreeVars(p, vars);
                CollectFreeVars(fn.Return, vars);
                break;
            case ListType lt:
                CollectFreeVars(lt.Element, vars);
                break;
            case MapType mt:
                CollectFreeVars(mt.Key, vars);
                CollectFreeVars(mt.Value, vars);
                break;
            case OptionalType ot:
                CollectFreeVars(ot.Inner, vars);
                break;
            case ForAllType fa:
                CollectFreeVars(fa.Body, vars);
                foreach (var v in fa.Vars) vars.Remove(v);
                break;
        }
    }
}
