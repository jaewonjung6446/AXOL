namespace Axol.TypeChecker;

public sealed class Unifier
{
    private readonly Dictionary<int, AxolType> _substitution = new();
    private int _nextId = 0;

    public TypeVar FreshVar() => new(++_nextId);

    public AxolType Apply(AxolType t) => t switch
    {
        TypeVar tv => _substitution.TryGetValue(tv.Id, out var bound) ? Apply(bound) : tv,
        FnType fn => new FnType(fn.Params.Select(Apply).ToList(), Apply(fn.Return)),
        ListType lt => new ListType(Apply(lt.Element)),
        MapType mt => new MapType(Apply(mt.Key), Apply(mt.Value)),
        OptionalType ot => new OptionalType(Apply(ot.Inner)),
        ForAllType fa => new ForAllType(fa.Vars, Apply(fa.Body)),
        _ => t
    };

    public bool Unify(AxolType a, AxolType b)
    {
        a = Apply(a);
        b = Apply(b);

        if (a.Equals(b)) return true;

        if (a is TypeVar tv1)
        {
            if (OccursIn(tv1.Id, b)) return false; // occurs check
            _substitution[tv1.Id] = b;
            return true;
        }

        if (b is TypeVar tv2)
        {
            if (OccursIn(tv2.Id, a)) return false;
            _substitution[tv2.Id] = a;
            return true;
        }

        // UnknownType unifies with anything
        if (a is UnknownType || b is UnknownType) return true;

        // Structural unification
        if (a is FnType fnA && b is FnType fnB)
        {
            if (fnA.Params.Count != fnB.Params.Count) return false;
            for (int i = 0; i < fnA.Params.Count; i++)
                if (!Unify(fnA.Params[i], fnB.Params[i])) return false;
            return Unify(fnA.Return, fnB.Return);
        }

        if (a is ListType ltA && b is ListType ltB)
            return Unify(ltA.Element, ltB.Element);

        if (a is MapType mtA && b is MapType mtB)
            return Unify(mtA.Key, mtB.Key) && Unify(mtA.Value, mtB.Value);

        if (a is OptionalType otA && b is OptionalType otB)
            return Unify(otA.Inner, otB.Inner);

        // Numeric widening: Int unifies with Float
        if ((a is IntType && b is FloatType) || (a is FloatType && b is IntType))
            return true;

        return false;
    }

    private bool OccursIn(int varId, AxolType t)
    {
        t = Apply(t);
        return t switch
        {
            TypeVar tv => tv.Id == varId,
            FnType fn => fn.Params.Any(p => OccursIn(varId, p)) || OccursIn(varId, fn.Return),
            ListType lt => OccursIn(varId, lt.Element),
            MapType mt => OccursIn(varId, mt.Key) || OccursIn(varId, mt.Value),
            OptionalType ot => OccursIn(varId, ot.Inner),
            _ => false
        };
    }

    public ForAllType Generalize(AxolType t, HashSet<int> envFreeVars)
    {
        var applied = Apply(t);
        var typeVars = new HashSet<int>();
        CollectTypeVars(applied, typeVars);
        typeVars.ExceptWith(envFreeVars);
        return new ForAllType(typeVars.ToArray(), applied);
    }

    public AxolType Instantiate(ForAllType scheme)
    {
        var mapping = new Dictionary<int, AxolType>();
        foreach (var v in scheme.Vars)
            mapping[v] = FreshVar();
        return SubstituteVars(scheme.Body, mapping);
    }

    private static AxolType SubstituteVars(AxolType t, Dictionary<int, AxolType> mapping) => t switch
    {
        TypeVar tv => mapping.TryGetValue(tv.Id, out var rep) ? rep : tv,
        FnType fn => new FnType(fn.Params.Select(p => SubstituteVars(p, mapping)).ToList(), SubstituteVars(fn.Return, mapping)),
        ListType lt => new ListType(SubstituteVars(lt.Element, mapping)),
        MapType mt => new MapType(SubstituteVars(mt.Key, mapping), SubstituteVars(mt.Value, mapping)),
        OptionalType ot => new OptionalType(SubstituteVars(ot.Inner, mapping)),
        _ => t
    };

    private static void CollectTypeVars(AxolType t, HashSet<int> vars)
    {
        switch (t)
        {
            case TypeVar tv: vars.Add(tv.Id); break;
            case FnType fn:
                foreach (var p in fn.Params) CollectTypeVars(p, vars);
                CollectTypeVars(fn.Return, vars);
                break;
            case ListType lt: CollectTypeVars(lt.Element, vars); break;
            case MapType mt: CollectTypeVars(mt.Key, vars); CollectTypeVars(mt.Value, vars); break;
            case OptionalType ot: CollectTypeVars(ot.Inner, vars); break;
        }
    }
}
