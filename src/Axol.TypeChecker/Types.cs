namespace Axol.TypeChecker;

public abstract record AxolType;

public sealed record IntType : AxolType { public static readonly IntType Instance = new(); }
public sealed record FloatType : AxolType { public static readonly FloatType Instance = new(); }
public sealed record BoolType : AxolType { public static readonly BoolType Instance = new(); }
public sealed record StringType : AxolType { public static readonly StringType Instance = new(); }
public sealed record UnitType : AxolType { public static readonly UnitType Instance = new(); }
public sealed record NilType : AxolType { public static readonly NilType Instance = new(); }
public sealed record ListType(AxolType Element) : AxolType;
public sealed record MapType(AxolType Key, AxolType Value) : AxolType;
public sealed record OptionalType(AxolType Inner) : AxolType;
public sealed record FnType(IReadOnlyList<AxolType> Params, AxolType Return) : AxolType;
public sealed record NamedType(string Name) : AxolType;
public sealed record UnknownType : AxolType { public static readonly UnknownType Instance = new(); }

// Phase 2 additions
public sealed record TypeVar(int Id) : AxolType;
public sealed record ForAllType(int[] Vars, AxolType Body) : AxolType;
public sealed record ErrorType(string Message) : AxolType;
