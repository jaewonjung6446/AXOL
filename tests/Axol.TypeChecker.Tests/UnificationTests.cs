using Axol.TypeChecker;
using Xunit;

namespace Axol.TypeChecker.Tests;

public class UnificationTests
{
    private readonly Unifier _u = new();

    // --- Basic unification ---

    [Fact]
    public void SameType_Unifies()
    {
        Assert.True(_u.Unify(IntType.Instance, IntType.Instance));
    }

    [Fact]
    public void DifferentPrimitives_Fail()
    {
        Assert.False(_u.Unify(IntType.Instance, StringType.Instance));
    }

    [Fact]
    public void IntFloat_NumericWidening()
    {
        Assert.True(_u.Unify(IntType.Instance, FloatType.Instance));
    }

    [Fact]
    public void FloatInt_NumericWidening()
    {
        Assert.True(_u.Unify(FloatType.Instance, IntType.Instance));
    }

    // --- TypeVar unification ---

    [Fact]
    public void TypeVar_UnifiesWithConcrete()
    {
        var tv = _u.FreshVar();
        Assert.True(_u.Unify(tv, IntType.Instance));
        Assert.Equal(IntType.Instance, _u.Apply(tv));
    }

    [Fact]
    public void TypeVar_UnifiesWithTypeVar()
    {
        var a = _u.FreshVar();
        var b = _u.FreshVar();
        Assert.True(_u.Unify(a, b));
    }

    [Fact]
    public void TypeVar_Transitive()
    {
        var a = _u.FreshVar();
        var b = _u.FreshVar();
        _u.Unify(a, b);
        _u.Unify(b, IntType.Instance);
        Assert.Equal(IntType.Instance, _u.Apply(a));
    }

    // --- Occurs check ---

    [Fact]
    public void OccursCheck_Fails()
    {
        var tv = _u.FreshVar();
        // Cannot unify t1 with List<t1>
        Assert.False(_u.Unify(tv, new ListType(tv)));
    }

    // --- Structural unification ---

    [Fact]
    public void ListType_Unifies()
    {
        Assert.True(_u.Unify(new ListType(IntType.Instance), new ListType(IntType.Instance)));
    }

    [Fact]
    public void ListType_DifferentElements_Fail()
    {
        Assert.False(_u.Unify(new ListType(IntType.Instance), new ListType(StringType.Instance)));
    }

    [Fact]
    public void MapType_Unifies()
    {
        Assert.True(_u.Unify(
            new MapType(StringType.Instance, IntType.Instance),
            new MapType(StringType.Instance, IntType.Instance)));
    }

    [Fact]
    public void MapType_DifferentValue_Fail()
    {
        Assert.False(_u.Unify(
            new MapType(StringType.Instance, IntType.Instance),
            new MapType(StringType.Instance, BoolType.Instance)));
    }

    [Fact]
    public void FnType_Unifies()
    {
        var a = new FnType(new[] { IntType.Instance }, BoolType.Instance);
        var b = new FnType(new[] { IntType.Instance }, BoolType.Instance);
        Assert.True(_u.Unify(a, b));
    }

    [Fact]
    public void FnType_DifferentArity_Fail()
    {
        var a = new FnType(new[] { IntType.Instance }, BoolType.Instance);
        var b = new FnType(new AxolType[] { IntType.Instance, IntType.Instance }, BoolType.Instance);
        Assert.False(_u.Unify(a, b));
    }

    [Fact]
    public void FnType_WithTypeVar()
    {
        var tv = _u.FreshVar();
        var a = new FnType(new[] { tv }, IntType.Instance);
        var b = new FnType(new[] { StringType.Instance }, IntType.Instance);
        Assert.True(_u.Unify(a, b));
        Assert.Equal(StringType.Instance, _u.Apply(tv));
    }

    [Fact]
    public void OptionalType_Unifies()
    {
        Assert.True(_u.Unify(
            new OptionalType(IntType.Instance),
            new OptionalType(IntType.Instance)));
    }

    [Fact]
    public void OptionalType_DifferentInner_Fail()
    {
        Assert.False(_u.Unify(
            new OptionalType(IntType.Instance),
            new OptionalType(StringType.Instance)));
    }

    // --- UnknownType wildcard ---

    [Fact]
    public void UnknownType_UnifiesWithAnything()
    {
        Assert.True(_u.Unify(UnknownType.Instance, IntType.Instance));
        Assert.True(_u.Unify(StringType.Instance, UnknownType.Instance));
    }

    // --- Generalization ---

    [Fact]
    public void Generalize_FreeVars()
    {
        var tv = _u.FreshVar();
        var fnType = new FnType(new[] { tv }, tv);
        var scheme = _u.Generalize(fnType, new HashSet<int>());
        Assert.NotEmpty(scheme.Vars);
    }

    [Fact]
    public void Instantiate_FreshVars()
    {
        var tv = _u.FreshVar();
        var fnType = new FnType(new[] { tv }, tv);
        var scheme = _u.Generalize(fnType, new HashSet<int>());
        var inst1 = _u.Instantiate(scheme);
        var inst2 = _u.Instantiate(scheme);
        // Two instantiations produce different type vars
        Assert.NotEqual(inst1, inst2);
    }

    // --- Apply ---

    [Fact]
    public void Apply_ResolvesChaindSubstitution()
    {
        var a = _u.FreshVar();
        var b = _u.FreshVar();
        _u.Unify(a, b);
        _u.Unify(b, IntType.Instance);
        Assert.Equal(IntType.Instance, _u.Apply(a));
        Assert.Equal(IntType.Instance, _u.Apply(b));
    }

    [Fact]
    public void Apply_ListType()
    {
        var tv = _u.FreshVar();
        _u.Unify(tv, IntType.Instance);
        var result = _u.Apply(new ListType(tv));
        Assert.Equal(new ListType(IntType.Instance), result);
    }
}
