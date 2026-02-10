using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Interpreter.Values;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Interpreter.Tests;

public class PatternMatchTests
{
    private static (AxolValue result, string output) Run(string source)
    {
        var writer = new StringWriter();
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var program = parser.ParseProgram();
        Assert.False(diag.HasErrors, string.Join("\n", diag.All.Select(d => d.ToJson())));
        var interp = new AxolInterpreter(writer, diag);
        var result = interp.Run(program);
        return (result, writer.ToString().TrimEnd().Replace("\r\n", "\n"));
    }

    // --- Existing pattern matching (backward compat) ---

    [Fact]
    public void LiteralMatch()
    {
        var (result, _) = Run("(X 1 1 \"one\" 2 \"two\" _ \"other\")");
        Assert.Equal("one", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void WildcardMatch()
    {
        var (result, _) = Run("(X 99 1 \"one\" _ \"other\")");
        Assert.Equal("other", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void VariableMatch()
    {
        var (result, _) = Run("(X 42 x x)");
        Assert.Equal(42L, Assert.IsType<IntVal>(result).Value);
    }

    // --- List destructuring ---

    [Fact]
    public void ListDestructure_Head()
    {
        var (result, _) = Run("(X (A 1 2 3) (A h rest...) h)");
        Assert.Equal(1L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void ListDestructure_Rest()
    {
        // rest... pattern stores variable as "rest" (without dots)
        var (result, _) = Run("(X (A 1 2 3) (A h rest...) (len rest))");
        Assert.Equal(2L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void ListDestructure_Empty()
    {
        var (result, _) = Run("(X (A) (A) \"empty\" _ \"nonempty\")");
        Assert.Equal("empty", Assert.IsType<StrVal>(result).Value);
    }

    // --- Struct destructuring ---

    [Fact]
    public void StructDestructure()
    {
        var (result, _) = Run("(v p (S Point x 10 y 20)) (X p (S Point x y) (+ x y))");
        Assert.Equal(30L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void StructDestructure_PartialFields()
    {
        var (result, _) = Run("(v p (S Point x 10 y 20)) (X p (S Point x y) x)");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Enum matching ---

    [Fact]
    public void EnumMatch_Simple()
    {
        var (result, _) = Run("(e Color Red Green Blue) (v c Color.Red) (X c Color.Red \"red\" Color.Green \"green\" _ \"other\")");
        Assert.Equal("red", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void EnumMatch_DataBearing()
    {
        var (result, _) = Run("(e Shape (Circle r) (Rect w h)) (v s (Shape.Circle 5)) (X s (Shape.Circle r) (* r r) _ 0)");
        Assert.Equal(25L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void EnumMatch_DataBearing_MultiField()
    {
        var (result, _) = Run("(e Shape (Circle r) (Rect w h)) (v s (Shape.Rect 3 4)) (X s (Shape.Rect w h) (* w h) _ 0)");
        Assert.Equal(12L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Guard clauses ---

    [Fact]
    public void GuardClause_Positive()
    {
        var (result, _) = Run("(X 5 x (when (> x 0)) \"positive\" x (when (< x 0)) \"negative\" _ \"zero\")");
        Assert.Equal("positive", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void GuardClause_Negative()
    {
        var (result, _) = Run("(X -3 x (when (> x 0)) \"positive\" x (when (< x 0)) \"negative\" _ \"zero\")");
        Assert.Equal("negative", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void GuardClause_Zero()
    {
        var (result, _) = Run("(X 0 x (when (> x 0)) \"positive\" x (when (< x 0)) \"negative\" _ \"zero\")");
        Assert.Equal("zero", Assert.IsType<StrVal>(result).Value);
    }

    // --- Complex patterns ---

    [Fact]
    public void PatternMatch_FibStyle()
    {
        var source = "(f fib_match n (X n 0 0 1 1 _ (+ (fib_match (- n 1)) (fib_match (- n 2))))) (fib_match 10)";
        var (result, _) = Run(source);
        Assert.Equal(55L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void PatternMatch_StringLiteral()
    {
        var (result, _) = Run("(X \"hello\" \"hello\" 1 \"world\" 2 _ 0)");
        Assert.Equal(1L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void PatternMatch_Bool()
    {
        var (result, _) = Run("(X true true \"yes\" false \"no\")");
        Assert.Equal("yes", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void PatternMatch_Nil()
    {
        var (result, _) = Run("(X nil nil \"nil\" _ \"not nil\")");
        Assert.Equal("nil", Assert.IsType<StrVal>(result).Value);
    }

    // --- Short enum aliases (.Variant) ---

    [Fact]
    public void ShortEnumAlias_SimpleMatch()
    {
        var (result, _) = Run("(e Color Red Green Blue) (v c Color.Red) (X c .Red \"red\" .Green \"green\" _ \"other\")");
        Assert.Equal("red", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void ShortEnumAlias_NoMatch_FallsToWildcard()
    {
        var (result, _) = Run("(e Color Red Green Blue) (v c Color.Blue) (X c .Red \"red\" .Green \"green\" _ \"other\")");
        Assert.Equal("other", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void ShortEnumAlias_DataBearing()
    {
        var (result, _) = Run("(e Shape (Circle r) (Rect w h)) (v s (Shape.Circle 5)) (X s (.Circle r) (* r r) _ 0)");
        Assert.Equal(25L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void ShortEnumAlias_DataBearing_MultiField()
    {
        var (result, _) = Run("(e Shape (Circle r) (Rect w h)) (v s (Shape.Rect 3 4)) (X s (.Rect w h) (* w h) _ 0)");
        Assert.Equal(12L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void ShortEnumAlias_MixedWithFull()
    {
        // Can use .Short and Full.Name in the same match
        var (result, _) = Run("(e Dir N S E W) (v d Dir.S) (X d Dir.N \"north\" .S \"south\" _ \"other\")");
        Assert.Equal("south", Assert.IsType<StrVal>(result).Value);
    }
}
