using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Interpreter.Values;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Interpreter.Tests;

public class BuiltinTests
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

    // --- String builtins ---

    [Fact]
    public void Upper()
    {
        var (result, _) = Run("(upper \"hello\")");
        Assert.Equal("HELLO", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void Lower()
    {
        var (result, _) = Run("(lower \"HELLO\")");
        Assert.Equal("hello", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void Split()
    {
        var (result, _) = Run("(split \"a,b,c\" \",\")");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(3, list.Items.Count);
        Assert.Equal("a", Assert.IsType<StrVal>(list.Items[0]).Value);
    }

    [Fact]
    public void Join()
    {
        var (result, _) = Run("(join \",\" (A \"a\" \"b\" \"c\"))");
        Assert.Equal("a,b,c", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void Trim()
    {
        var (result, _) = Run("(trim \"  hello  \")");
        Assert.Equal("hello", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void Replace()
    {
        var (result, _) = Run("(replace \"hello world\" \"world\" \"axol\")");
        Assert.Equal("hello axol", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void StartsWith()
    {
        var (result, _) = Run("(starts_with \"hello\" \"hel\")");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void EndsWith()
    {
        var (result, _) = Run("(ends_with \"hello\" \"llo\")");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void Slice_String()
    {
        var (result, _) = Run("(slice \"hello\" 1)");
        Assert.Equal("ello", Assert.IsType<StrVal>(result).Value);
    }

    // --- Array HOF builtins ---

    [Fact]
    public void Map()
    {
        var (result, _) = Run("(map (A 1 2 3) (L x (* x 2)))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(3, list.Items.Count);
        Assert.Equal(2L, Assert.IsType<IntVal>(list.Items[0]).Value);
        Assert.Equal(4L, Assert.IsType<IntVal>(list.Items[1]).Value);
        Assert.Equal(6L, Assert.IsType<IntVal>(list.Items[2]).Value);
    }

    [Fact]
    public void Filter()
    {
        var (result, _) = Run("(filter (A 1 2 3 4 5) (L x (> x 3)))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(2, list.Items.Count);
        Assert.Equal(4L, Assert.IsType<IntVal>(list.Items[0]).Value);
        Assert.Equal(5L, Assert.IsType<IntVal>(list.Items[1]).Value);
    }

    [Fact]
    public void Reduce()
    {
        // reduce expects (reduce list init fn); multi-param lambda needs (L (params) body)
        var (result, _) = Run("(reduce (A 1 2 3 4) 0 (L (acc x) (+ acc x)))");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Sort()
    {
        var (result, _) = Run("(sort (A 3 1 2))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(1L, Assert.IsType<IntVal>(list.Items[0]).Value);
        Assert.Equal(2L, Assert.IsType<IntVal>(list.Items[1]).Value);
        Assert.Equal(3L, Assert.IsType<IntVal>(list.Items[2]).Value);
    }

    [Fact]
    public void Reverse()
    {
        var (result, _) = Run("(reverse (A 1 2 3))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(3L, Assert.IsType<IntVal>(list.Items[0]).Value);
        Assert.Equal(1L, Assert.IsType<IntVal>(list.Items[2]).Value);
    }

    [Fact]
    public void Flatten()
    {
        var (result, _) = Run("(flatten (A (A 1 2) (A 3 4)))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(4, list.Items.Count);
    }

    [Fact]
    public void Zip()
    {
        var (result, _) = Run("(zip (A 1 2) (A 3 4))");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(2, list.Items.Count);
    }

    [Fact]
    public void Find()
    {
        var (result, _) = Run("(find (A 1 2 3 4) (L x (> x 2)))");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Any()
    {
        var (result, _) = Run("(any (A 1 2 3) (L x (> x 2)))");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    [Fact]
    public void All()
    {
        var (result, _) = Run("(all (A 1 2 3) (L x (> x 0)))");
        Assert.True(Assert.IsType<BoolVal>(result).Value);
    }

    // --- Math builtins ---

    [Fact]
    public void Floor()
    {
        var (result, _) = Run("(floor 3.7)");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Ceil()
    {
        var (result, _) = Run("(ceil 3.2)");
        Assert.Equal(4L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Round()
    {
        var (result, _) = Run("(round 3.5)");
        Assert.Equal(4L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Sin()
    {
        var (result, _) = Run("(sin 0.0)");
        Assert.Equal(0.0, Assert.IsType<FloatVal>(result).Value, 5);
    }

    [Fact]
    public void Cos()
    {
        var (result, _) = Run("(cos 0.0)");
        Assert.Equal(1.0, Assert.IsType<FloatVal>(result).Value, 5);
    }

    [Fact]
    public void Pow()
    {
        var (result, _) = Run("(pow 2.0 3.0)");
        Assert.Equal(8.0, Assert.IsType<FloatVal>(result).Value, 5);
    }

    [Fact]
    public void Log()
    {
        var (result, _) = Run("(log 1.0)");
        Assert.Equal(0.0, Assert.IsType<FloatVal>(result).Value, 5);
    }

    [Fact]
    public void Random_ReturnsFloat()
    {
        var (result, _) = Run("(random)");
        var val = Assert.IsType<FloatVal>(result);
        Assert.InRange(val.Value, 0.0, 1.0);
    }

    // --- Util builtins ---

    [Fact]
    public void Format()
    {
        var (result, _) = Run("(format \"hello {0}\" \"world\")");
        Assert.Contains("hello", Assert.IsType<StrVal>(result).Value);
    }

    [Fact]
    public void AssertEq_Pass()
    {
        var (result, _) = Run("(assert_eq 1 1)");
        Assert.IsType<UnitVal>(result);
    }

    [Fact]
    public void AssertEq_Fail()
    {
        Assert.ThrowsAny<Exception>(() => Run("(assert_eq 1 2)"));
    }

    // --- Step 2: New syntax integration ---

    [Fact]
    public void ArrayLiteral_Syntax()
    {
        var (result, _) = Run("[1 2 3]");
        var list = Assert.IsType<ListVal>(result);
        Assert.Equal(3, list.Items.Count);
    }

    [Fact]
    public void MapLiteral_Syntax()
    {
        var (result, _) = Run("{\"a\" 1 \"b\" 2}");
        var map = Assert.IsType<MapVal>(result);
        Assert.Equal(2, map.Entries.Count);
    }

    [Fact]
    public void DotAccess_Syntax()
    {
        var (result, _) = Run("(v p (S Point x 10 y 20)) p.x");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }
}
