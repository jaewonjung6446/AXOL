using Axol.Core;
using Xunit;

namespace Axol.Interpreter.Tests;

public class IndentPreprocessorTests
{
    [Fact]
    public void EmptyInput_ReturnsEmpty()
    {
        Assert.Equal("", IndentPreprocessor.Process(""));
    }

    [Fact]
    public void BlankLines_ReturnsEmpty()
    {
        Assert.Equal("", IndentPreprocessor.Process("  \n\n  \n"));
    }

    [Fact]
    public void SingleLine_WrapsInParens()
    {
        var result = IndentPreprocessor.Process("v x 10");
        Assert.Equal("(v x 10)", result);
    }

    [Fact]
    public void MultipleFlatLines_EachWrapped()
    {
        var input = "v x 10\nv y 20";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10) (v y 20)", result);
    }

    [Fact]
    public void IndentedChild_BecomesNested()
    {
        var input = "f add a b\n  + a b";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(f add a b (+ a b))", result);
    }

    [Fact]
    public void MultipleChildren_AllNested()
    {
        var input = "D\n  v x 10\n  v y 20\n  + x y";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(D (v x 10) (v y 20) (+ x y))", result);
    }

    [Fact]
    public void DeepNesting()
    {
        var input = "f fib n\n  ? (<= n 1) n\n    + (fib (- n 1)) (fib (- n 2))";
        var result = IndentPreprocessor.Process(input);
        // The ? line has a child (the + line), so ? stays open
        Assert.Contains("(f fib n", result);
        Assert.Contains("(? (<= n 1) n", result);
    }

    [Fact]
    public void SexprLine_PassedThrough()
    {
        var input = "(v x 10)\n(v y 20)";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10) (v y 20)", result);
    }

    [Fact]
    public void MixedSexprAndIndent()
    {
        var input = "v x 10\n(+ x 20)";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10) (+ x 20)", result);
    }

    [Fact]
    public void CommentLines_Skipped()
    {
        var input = "; this is a comment\nv x 10";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10)", result);
    }

    [Fact]
    public void InlineComments_Stripped()
    {
        var input = "v x 10 ; set x to 10";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10)", result);
    }

    [Fact]
    public void TabIndentation_Handled()
    {
        var input = "f foo x\n\t+ x 1";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(f foo x (+ x 1))", result);
    }

    [Fact]
    public void Dedent_ClosesCorrectly()
    {
        var input = "f outer\n  f inner x\n    + x 1\n  inner 5";
        var result = IndentPreprocessor.Process(input);
        // inner has child (+x1), then dedent closes inner, then (inner 5) is child of outer
        Assert.Contains("(f inner x (+ x 1))", result);
        Assert.Contains("(inner 5)", result);
    }

    [Fact]
    public void RealWorldExample_FunctionDef()
    {
        var input = @"f apply_dmg ent amt
  Q (>= amt 0)
  v raw (- amt (@ ent def))
  v dmg (? (< raw 0) 0 raw)
  ent";
        var result = IndentPreprocessor.Process(input);
        Assert.StartsWith("(f apply_dmg ent amt", result);
        Assert.Contains("(Q (>= amt 0))", result);
        Assert.Contains("(v raw (- amt (@ ent def)))", result);
        Assert.EndsWith(")", result);
    }

    [Fact]
    public void CarriageReturn_Handled()
    {
        var input = "v x 10\r\nv y 20\r\n";
        var result = IndentPreprocessor.Process(input);
        Assert.Equal("(v x 10) (v y 20)", result);
    }
}
