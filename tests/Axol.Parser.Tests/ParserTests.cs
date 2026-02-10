using Axol.Core.Ast;
using Axol.Core.Diagnostics;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Parser.Tests;

public class ParserTests
{
    private static Program Parse(string source)
    {
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var parser = new AxolParser(tokens);
        return parser.ParseProgram();
    }

    [Fact]
    public void ParseNumberLiteral()
    {
        var prog = Parse("42");
        Assert.Single(prog.Forms);
        Assert.IsType<NumberLitInt>(prog.Forms[0]);
        Assert.Equal(42L, ((NumberLitInt)prog.Forms[0]).Value);
    }

    [Fact]
    public void ParseStringLiteral()
    {
        var prog = Parse("\"hello\"");
        Assert.Single(prog.Forms);
        Assert.IsType<StringLit>(prog.Forms[0]);
        Assert.Equal("hello", ((StringLit)prog.Forms[0]).Value);
    }

    [Fact]
    public void ParseSymbol()
    {
        var prog = Parse("foo");
        Assert.Single(prog.Forms);
        Assert.IsType<SymbolRef>(prog.Forms[0]);
        Assert.Equal("foo", ((SymbolRef)prog.Forms[0]).Name);
    }

    [Fact]
    public void ParseBooleans()
    {
        var prog = Parse("true false");
        Assert.Equal(2, prog.Forms.Count);
        Assert.IsType<BoolLit>(prog.Forms[0]);
        Assert.True(((BoolLit)prog.Forms[0]).Value);
        Assert.False(((BoolLit)prog.Forms[1]).Value);
    }

    [Fact]
    public void ParseSimpleListForm()
    {
        var prog = Parse("(+ 1 2)");
        Assert.Single(prog.Forms);
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("+", form.Keyword);
        Assert.Equal(2, form.Args.Count);
        Assert.IsType<NumberLitInt>(form.Args[0]);
        Assert.IsType<NumberLitInt>(form.Args[1]);
    }

    [Fact]
    public void ParseNestedForms()
    {
        var prog = Parse("(+ 1 (* 2 3))");
        var outer = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("+", outer.Keyword);
        var inner = Assert.IsType<ListForm>(outer.Args[1]);
        Assert.Equal("*", inner.Keyword);
    }

    [Fact]
    public void ParseFunctionDef()
    {
        var prog = Parse("(f add [i i -> i] a b (+ a b))");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("f", form.Keyword);
        // args: name, type_ann, param a, param b, body
        Assert.True(form.Args.Count >= 4);
        Assert.IsType<SymbolRef>(form.Args[0]); // name "add"
        Assert.IsType<TypeAnnotation>(form.Args[1]); // [i i -> i]
    }

    [Fact]
    public void ParseTypeAnnotation()
    {
        var prog = Parse("[i i -> i]");
        var ta = Assert.IsType<TypeAnnotation>(prog.Forms[0]);
        Assert.Equal(2, ta.Types.Count);
        Assert.NotNull(ta.ReturnType);
    }

    [Fact]
    public void ParseConditional()
    {
        var prog = Parse("(? (> x 0) x 0)");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("?", form.Keyword);
        Assert.Equal(3, form.Args.Count);
    }

    [Fact]
    public void ParseVariableBinding()
    {
        var prog = Parse("(v x 42)");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("v", form.Keyword);
        Assert.Equal(2, form.Args.Count);
    }

    [Fact]
    public void ParseEmptyForm()
    {
        var prog = Parse("()");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("D", form.Keyword);
        Assert.Empty(form.Args);
    }

    [Fact]
    public void ParseFibonacci()
    {
        var prog = Parse("(f fib [i -> i] n (? (<= n 1) n (+ (fib (- n 1)) (fib (- n 2)))))");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("f", form.Keyword);
    }

    [Fact]
    public void NoParseErrors()
    {
        var lexer = new AxolLexer("(f id [i -> i] x x)");
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        parser.ParseProgram();
        Assert.False(diag.HasErrors);
    }
}
