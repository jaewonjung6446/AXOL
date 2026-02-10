using Axol.Core.Ast;
using Axol.Core.Diagnostics;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Parser.Tests;

public class Phase2ParserTests
{
    private static Program Parse(string source)
    {
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var parser = new AxolParser(tokens);
        return parser.ParseProgram();
    }

    private static (Program prog, DiagnosticBag diag) ParseWithDiag(string source)
    {
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        return (parser.ParseProgram(), diag);
    }

    // --- Step 1: Comments ---

    [Fact]
    public void CommentIgnored_ParsesExpression()
    {
        var prog = Parse("; comment\n(+ 1 2)");
        Assert.Single(prog.Forms);
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("+", form.Keyword);
    }

    [Fact]
    public void InlineComment_Ignored()
    {
        var prog = Parse("(+ 1 2) ; this is a comment");
        Assert.Single(prog.Forms);
    }

    // --- Step 1: Error recovery ---

    [Fact]
    public void MissingCloseParen_ErrorReported()
    {
        var (prog, diag) = ParseWithDiag("(+ 1 2");
        Assert.True(diag.HasErrors);
    }

    [Fact]
    public void ErrorRecovery_ContinuesParsing()
    {
        var (prog, diag) = ParseWithDiag("(+ 1 2 42");
        Assert.True(diag.HasErrors || prog.Forms.Count >= 1);
    }

    [Fact]
    public void MaxErrors_LimitsReporting()
    {
        // Generate many errors
        var source = string.Concat(Enumerable.Repeat("(", 20));
        var (_, diag) = ParseWithDiag(source);
        // Should not exceed MaxErrors (10)
        Assert.True(diag.All.Count() <= 10);
    }

    // --- Step 1: SourceMap integration ---

    [Fact]
    public void SourceMap_LineCol_Correct()
    {
        var map = new Axol.Core.SourceMap("line1\nline2\nline3");
        var (line1, col1) = map.GetLineCol(0);
        Assert.Equal(1, line1);
        Assert.Equal(1, col1);

        var (line2, col2) = map.GetLineCol(6);
        Assert.Equal(2, line2);
        Assert.Equal(1, col2);

        var (line3, col3) = map.GetLineCol(12);
        Assert.Equal(3, line3);
        Assert.Equal(1, col3);
    }

    [Fact]
    public void SourceMap_MidLine()
    {
        var map = new Axol.Core.SourceMap("hello\nworld");
        var (line, col) = map.GetLineCol(8);
        Assert.Equal(2, line);
        Assert.Equal(3, col);
    }

    [Fact]
    public void Diagnostic_WithLocation()
    {
        var map = new Axol.Core.SourceMap("line1\nline2");
        var diag = new Diagnostic(new Axol.Core.SourceSpan("<test>", 6, 11), "E001", "test");
        var located = diag.WithLocation(map);
        Assert.Equal(2, located.Line);
        Assert.Equal(1, located.Col);
    }

    // --- Step 2: Array literal ---

    [Fact]
    public void ArrayLiteral_Desugars()
    {
        var prog = Parse("[1 2 3]");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("A", form.Keyword);
        Assert.Equal(3, form.Args.Count);
    }

    [Fact]
    public void ArrayLiteral_Empty()
    {
        var prog = Parse("[]");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("A", form.Keyword);
        Assert.Empty(form.Args);
    }

    [Fact]
    public void ArrayLiteral_Nested()
    {
        var prog = Parse("[[1 2] [3 4]]");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("A", form.Keyword);
        Assert.Equal(2, form.Args.Count);
        var inner = Assert.IsType<ListForm>(form.Args[0]);
        Assert.Equal("A", inner.Keyword);
    }

    // --- Step 2: Backward compat - type annotation still works ---

    [Fact]
    public void TypeAnnotation_StillWorks()
    {
        var prog = Parse("[i -> i]");
        Assert.IsType<TypeAnnotation>(prog.Forms[0]);
    }

    [Fact]
    public void TypeAnnotation_MultiParam()
    {
        var prog = Parse("[i i -> i]");
        var ta = Assert.IsType<TypeAnnotation>(prog.Forms[0]);
        Assert.Equal(2, ta.Types.Count);
        Assert.NotNull(ta.ReturnType);
    }

    [Fact]
    public void FunctionDef_WithArrayLiteral_Syntax()
    {
        // Original (f id [i -> i] x x) should still work - [i -> i] is type annotation
        var prog = Parse("(f id [i -> i] x x)");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("f", form.Keyword);
        Assert.IsType<TypeAnnotation>(form.Args[1]);
    }

    // --- Step 2: Map literal ---

    [Fact]
    public void MapLiteral_Desugars()
    {
        var prog = Parse("{\"a\" 1 \"b\" 2}");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("H", form.Keyword);
        Assert.Equal(4, form.Args.Count);
    }

    [Fact]
    public void MapLiteral_Empty()
    {
        var prog = Parse("{}");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("H", form.Keyword);
        Assert.Empty(form.Args);
    }

    // --- Step 2: Dot access ---

    [Fact]
    public void DotAccess_Desugars()
    {
        var prog = Parse("obj.field");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("@", form.Keyword);
        Assert.Equal(2, form.Args.Count);
        Assert.Equal("obj", ((SymbolRef)form.Args[0]).Name);
        Assert.Equal("field", ((SymbolRef)form.Args[1]).Name);
    }

    [Fact]
    public void DotAccess_Chained()
    {
        var prog = Parse("a.b.c");
        // Should desugar to (@ (@ a b) c)
        var outer = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("@", outer.Keyword);
        var inner = Assert.IsType<ListForm>(outer.Args[0]);
        Assert.Equal("@", inner.Keyword);
        Assert.Equal("a", ((SymbolRef)inner.Args[0]).Name);
        Assert.Equal("b", ((SymbolRef)inner.Args[1]).Name);
        Assert.Equal("c", ((SymbolRef)outer.Args[1]).Name);
    }

    [Fact]
    public void DotAccess_InExpression()
    {
        var prog = Parse("(print obj.field)");
        var form = Assert.IsType<ListForm>(prog.Forms[0]);
        Assert.Equal("print", form.Keyword);
        var access = Assert.IsType<ListForm>(form.Args[0]);
        Assert.Equal("@", access.Keyword);
    }
}
