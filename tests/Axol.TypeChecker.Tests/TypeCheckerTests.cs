using Axol.Core.Diagnostics;
using Axol.Lexer;
using Axol.Parser;
using Axol.TypeChecker;
using Xunit;

namespace Axol.TypeChecker.Tests;

public class TypeCheckerTests
{
    private static (DiagnosticBag diag, Axol.Core.Ast.Program prog) CheckSource(string source)
    {
        var lexer = new AxolLexer(source);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var prog = parser.ParseProgram();
        var checker = new AxolTypeChecker(diag);
        checker.Check(prog);
        return (diag, prog);
    }

    [Fact]
    public void ValidProgram_NoErrors()
    {
        var (diag, _) = CheckSource("(f id [i -> i] x x)");
        Assert.False(diag.HasErrors);
    }

    [Fact]
    public void ValidExpression_NoErrors()
    {
        var (diag, _) = CheckSource("(+ 1 2)");
        Assert.False(diag.HasErrors);
    }

    [Fact]
    public void ValidBindings_NoErrors()
    {
        var (diag, _) = CheckSource("(v x 42) (m y 10)");
        Assert.False(diag.HasErrors);
    }
}
