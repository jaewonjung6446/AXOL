using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Interpreter.Values;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.Interpreter.Tests;

public class ModuleTests
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

    // --- Module definition ---

    [Fact]
    public void Module_DefineAndUseQualified()
    {
        var (result, _) = Run("(M math (f square x (* x x))) (math.square 5)");
        Assert.Equal(25L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Module_MultipleFunctions()
    {
        var (result, _) = Run("(M math (f square x (* x x)) (f cube x (* x (* x x)))) (+ (math.square 3) (math.cube 2))");
        Assert.Equal(17L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Module_Variables()
    {
        var (result, _) = Run("(M consts (v pi 3)) consts.pi");
        Assert.Equal(3L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Import ---

    [Fact]
    public void Import_Module()
    {
        var (result, _) = Run("(M math (f double x (* x 2))) (import math) (math.double 5)");
        Assert.Equal(10L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Use ---

    [Fact]
    public void Use_QualifiedName()
    {
        var (result, _) = Run("(M math (f square x (* x x))) (use math.square) (square 7)");
        Assert.Equal(49L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Use_MultipleNames()
    {
        var (result, _) = Run("(M math (f square x (* x x)) (f cube x (* x (* x x)))) (use math.square) (use math.cube) (+ (square 3) (cube 2))");
        Assert.Equal(17L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Namespace isolation ---

    [Fact]
    public void Module_IsolatedScope()
    {
        // Variable defined inside module shouldn't leak to outer scope
        var (result, _) = Run("(v outer 10) (M inner (v x 99) (f get_x x)) (inner.get_x)");
        Assert.Equal(99L, Assert.IsType<IntVal>(result).Value);
    }

    [Fact]
    public void Module_AccessOuterScope()
    {
        // Module can see outer scope bindings
        var (result, _) = Run("(v base 10) (M math (f add_base x (+ x base))) (math.add_base 5)");
        Assert.Equal(15L, Assert.IsType<IntVal>(result).Value);
    }

    // --- Module registry ---

    [Fact]
    public void Module_Registry()
    {
        var writer = new StringWriter();
        var lexer = new AxolLexer("(M test (f hello \"hi\"))");
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var program = parser.ParseProgram();
        var interp = new AxolInterpreter(writer, diag);
        interp.Run(program);
        Assert.NotNull(interp.Modules.Get("test"));
    }

    // --- Nested access in module ---

    [Fact]
    public void Module_FunctionReturnsValue()
    {
        var (result, _) = Run("(M util (f identity x x)) (util.identity 42)");
        Assert.Equal(42L, Assert.IsType<IntVal>(result).Value);
    }
}
