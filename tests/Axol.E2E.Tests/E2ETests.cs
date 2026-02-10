using Axol.Core;
using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Lexer;
using Axol.Parser;
using Xunit;

namespace Axol.E2E.Tests;

public class E2ETests
{
    private static string RunFile(string relativePath)
    {
        var dir = AppContext.BaseDirectory;
        var path = Path.Combine(dir, relativePath);
        var source = File.ReadAllText(path);

        // Auto-preprocess .axoli files
        if (path.EndsWith(".axoli", StringComparison.OrdinalIgnoreCase))
            source = IndentPreprocessor.Process(source);

        var writer = new StringWriter();
        var lexer = new AxolLexer(source, path);
        var tokens = lexer.Tokenize();
        var diag = new DiagnosticBag();
        var parser = new AxolParser(tokens, diag);
        var program = parser.ParseProgram();
        Assert.False(diag.HasErrors, "Parse errors: " + string.Join("\n", diag.All.Select(d => d.ToJson())));
        var interp = new AxolInterpreter(writer, diag);
        interp.Run(program);
        return writer.ToString().TrimEnd().Replace("\r\n", "\n");
    }

    [Fact]
    public void Fibonacci_Output()
    {
        var output = RunFile(Path.Combine("fixtures", "fibonacci.axol"));
        var expected = "0\n1\n1\n2\n3\n5\n8\n13\n21\n34";
        Assert.Equal(expected, output);
    }

    [Fact]
    public void Contracts_Pass()
    {
        var output = RunFile(Path.Combine("fixtures", "contracts.axol"));
        Assert.Equal("OK", output);
    }

    [Fact]
    public void Types_Output()
    {
        var output = RunFile(Path.Combine("fixtures", "types.axol"));
        var expected = "i\ns\nb\n*\n%";
        Assert.Equal(expected, output);
    }

    [Fact]
    public void IndentMode_AxoliFile()
    {
        var output = RunFile(Path.Combine("fixtures", "indent_mode.axoli"));
        Assert.Equal("30", output);
    }
}
