using System.CommandLine;
using Axol.Core;
using Axol.Core.Diagnostics;
using Axol.Interpreter;
using Axol.Lexer;
using Axol.Parser;
using Axol.TypeChecker;

var fileArg = new Argument<FileInfo>("file", "AXOL source file");

var runCmd = new Command("run", "Run an AXOL program") { fileArg };
runCmd.SetHandler(file =>
{
    var source = File.ReadAllText(file.FullName);
    var result = RunPipeline(source, file.Name, execute: true);
    System.Environment.ExitCode = result ? 0 : 1;
}, fileArg);

var checkCmd = new Command("check", "Parse and type-check only") { fileArg };
checkCmd.SetHandler(file =>
{
    var source = File.ReadAllText(file.FullName);
    var result = RunPipeline(source, file.Name, execute: false);
    if (result)
        Console.WriteLine("{\"status\":\"ok\"}");
    System.Environment.ExitCode = result ? 0 : 1;
}, fileArg);

var tokensCmd = new Command("tokens", "Count tokens in a file") { fileArg };
tokensCmd.SetHandler(file =>
{
    var source = File.ReadAllText(file.FullName);
    var lexer = new AxolLexer(source, file.Name);
    var tokens = lexer.Tokenize();
    var count = tokens.Count(t => t.Kind != Axol.Core.Tokens.TokenKind.Eof);
    Console.WriteLine($"{{\"file\":\"{file.Name}\",\"tokens\":{count},\"chars\":{source.Length}}}");
}, fileArg);

var replCmd = new Command("repl", "Interactive REPL");
replCmd.SetHandler(() =>
{
    Console.WriteLine("AXOL REPL (type 'exit' to quit)");
    var diagnostics = new DiagnosticBag();
    var interpreter = new AxolInterpreter(Console.Out, diagnostics);

    while (true)
    {
        Console.Write("axol> ");
        var line = Console.ReadLine();
        if (line == null || line.Trim() == "exit") break;
        if (string.IsNullOrWhiteSpace(line)) continue;

        try
        {
            diagnostics.Clear();
            var sourceMap = new SourceMap(line);
            var lexer = new AxolLexer(line, "<repl>");
            var tokens = lexer.Tokenize();
            var parser = new AxolParser(tokens, diagnostics);
            var program = parser.ParseProgram();

            if (diagnostics.HasErrors)
            {
                foreach (var d in diagnostics.All)
                    Console.Error.WriteLine(d.WithLocation(sourceMap).ToJson());
                continue;
            }

            var result = interpreter.Run(program);
            if (result is not Axol.Interpreter.Values.UnitVal)
                Console.WriteLine(result.Display());
        }
        catch (AxolRuntimeException ex)
        {
            if (ex.JsonError != null)
                Console.Error.WriteLine(ex.JsonError);
            else
                Console.Error.WriteLine($"{{\"code\":\"E999\",\"msg\":\"{ex.Message.Replace("\"", "\\\"")}\"}}");
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"{{\"code\":\"E999\",\"msg\":\"{ex.Message.Replace("\"", "\\\"")}\"}}");
        }
    }
});

var rootCmd = new RootCommand("AXOL - AI-optimized programming language") { runCmd, checkCmd, tokensCmd, replCmd };
return await rootCmd.InvokeAsync(args);

static bool RunPipeline(string source, string fileName, bool execute)
{
    var diagnostics = new DiagnosticBag();
    var sourceMap = new SourceMap(source);

    var lexer = new AxolLexer(source, fileName);
    var tokens = lexer.Tokenize();

    var parser = new AxolParser(tokens, diagnostics);
    var program = parser.ParseProgram();

    if (diagnostics.HasErrors)
    {
        foreach (var d in diagnostics.All)
            Console.Error.WriteLine(d.WithLocation(sourceMap).ToJson());
        return false;
    }

    var typeChecker = new AxolTypeChecker(diagnostics);
    typeChecker.Check(program);

    if (diagnostics.HasErrors)
    {
        foreach (var d in diagnostics.All)
            Console.Error.WriteLine(d.WithLocation(sourceMap).ToJson());
        return false;
    }

    if (!execute) return true;

    try
    {
        var interpreter = new AxolInterpreter(Console.Out, diagnostics);
        interpreter.Run(program);
        return true;
    }
    catch (AssertionFailedException ex)
    {
        Console.Error.WriteLine(ex.JsonError ?? ex.Message);
        return false;
    }
    catch (ContractViolationException ex)
    {
        Console.Error.WriteLine(ex.JsonError ?? ex.Message);
        return false;
    }
    catch (AxolRuntimeException ex)
    {
        Console.Error.WriteLine(ex.JsonError ?? $"{{\"code\":\"E999\",\"msg\":\"{ex.Message.Replace("\"", "\\\"")}\"}}");
        return false;
    }
}
