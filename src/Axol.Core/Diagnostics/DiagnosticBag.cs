namespace Axol.Core.Diagnostics;

public sealed class DiagnosticBag
{
    private readonly List<Diagnostic> _diagnostics = new();

    public IReadOnlyList<Diagnostic> All => _diagnostics;
    public bool HasErrors => _diagnostics.Any(d => d.Severity == DiagnosticSeverity.Error);

    public void Report(Diagnostic diagnostic) => _diagnostics.Add(diagnostic);

    public void ReportError(SourceSpan span, string code, string message, string? function = null, string? expression = null)
        => Report(new Diagnostic(span, code, message, DiagnosticSeverity.Error, function, expression));

    public void Clear() => _diagnostics.Clear();
}
