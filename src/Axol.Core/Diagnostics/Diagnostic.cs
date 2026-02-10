using System.Text.Json;
using System.Text.Json.Serialization;

namespace Axol.Core.Diagnostics;

public enum DiagnosticSeverity { Error, Warning, Info }

public sealed record Diagnostic(
    SourceSpan Span,
    string Code,
    string Message,
    DiagnosticSeverity Severity = DiagnosticSeverity.Error,
    string? Function = null,
    string? Expression = null,
    int? Line = null,
    int? Col = null)
{
    public Diagnostic WithLocation(SourceMap? sourceMap) =>
        sourceMap is null ? this : this with
        {
            Line = sourceMap.GetLineCol(Span.Start).Line,
            Col = sourceMap.GetLineCol(Span.Start).Col
        };

    public string ToJson()
    {
        var obj = new
        {
            loc = new[] { Span.Start, Span.End },
            code = Code,
            msg = Message,
            fn = Function,
            expr = Expression,
            line = Line,
            col = Col
        };
        return JsonSerializer.Serialize(obj, JsonCtx.Default.Object);
    }
}

[JsonSerializable(typeof(object))]
internal partial class JsonCtx : JsonSerializerContext { }
