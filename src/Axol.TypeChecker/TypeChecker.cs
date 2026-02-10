using Axol.Core.Ast;
using Axol.Core.Diagnostics;

namespace Axol.TypeChecker;

public sealed class AxolTypeChecker
{
    private readonly DiagnosticBag _diagnostics;
    private readonly TypeInference _inference;

    public DiagnosticBag Diagnostics => _diagnostics;

    public AxolTypeChecker(DiagnosticBag? diagnostics = null)
    {
        _diagnostics = diagnostics ?? new DiagnosticBag();
        _inference = new TypeInference(null, _diagnostics);
    }

    public void Check(Program program)
    {
        // Phase 1: structural validation
        foreach (var form in program.Forms)
            CheckForm(form);

        // Phase 2: Hindley-Milner type inference
        _inference.Infer(program);
    }

    private void CheckForm(AstNode node)
    {
        if (node is ListForm form)
        {
            // Validate known keywords have minimum arg counts
            switch (form.Keyword)
            {
                case "f" when form.Args.Count < 2:
                    _diagnostics.ReportError(form.Span, "T001", "Function definition requires name and body");
                    break;
                case "v" or "m" when form.Args.Count < 2:
                    _diagnostics.ReportError(form.Span, "T002", $"{form.Keyword}: requires name and value");
                    break;
                case "?" when form.Args.Count < 2:
                    _diagnostics.ReportError(form.Span, "T003", "Conditional requires condition and then-branch");
                    break;
            }

            foreach (var arg in form.Args)
                CheckForm(arg);
        }
    }
}
