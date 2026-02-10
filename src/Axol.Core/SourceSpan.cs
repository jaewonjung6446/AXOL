namespace Axol.Core;

public readonly record struct SourceSpan(string File, int Start, int End)
{
    public static readonly SourceSpan None = new("", 0, 0);

    public override string ToString() => $"{File}[{Start}..{End}]";
}
