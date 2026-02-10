namespace Axol.Core.Tokens;

public readonly record struct Token(TokenKind Kind, string Lexeme, object? Value, SourceSpan Span)
{
    public static Token Eof(SourceSpan span) => new(TokenKind.Eof, "", null, span);
    public static Token Error(string message, SourceSpan span) => new(TokenKind.Error, message, null, span);
}
