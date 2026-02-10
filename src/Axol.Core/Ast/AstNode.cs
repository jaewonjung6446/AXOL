namespace Axol.Core.Ast;

public abstract record AstNode(SourceSpan Span);

public sealed record NumberLitInt(long Value, SourceSpan Span) : AstNode(Span);
public sealed record NumberLitFloat(double Value, SourceSpan Span) : AstNode(Span);
public sealed record StringLit(string Value, SourceSpan Span) : AstNode(Span);
public sealed record BoolLit(bool Value, SourceSpan Span) : AstNode(Span);
public sealed record NilLit(SourceSpan Span) : AstNode(Span);
public sealed record SymbolRef(string Name, SourceSpan Span) : AstNode(Span);
public sealed record ListForm(string Keyword, IReadOnlyList<AstNode> Args, SourceSpan Span) : AstNode(Span);
public sealed record TypeAnnotation(IReadOnlyList<TypeNode> Types, TypeNode? ReturnType, SourceSpan Span) : AstNode(Span);
public sealed record Program(IReadOnlyList<AstNode> Forms, SourceSpan Span) : AstNode(Span);
