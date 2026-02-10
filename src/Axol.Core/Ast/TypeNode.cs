namespace Axol.Core.Ast;

public abstract record TypeNode(SourceSpan Span);

public sealed record PrimitiveType(string Name, SourceSpan Span) : TypeNode(Span);
public sealed record OptionalType(TypeNode Inner, SourceSpan Span) : TypeNode(Span);
public sealed record ArrayType(TypeNode Element, SourceSpan Span) : TypeNode(Span);
public sealed record MapType(TypeNode Key, TypeNode ValueType, SourceSpan Span) : TypeNode(Span);
public sealed record FunctionType(IReadOnlyList<TypeNode> Params, TypeNode Return, SourceSpan Span) : TypeNode(Span);
public sealed record NamedType(string Name, SourceSpan Span) : TypeNode(Span);
