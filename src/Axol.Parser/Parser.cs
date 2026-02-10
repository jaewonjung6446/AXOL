using Axol.Core;
using Axol.Core.Ast;
using Axol.Core.Diagnostics;
using Axol.Core.Tokens;

namespace Axol.Parser;

public sealed class AxolParser
{
    private readonly List<Token> _tokens;
    private readonly DiagnosticBag _diagnostics;
    private int _pos;
    private int _errorCount;
    private const int MaxErrors = 10;

    public DiagnosticBag Diagnostics => _diagnostics;

    public AxolParser(List<Token> tokens, DiagnosticBag? diagnostics = null)
    {
        _tokens = tokens;
        _diagnostics = diagnostics ?? new DiagnosticBag();
        _pos = 0;
        _errorCount = 0;
    }

    private Token Current => _pos < _tokens.Count ? _tokens[_pos] : Token.Eof(SourceSpan.None);
    private Token Peek(int offset = 0) => _pos + offset < _tokens.Count ? _tokens[_pos + offset] : Token.Eof(SourceSpan.None);

    private Token Advance()
    {
        var tok = Current;
        _pos++;
        return tok;
    }

    private Token Expect(TokenKind kind, string context)
    {
        if (Current.Kind == kind)
            return Advance();

        if (_errorCount < MaxErrors)
        {
            _diagnostics.ReportError(Current.Span, "E001", $"Expected {kind} in {context}, got {Current.Kind} '{Current.Lexeme}'");
            _errorCount++;
        }
        return Current;
    }

    private void SynchronizeTo(params TokenKind[] kinds)
    {
        while (Current.Kind != TokenKind.Eof)
        {
            foreach (var kind in kinds)
                if (Current.Kind == kind) return;
            Advance();
        }
    }

    public Program ParseProgram()
    {
        var forms = new List<AstNode>();
        var start = Current.Span;

        while (Current.Kind != TokenKind.Eof)
        {
            if (_errorCount >= MaxErrors) break;
            var node = ParseForm();
            if (node != null)
                forms.Add(node);
        }

        var end = Current.Span;
        return new Program(forms, new SourceSpan(start.File, start.Start, end.End));
    }

    private AstNode? ParseForm()
    {
        switch (Current.Kind)
        {
            case TokenKind.LParen:
                return ParseListForm();
            case TokenKind.LBracket:
                return ParseBracketForm();
            case TokenKind.LBrace:
                return ParseMapLiteral();
            case TokenKind.IntLiteral:
            {
                var tok = Advance();
                return new NumberLitInt((long)tok.Value!, tok.Span);
            }
            case TokenKind.FloatLiteral:
            {
                var tok = Advance();
                return new NumberLitFloat((double)tok.Value!, tok.Span);
            }
            case TokenKind.StrLiteral:
            {
                var tok = Advance();
                return new StringLit((string)tok.Value!, tok.Span);
            }
            case TokenKind.Symbol:
            {
                var tok = Advance();
                if (tok.Lexeme == "true") return new BoolLit(true, tok.Span);
                if (tok.Lexeme == "false") return new BoolLit(false, tok.Span);
                if (tok.Lexeme == "nil") return new NilLit(tok.Span);

                // Dot access desugaring: a.b -> (@ a b), a.b.c -> (@ (@ a b) c)
                if (tok.Lexeme.Contains('.') && !tok.Lexeme.StartsWith('.') && !tok.Lexeme.EndsWith('.'))
                {
                    return DesugarDotAccess(tok.Lexeme, tok.Span);
                }

                return new SymbolRef(tok.Lexeme, tok.Span);
            }
            case TokenKind.Arrow:
            {
                var tok = Advance();
                return new SymbolRef("->", tok.Span);
            }
            case TokenKind.Eof:
                return null;
            default:
            {
                if (_errorCount < MaxErrors)
                {
                    _diagnostics.ReportError(Current.Span, "E002", $"Unexpected token {Current.Kind} '{Current.Lexeme}'");
                    _errorCount++;
                }
                Advance();
                return null;
            }
        }
    }

    private AstNode DesugarDotAccess(string lexeme, SourceSpan span)
    {
        var parts = lexeme.Split('.');
        AstNode result = new SymbolRef(parts[0], span);
        for (int i = 1; i < parts.Length; i++)
        {
            result = new ListForm("@", new List<AstNode> { result, new SymbolRef(parts[i], span) }, span);
        }
        return result;
    }

    /// <summary>
    /// Decides whether `[...]` is a type annotation or an array literal.
    /// Type annotation context: inside a function def (f name [types...] ...)
    /// where the bracket immediately follows a symbol that looks like a function name.
    /// Array literal: standalone or in expression position.
    /// Heuristic: If the content starts with type-like symbols (i, f, b, s, n, u, *, %, ?)
    /// or contains ->, treat as type annotation.
    /// </summary>
    private AstNode ParseBracketForm()
    {
        // Check if this looks like a type annotation
        if (LooksLikeTypeAnnotation())
            return ParseTypeAnnotation();
        return ParseArrayLiteral();
    }

    private bool LooksLikeTypeAnnotation()
    {
        // Save position and look ahead
        int savedPos = _pos;
        // Skip opening [
        _pos++;

        bool isType = false;

        while (_pos < _tokens.Count && _tokens[_pos].Kind != TokenKind.RBracket && _tokens[_pos].Kind != TokenKind.Eof)
        {
            var tok = _tokens[_pos];
            if (tok.Kind == TokenKind.Arrow)
            {
                isType = true;
                break;
            }
            if (tok.Kind == TokenKind.Symbol)
            {
                var lex = tok.Lexeme;
                if (lex is "i" or "f" or "b" or "s" or "n" or "u" or "*" or "%" or "?")
                {
                    isType = true;
                    break;
                }
            }
            // If we see a number, string, or list form inside brackets, it's an array literal
            if (tok.Kind is TokenKind.IntLiteral or TokenKind.FloatLiteral or TokenKind.StrLiteral
                or TokenKind.LParen or TokenKind.LBracket or TokenKind.LBrace)
            {
                isType = false;
                break;
            }
            _pos++;
        }

        _pos = savedPos;
        return isType;
    }

    private AstNode ParseArrayLiteral()
    {
        var start = Current.Span;
        Expect(TokenKind.LBracket, "array literal");

        var items = new List<AstNode>();
        while (Current.Kind != TokenKind.RBracket && Current.Kind != TokenKind.Eof)
        {
            if (_errorCount >= MaxErrors) { SynchronizeTo(TokenKind.RBracket); break; }
            var node = ParseForm();
            if (node != null)
                items.Add(node);
        }

        var endTok = Expect(TokenKind.RBracket, "array literal close");
        // Desugar to (A item1 item2 ...)
        return new ListForm("A", items, new SourceSpan(start.File, start.Start, endTok.Span.End));
    }

    private AstNode ParseMapLiteral()
    {
        var start = Current.Span;
        Expect(TokenKind.LBrace, "map literal");

        var items = new List<AstNode>();
        while (Current.Kind != TokenKind.RBrace && Current.Kind != TokenKind.Eof)
        {
            if (_errorCount >= MaxErrors) { SynchronizeTo(TokenKind.RBrace); break; }
            var node = ParseForm();
            if (node != null)
                items.Add(node);
        }

        var endTok = Expect(TokenKind.RBrace, "map literal close");
        // Desugar to (H key1 val1 key2 val2 ...)
        return new ListForm("H", items, new SourceSpan(start.File, start.Start, endTok.Span.End));
    }

    private AstNode ParseListForm()
    {
        var start = Current.Span;
        Expect(TokenKind.LParen, "list form");

        if (Current.Kind == TokenKind.RParen)
        {
            var end = Advance();
            return new ListForm("D", new List<AstNode>(), new SourceSpan(start.File, start.Start, end.Span.End));
        }

        string keyword;
        if (Current.Kind == TokenKind.Symbol)
        {
            keyword = Current.Lexeme;
            Advance();
        }
        else
        {
            if (_errorCount < MaxErrors)
            {
                _diagnostics.ReportError(Current.Span, "E003", $"Expected keyword/symbol at start of list form, got {Current.Kind}");
                _errorCount++;
            }
            keyword = "?err";
        }

        var args = new List<AstNode>();
        while (Current.Kind != TokenKind.RParen && Current.Kind != TokenKind.Eof)
        {
            if (_errorCount >= MaxErrors) { SynchronizeTo(TokenKind.RParen); break; }
            var node = ParseForm();
            if (node != null)
                args.Add(node);
        }

        if (Current.Kind == TokenKind.RParen)
        {
            var endTok = Advance();
            return new ListForm(keyword, args, new SourceSpan(start.File, start.Start, endTok.Span.End));
        }
        else
        {
            // Error recovery: missing closing paren
            if (_errorCount < MaxErrors)
            {
                _diagnostics.ReportError(Current.Span, "E001", "Expected ) to close list form");
                _errorCount++;
            }
            return new ListForm(keyword, args, new SourceSpan(start.File, start.Start, Current.Span.End));
        }
    }

    private TypeAnnotation ParseTypeAnnotation()
    {
        var start = Current.Span;
        Expect(TokenKind.LBracket, "type annotation");

        var types = new List<TypeNode>();
        TypeNode? returnType = null;

        while (Current.Kind != TokenKind.RBracket && Current.Kind != TokenKind.Eof)
        {
            if (Current.Kind == TokenKind.Arrow)
            {
                Advance();
                returnType = ParseTypeExpr();
                break;
            }
            types.Add(ParseTypeExpr());
        }

        var endTok = Expect(TokenKind.RBracket, "type annotation close");
        return new TypeAnnotation(types, returnType, new SourceSpan(start.File, start.Start, endTok.Span.End));
    }

    private TypeNode ParseTypeExpr()
    {
        var start = Current.Span;

        if (Current.Kind == TokenKind.Symbol)
        {
            var lexeme = Current.Lexeme;

            switch (lexeme)
            {
                case "?":
                {
                    Advance();
                    var inner = ParseTypeExpr();
                    return new OptionalType(inner, new SourceSpan(start.File, start.Start, inner.Span.End));
                }
                case "*":
                {
                    Advance();
                    var elem = ParseTypeExpr();
                    return new ArrayType(elem, new SourceSpan(start.File, start.Start, elem.Span.End));
                }
                case "%":
                {
                    Advance();
                    var key = ParseTypeExpr();
                    var val = ParseTypeExpr();
                    return new MapType(key, val, new SourceSpan(start.File, start.Start, val.Span.End));
                }
                case "i" or "f" or "b" or "s" or "n" or "u":
                {
                    Advance();
                    return new PrimitiveType(lexeme, start);
                }
                default:
                {
                    Advance();
                    return new NamedType(lexeme, start);
                }
            }
        }

        if (_errorCount < MaxErrors)
        {
            _diagnostics.ReportError(Current.Span, "E004", $"Expected type expression, got {Current.Kind}");
            _errorCount++;
        }
        Advance();
        return new PrimitiveType("u", start);
    }
}
