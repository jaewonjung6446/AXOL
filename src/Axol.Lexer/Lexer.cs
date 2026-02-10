using Axol.Core;
using Axol.Core.Tokens;

namespace Axol.Lexer;

public sealed class AxolLexer
{
    private readonly string _source;
    private readonly string _file;
    private int _pos;

    public string Source => _source;

    public AxolLexer(string source, string file = "<stdin>")
    {
        _source = source;
        _file = file;
        _pos = 0;
    }

    public List<Token> Tokenize()
    {
        var tokens = new List<Token>();
        while (true)
        {
            var tok = NextToken();
            tokens.Add(tok);
            if (tok.Kind == TokenKind.Eof || tok.Kind == TokenKind.Error)
                break;
        }
        return tokens;
    }

    public Token NextToken()
    {
        SkipWhitespaceAndComments();

        if (_pos >= _source.Length)
            return Token.Eof(Span(_pos, _pos));

        var start = _pos;
        var ch = _source[_pos];

        switch (ch)
        {
            case '(':
                _pos++;
                return new Token(TokenKind.LParen, "(", null, Span(start, _pos));
            case ')':
                _pos++;
                return new Token(TokenKind.RParen, ")", null, Span(start, _pos));
            case '[':
                _pos++;
                return new Token(TokenKind.LBracket, "[", null, Span(start, _pos));
            case ']':
                _pos++;
                return new Token(TokenKind.RBracket, "]", null, Span(start, _pos));
            case '{':
                _pos++;
                return new Token(TokenKind.LBrace, "{", null, Span(start, _pos));
            case '}':
                _pos++;
                return new Token(TokenKind.RBrace, "}", null, Span(start, _pos));
            case '"':
                return ReadString();
            case '-' when _pos + 1 < _source.Length && _source[_pos + 1] == '>':
                _pos += 2;
                return new Token(TokenKind.Arrow, "->", null, Span(start, _pos));
            default:
                if (ch == '-' && _pos + 1 < _source.Length && char.IsDigit(_source[_pos + 1]))
                    return ReadNumber();
                if (char.IsDigit(ch))
                    return ReadNumber();
                return ReadSymbol();
        }
    }

    private Token ReadString()
    {
        var start = _pos;
        _pos++; // skip opening "
        var sb = new System.Text.StringBuilder();

        while (_pos < _source.Length && _source[_pos] != '"')
        {
            if (_source[_pos] == '\\' && _pos + 1 < _source.Length)
            {
                _pos++;
                switch (_source[_pos])
                {
                    case 'n': sb.Append('\n'); break;
                    case 't': sb.Append('\t'); break;
                    case '\\': sb.Append('\\'); break;
                    case '"': sb.Append('"'); break;
                    default: sb.Append('\\'); sb.Append(_source[_pos]); break;
                }
                _pos++;
            }
            else
            {
                sb.Append(_source[_pos]);
                _pos++;
            }
        }

        if (_pos >= _source.Length)
            return Token.Error("Unterminated string", Span(start, _pos));

        _pos++; // skip closing "
        var lexeme = _source[start.._pos];
        return new Token(TokenKind.StrLiteral, lexeme, sb.ToString(), Span(start, _pos));
    }

    private Token ReadNumber()
    {
        var start = _pos;
        if (_source[_pos] == '-') _pos++;

        while (_pos < _source.Length && char.IsDigit(_source[_pos]))
            _pos++;

        bool isFloat = false;
        if (_pos < _source.Length && _source[_pos] == '.' && _pos + 1 < _source.Length && char.IsDigit(_source[_pos + 1]))
        {
            isFloat = true;
            _pos++; // skip dot
            while (_pos < _source.Length && char.IsDigit(_source[_pos]))
                _pos++;
        }

        var lexeme = _source[start.._pos];

        if (isFloat)
        {
            var val = double.Parse(lexeme, System.Globalization.CultureInfo.InvariantCulture);
            return new Token(TokenKind.FloatLiteral, lexeme, val, Span(start, _pos));
        }
        else
        {
            var val = long.Parse(lexeme);
            return new Token(TokenKind.IntLiteral, lexeme, val, Span(start, _pos));
        }
    }

    private Token ReadSymbol()
    {
        var start = _pos;
        while (_pos < _source.Length && !IsDelimiter(_source[_pos]))
            _pos++;

        var lexeme = _source[start.._pos];

        // Handle "..." rest patterns
        if (lexeme.EndsWith("..."))
        {
            return new Token(TokenKind.Symbol, lexeme, null, Span(start, _pos));
        }

        if (lexeme == "true")
            return new Token(TokenKind.Symbol, lexeme, true, Span(start, _pos));
        if (lexeme == "false")
            return new Token(TokenKind.Symbol, lexeme, false, Span(start, _pos));
        if (lexeme == "nil")
            return new Token(TokenKind.Symbol, lexeme, null, Span(start, _pos));

        return new Token(TokenKind.Symbol, lexeme, null, Span(start, _pos));
    }

    private void SkipWhitespaceAndComments()
    {
        while (_pos < _source.Length)
        {
            if (char.IsWhiteSpace(_source[_pos]))
            {
                _pos++;
            }
            else if (_source[_pos] == ';')
            {
                // Skip line comment
                while (_pos < _source.Length && _source[_pos] != '\n')
                    _pos++;
            }
            else
            {
                break;
            }
        }
    }

    private static bool IsDelimiter(char ch) =>
        char.IsWhiteSpace(ch) || ch == '(' || ch == ')' || ch == '[' || ch == ']'
        || ch == '{' || ch == '}' || ch == '"' || ch == ';';

    private SourceSpan Span(int start, int end) => new(_file, start, end);
}
