using Axol.Core.Tokens;
using Axol.Lexer;
using Xunit;

namespace Axol.Lexer.Tests;

public class LexerTests
{
    private static List<Token> Lex(string source) => new AxolLexer(source).Tokenize();

    [Fact]
    public void Parentheses()
    {
        var tokens = Lex("()");
        Assert.Equal(TokenKind.LParen, tokens[0].Kind);
        Assert.Equal(TokenKind.RParen, tokens[1].Kind);
        Assert.Equal(TokenKind.Eof, tokens[2].Kind);
    }

    [Fact]
    public void Brackets()
    {
        var tokens = Lex("[i -> i]");
        Assert.Equal(TokenKind.LBracket, tokens[0].Kind);
        Assert.Equal(TokenKind.Symbol, tokens[1].Kind);
        Assert.Equal("i", tokens[1].Lexeme);
        Assert.Equal(TokenKind.Arrow, tokens[2].Kind);
        Assert.Equal(TokenKind.Symbol, tokens[3].Kind);
        Assert.Equal(TokenKind.RBracket, tokens[4].Kind);
    }

    [Fact]
    public void IntegerLiteral()
    {
        var tokens = Lex("42");
        Assert.Equal(TokenKind.IntLiteral, tokens[0].Kind);
        Assert.Equal(42L, tokens[0].Value);
    }

    [Fact]
    public void NegativeInteger()
    {
        var tokens = Lex("-7");
        Assert.Equal(TokenKind.IntLiteral, tokens[0].Kind);
        Assert.Equal(-7L, tokens[0].Value);
    }

    [Fact]
    public void FloatLiteral()
    {
        var tokens = Lex("3.14");
        Assert.Equal(TokenKind.FloatLiteral, tokens[0].Kind);
        Assert.Equal(3.14, tokens[0].Value);
    }

    [Fact]
    public void StringLiteral()
    {
        var tokens = Lex("\"hello world\"");
        Assert.Equal(TokenKind.StrLiteral, tokens[0].Kind);
        Assert.Equal("hello world", tokens[0].Value);
    }

    [Fact]
    public void StringEscapes()
    {
        var tokens = Lex("\"line1\\nline2\"");
        Assert.Equal("line1\nline2", tokens[0].Value);
    }

    [Fact]
    public void Symbols()
    {
        var tokens = Lex("f v m ? X + <= fib apply_dmg");
        Assert.All(tokens.Take(tokens.Count - 1), t => Assert.Equal(TokenKind.Symbol, t.Kind));
        Assert.Equal("f", tokens[0].Lexeme);
        Assert.Equal("v", tokens[1].Lexeme);
        Assert.Equal("m", tokens[2].Lexeme);
        Assert.Equal("?", tokens[3].Lexeme);
        Assert.Equal("X", tokens[4].Lexeme);
        Assert.Equal("+", tokens[5].Lexeme);
        Assert.Equal("<=", tokens[6].Lexeme);
        Assert.Equal("fib", tokens[7].Lexeme);
        Assert.Equal("apply_dmg", tokens[8].Lexeme);
    }

    [Fact]
    public void FibonacciTokens()
    {
        var tokens = Lex("(f fib [i -> i] n 0)");
        var expected = new[]
        {
            TokenKind.LParen, TokenKind.Symbol, TokenKind.Symbol,
            TokenKind.LBracket, TokenKind.Symbol, TokenKind.Arrow, TokenKind.Symbol, TokenKind.RBracket,
            TokenKind.Symbol, TokenKind.IntLiteral, TokenKind.RParen, TokenKind.Eof
        };
        Assert.Equal(expected.Length, tokens.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], tokens[i].Kind);
    }

    [Fact]
    public void BoolAndNil()
    {
        var tokens = Lex("true false nil");
        Assert.Equal("true", tokens[0].Lexeme);
        Assert.Equal("false", tokens[1].Lexeme);
        Assert.Equal("nil", tokens[2].Lexeme);
    }

    [Fact]
    public void Arrow()
    {
        var tokens = Lex("->");
        Assert.Equal(TokenKind.Arrow, tokens[0].Kind);
        Assert.Equal("->", tokens[0].Lexeme);
    }

    [Fact]
    public void MinusAsSymbol()
    {
        // standalone minus (not followed by digit)
        var tokens = Lex("(- x 1)");
        Assert.Equal(TokenKind.LParen, tokens[0].Kind);
        Assert.Equal(TokenKind.Symbol, tokens[1].Kind);
        Assert.Equal("-", tokens[1].Lexeme);
    }
}
